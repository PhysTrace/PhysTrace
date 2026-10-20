"""Standalone PE-AV-Large scorer: its video and video-text branches only, driven as `pe_av_similarity.py` does.

By default every frame of a window is encoded; the demo's "CLIP input frames" samples that many instead.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import importlib.machinery
import importlib.util
import logging
import os
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# The copied tree is imported as the top-level package `core`, the name its own absolute imports use.
_VENDOR_DIR = Path('AlignmentModels/PE_AV_large')
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

logger = logging.getLogger(__name__)

MODEL_NAME = 'PE-AV-Large'
CHECKPOINT_REPO = 'facebook/pe-av-large@perception_models'
CHECKPOINT_DIR = Path('AlignmentModels/weights/PE-AV-Large')
WEIGHTS_FILE = 'model.safetensors'
MAX_TEXT_TOKENS = 512           # PEAudioVisualTransform's own truncation length
_TEXT_COLLAPSE_COSINE = 0.97


def file_identity(path) -> dict:
    """Path, size and mtime: enough to notice a checkpoint swapped underneath a cache."""
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _raises_unavailable(name: str):
    """A stand-in for a symbol of a package that is not installed and refuses to run."""
    def fail(*args, **kwargs):
        raise RuntimeError(f'{name} is a stub: its package is not installed in this environment')

    return fail


def _missing_attribute(name: str):
    """Module-level __getattr__ that names the stub instead of a bare AttributeError."""
    def getattr_(attribute: str):
        raise RuntimeError(f'{name}.{attribute} is a stub: its package is not installed in this environment')

    return getattr_


def _build_stub_modules() -> dict[str, types.ModuleType]:
    """The names the copied tree imports at module level but never calls on the video-text path."""
    attention_bias = type('AttentionBias', (), {'__doc__': 'xformers stub: only annotations and isinstance use it.'})
    fmha = types.ModuleType('xformers.ops.fmha')
    fmha.__getattr__ = _missing_attribute('xformers.ops.fmha')
    members = {
        'xformers': {},
        'xformers.ops': {'AttentionBias': attention_bias, 'fmha': fmha},
        'torchcodec': {},
        'torchcodec.decoders': {'AudioDecoder': _raises_unavailable('torchcodec.decoders.AudioDecoder'),
                                'VideoDecoder': _raises_unavailable('torchcodec.decoders.VideoDecoder')},
    }
    modules = {}
    for name, attributes in members.items():
        module = types.ModuleType(name)
        # A spec lets `from x.y import z` resolve; no __file__ keeps availability probes honest.
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        modules[name] = module
    return modules


@contextmanager
def _missing_package_stubs() -> Iterator[None]:
    """Make xformers and torchcodec importable for the vendor import, and remove the stubs afterwards.

    `sys.modules` is process-wide: leftover stubs would tell vLLM that xformers is installed.
    """
    really_installed = {top for top in ('xformers', 'torchcodec') if importlib.util.find_spec(top) is not None}
    installed = []
    for name, module in _build_stub_modules().items():
        if name in sys.modules or name.partition('.')[0] in really_installed:
            continue
        sys.modules[name] = module
        installed.append(name)
        parent, _, leaf = name.rpartition('.')
        if parent:
            setattr(sys.modules[parent], leaf, module)
    try:
        yield
    finally:
        for name in reversed(installed):
            sys.modules.pop(name, None)


def _import_vendor_classes() -> tuple:
    """PEAudioVisual and its transform, imported with the stubs alive only for the import."""
    with _missing_package_stubs():
        package = importlib.import_module('core.audio_visual_encoder')
    return package.PEAudioVisual, package.PEAudioVisualTransform


PEAudioVisual, PEAudioVisualTransform = _import_vendor_classes()


def _checkpoint_dir() -> Path:
    """The local PE-AV checkpoint directory."""
    path = Path(os.environ.get('PHYSTRACE_PE_AV_DIR') or CHECKPOINT_DIR)
    for filename in ('config.json', WEIGHTS_FILE, 'tokenizer.json'):
        if not (path / filename).is_file():
            raise FileNotFoundError(f'PE-AV checkpoint file missing: {path / filename}; download '
                                    f'{CHECKPOINT_REPO} there or set PHYSTRACE_PE_AV_DIR')
    return path


def _drop_audio_branch(model) -> None:
    """Free the audio tower and the audio-visual fusion, which video-text scoring never runs (1.4B of 2.3B params)."""
    model.audio_visual_model.audio_model = None
    model.audio_visual_model.audio_visual_transformer = None


def sample_indices(total: int, num_frames: int | None = None) -> list[int]:
    """Which frames of a `total`-frame window are encoded: all of them, or `num_frames` evenly spaced.

    Spacing includes the first and last frame; a window with no more than `num_frames` frames is read whole.
    """
    if total == 0:
        raise ValueError('got 0 frames')
    if num_frames is not None and num_frames < 1:
        raise ValueError(f'num_frames must be at least 1, or None for every frame; got {num_frames}')
    if num_frames is None or num_frames >= total:
        return list(range(total))
    if num_frames == 1:
        return [(total - 1) // 2]
    return [int(index * (total - 1) / (num_frames - 1)) for index in range(num_frames)]


class PEAVLargeScorer:
    """Encode videos/text and compute cosine similarity with PE-AV-Large."""

    def __init__(self, device: str = 'cuda', num_frames: int | None = None):
        """Load the checkpoint; `num_frames` None reads every frame of a window (see sample_indices)."""
        if num_frames is not None and num_frames < 1:
            raise ValueError(f'num_frames must be at least 1, or None for every frame; got {num_frames}')
        frames = 'all frames' if num_frames is None else f'{num_frames} frames per window'
        print(f'[{MODEL_NAME}]')
        print(f'  video + text (PE-AV large, {frames}): {CHECKPOINT_REPO}')

        checkpoint_dir = _checkpoint_dir()
        self.device = device
        self.num_frames = num_frames
        self._text_cache: dict[str, torch.Tensor] = {}

        # As the reference script runs it: fp32 weights, bf16 autocast at inference.
        self.model = PEAudioVisual.from_config(str(checkpoint_dir), pretrained=True)
        _drop_audio_branch(self.model)
        self.model = self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.transform = PEAudioVisualTransform.from_config(str(checkpoint_dir))

        self._check_text_tower()
        self.feature_cache_identity = dict(
            model=MODEL_NAME, feature_dim=self.model.config.output_dim,
            frames='all' if num_frames is None else num_frames,
            dtype='float32+autocast-bfloat16', torch_version=torch.__version__,
            preprocessing=repr(self.transform.video_processor.frame_transform),
            # uniform-v1 repeated frames to fill a window shorter than num_frames; v2 reads it whole.
            sampler='all-frames-v1' if num_frames is None else 'uniform-v2', encoder_version=1,
            weights={'checkpoint': file_identity(checkpoint_dir / WEIGHTS_FILE)},
        )

    @property
    def text_tokenizer(self) -> Any:
        """The tokenizer the text tower reads, for checking caption length."""
        return self.transform.tokenizer

    def clear_text_cache(self) -> None:
        """Drop every cached text embedding."""
        self._text_cache.clear()

    @torch.no_grad()
    def _check_text_tower(self) -> None:
        """Verify the text tower separates two unrelated captions."""
        dog = self.encode_text('A dog is running on the grass')
        cube = self.encode_text('A red cube is falling into a box')
        cosine = (dog @ cube.T).item()
        self.clear_text_cache()
        if cosine > _TEXT_COLLAPSE_COSINE:
            raise RuntimeError(f"text tower looks collapsed: cosine('dog', 'cube') = {cosine:.4f} "
                               f'> {_TEXT_COLLAPSE_COSINE} (check the checkpoint and tokenizer)')
        logger.info('text tower sanity: cosine(unrelated prompts) = %.4f', cosine)

    @torch.no_grad()
    def encode_frames(self, frames: list) -> torch.Tensor:
        """Already-decoded BGR frames (e.g. a slice of a longer video) -> L2-normalized [1, D]."""
        if not frames:
            raise ValueError('got 0 frames')
        frames = [frames[index] for index in sample_indices(len(frames), self.num_frames)]
        rgb = np.stack([np.ascontiguousarray(frame[:, :, ::-1]) for frame in frames])
        # [T, C, H, W] uint8: the layout torchcodec hands the same transform.
        video = torch.from_numpy(rgb).permute(0, 3, 1, 2)
        inputs = self.transform(videos=[video]).to(self.device)
        with torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16):
            feature = self.model.encode_video(inputs['pixel_values_videos'], inputs['padding_mask_videos'])
        # The contrastive heads are not normalized; cosine needs both sides on the unit sphere.
        return F.normalize(feature.float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """Returns an L2-normalized [1, D] embedding for the text (cached per-instance)."""
        if text not in self._text_cache:
            inputs = self.transform(text=[text]).to(self.device)
            with torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16):
                feature = self.model.encode_video_text(inputs['input_ids'], inputs['attention_mask'])
            self._text_cache[text] = F.normalize(feature.float(), dim=-1)
        return self._text_cache[text]
