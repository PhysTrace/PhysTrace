"""Standalone InternVideo2-CLIP-6B scorer with checkpoint-backed initialization."""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import importlib.machinery
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
import types
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LlamaTokenizer

try:
    from transformers.initialization import no_init_weights
except ImportError:                                  # transformers < 5.0 keeps it in modeling_utils
    from transformers.modeling_utils import no_init_weights

_VENDOR_DIR = Path('AlignmentModels/InternVideo2/multi_modality')
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

logger = logging.getLogger(__name__)

MODEL_NAME = 'InternVideo2-CLIP-6B'

VISION_FILE = 'internvideo2-s2_6b-224p-f4_with_audio_encoder.pt'
TEXT_TOWER_FILE = 'internvl_c_13b_224px.pth'
CLIP_DELTA_FILE = '6B_clip.pth'
TOKENIZER_FILES = ['tokenizer.model', 'tokenizer_config.json', 'special_tokens_map.json']
CLIP_MODEL_DIR = Path('AlignmentModels/weights/InternVideo2-CLIP')
# The vision checkpoint is the stage-2 one, shared with the stage-2 scorer rather than copied (29 GB).
VISION_DIR = Path('AlignmentModels/weights/InternVideo2-Stage2')
_MIN_SANE_VOCAB = 1000
_TOKENIZER_WAIT_S = 120
_MAX_REPORTED_KEYS = 10

LLAMA_7B_CONFIG = {
    'architectures': ['LlamaForCausalLM'],
    'model_type': 'llama',
    'hidden_size': 4096,
    'intermediate_size': 11008,
    'num_attention_heads': 32,
    'num_hidden_layers': 32,
    'num_key_value_heads': 32,
    'max_position_embeddings': 2048,
    'rms_norm_eps': 1e-6,
    'initializer_range': 0.02,
    'hidden_act': 'silu',
    'tie_word_embeddings': False,
    'torch_dtype': 'float16',
    'bos_token_id': 1,
    'eos_token_id': 2,
    'pad_token_id': 0,
}

# The only flash_attn names the vendor modules import at module level.
_FLASH_ATTN_STUBS = {
    'flash_attn': (),
    'flash_attn.modules': (),
    'flash_attn.modules.mlp': ('FusedMLP',),
    'flash_attn.ops': (),
    'flash_attn.ops.rms_norm': ('DropoutAddRMSNorm',),
    'flash_attn.flash_attn_interface': ('flash_attn_varlen_qkvpacked_func',),
    'flash_attn.bert_padding': ('unpad_input', 'pad_input'),
}


def _raises_unavailable(name: str):
    """A stand-in for a flash_attn symbol that refuses to run."""
    def fail(*args, **kwargs):
        raise RuntimeError(f'{name} requires flash_attn, which is not installed; '
                           'this model was built for the naive attention path')

    return fail


def _flash_attn_available() -> bool:
    """True if flash_attn is really installed."""
    try:
        import flash_attn  # noqa: F401, PLC0415 - probing whether it is installed
    except ImportError:
        return False
    return True


HAS_FLASH_ATTN = _flash_attn_available()


@contextmanager
def _flash_attn_stubs():
    """Make the flash_attn names importable, for the vendor import below and nothing else.

    The vendor modules import FusedMLP, DropoutAddRMSNorm and the flash kernels at module
    level even when the model is configured not to use them, so without flash_attn (an
    environment built around vLLM's torch pin, for example) they cannot be imported at all.
    The stubs only have to exist while that happens: with the flags off nothing calls them,
    and the imported modules keep their own references afterwards.

    They are removed again on the way out, because `sys.modules` is process-wide. Left in
    place they tell every other library that flash_attn is installed -- vLLM believes it,
    takes its flash-attn path and dies on `flash_attn.ops.triton`, which no stub provides.
    """
    installed = []
    for name, members in _FLASH_ATTN_STUBS.items():
        if name in sys.modules:
            continue   # a real flash_attn, or someone else's stub: leave it alone
        module = types.ModuleType(name)
        # A spec lets `from flash_attn.x import y` resolve. No __file__ and no __version__
        # keeps transformers' own probe answering "not installed", which is the truth.
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        for member in members:
            setattr(module, member, _raises_unavailable(member))
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


def _import_vendor_modules() -> tuple:
    """The vendored InternVideo2 classes, imported with the stubs alive only for the import."""
    with _flash_attn_stubs():
        vision = importlib.import_module('models.backbones.internvideo2.internvideo2_clip_vision')
        clip = importlib.import_module('models.internvideo2_clip')
        easydict = importlib.import_module('utils.easydict')
    return vision, clip.InternVideo2_CLIP, easydict.EasyDict


internvideo2_clip_vision, InternVideo2_CLIP, EasyDict = _import_vendor_modules()


def file_identity(path) -> dict:
    """Path, size and mtime: enough to notice a checkpoint swapped underneath a cache."""
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


_ALLOWED_UNEXPECTED = re.compile(r'^text_encoder\.transformer\.layers\.\d+\.self_attn\.rotary_emb\.inv_freq$')
_TEXT_COLLAPSE_COSINE = 0.97


def _local_checkpoint_path(filename: str, root: Path = CLIP_MODEL_DIR) -> str:
    """Resolve one checkpoint file to its local path, under `root` (the CLIP model directory by default).

    Raises:
        FileNotFoundError: If the file has not been staged locally.
    """
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f'Local CLIP file missing: {path}')
    return str(path)


def _wait_for_tokenizer(tokenizer_model: Path) -> None:
    """Wait for a tokenizer another shard may still be staging.

    Raises:
        RuntimeError: If it has not appeared within `_TOKENIZER_WAIT_S` seconds.
    """
    for _ in range(_TOKENIZER_WAIT_S):
        if tokenizer_model.exists():
            return
        time.sleep(1)
    raise RuntimeError(f"{tokenizer_model} did not appear within {_TOKENIZER_WAIT_S}s -- another shard's "
                       f'download probably failed. Remove {tokenizer_model.parent} and rerun a single shard first.')


def _write_llama_config(llama_dir: Path, config_path: Path) -> None:
    """Synthesize the LlamaConfig the text tower needs, once.

    Raises:
        RuntimeError: If the tokenizer loads an implausibly small vocabulary, which would
            poison every later run that reads the written config.
    """
    tokenizer = LlamaTokenizer.from_pretrained(str(llama_dir), local_files_only=True, legacy=False)
    vocab_size = len(tokenizer)
    if vocab_size < _MIN_SANE_VOCAB:
        raise RuntimeError(f'tokenizer at {llama_dir} loaded only {vocab_size} tokens (expected >= '
                           f'{_MIN_SANE_VOCAB}); refusing to write a config that would poison later runs.')
    config = dict(LLAMA_7B_CONFIG, vocab_size=vocab_size)

    # Atomic replacement prevents other workers from reading a partial config.
    tmp_path = config_path.with_name(f'config.json.{os.getpid()}.tmp')
    tmp_path.write_text(json.dumps(config, indent=2))
    os.replace(tmp_path, config_path)
    logger.info('wrote synthesized LlamaConfig (vocab_size=%d) to %s', vocab_size, config_path)


def _prepare_llama_dir() -> str:
    """Prepare the tokenizer and LLaMA config; text weights come from InternVL.

    Raises:
        FileNotFoundError: If a tokenizer file has not been staged locally.
    """
    llama_dir = CLIP_MODEL_DIR / 'tokenizer'
    for filename in TOKENIZER_FILES:
        if not (llama_dir / filename).is_file():
            raise FileNotFoundError(f'Local tokenizer missing: {llama_dir / filename}')

    _wait_for_tokenizer(llama_dir / 'tokenizer.model')

    config_path = llama_dir / 'config.json'
    if not config_path.exists():
        _write_llama_config(llama_dir, config_path)
    return str(llama_dir)


def _fetch_internvideo2clip_6b() -> dict:
    """Resolve every checkpoint and the tokenizer to a local path."""
    return {
        'vision_ckpt_path': _local_checkpoint_path(VISION_FILE, VISION_DIR),
        'text_ckpt_path': _local_checkpoint_path(TEXT_TOWER_FILE),
        'extra_ckpt_path': _local_checkpoint_path(CLIP_DELTA_FILE),
        'llama_path': _prepare_llama_dir(),
    }



def sample_indices(total: int, num_frames: int) -> list[int]:
    """Interval midpoints over `total` frames; short clips repeat the last index.

    Raises:
        ValueError: If there is no frame to sample.
    """
    if total == 0:
        raise ValueError('got 0 frames')
    num_intervals = min(num_frames, total)
    bounds = np.linspace(start=0, stop=total, num=num_intervals + 1).astype(int)
    indices = [int(bounds[i] + bounds[i + 1] - 1) // 2 for i in range(num_intervals)]
    return indices + [indices[-1]] * (num_frames - len(indices))


def downsample_frames(frames: list, num_frames: int) -> list:
    """Sample interval midpoints; pad short clips with the last sampled frame."""
    return [frames[index] for index in sample_indices(len(frames), num_frames)]


def frames_to_tensor(frames: list, transform, device: str = 'cuda') -> torch.Tensor:
    """Convert BGR frames to a normalized [1, T, C, H, W] tensor."""
    rgb = np.stack([np.ascontiguousarray(frame[:, :, ::-1]) for frame in frames])
    tube = torch.from_numpy(rgb).permute(0, 3, 1, 2)
    return transform(tube).unsqueeze(0).to(device)


def _build_config(paths: dict, num_frames: int = 8) -> EasyDict:
    """Build the official 6B zero-shot config with local checkpoint paths."""
    return EasyDict({
        'model': {
            'vision_encoder': {
                'name': 'internvideo2_6B',
                'in_chans': 3,
                'patch_size': 14,
                'img_size': 224,
                'qkv_bias': False,
                'drop_path_rate': 0.35,
                'head_drop_path_rate': 0.0,
                'embed_dim': 3200,
                'num_heads': 25,
                'mlp_ratio': 4,
                'init_values': 0.1,
                'qk_normalization': True,
                'depth': 48,
                'use_flash_attn': HAS_FLASH_ATTN,
                'use_fused_rmsnorm': HAS_FLASH_ATTN,
                'use_fused_mlp': HAS_FLASH_ATTN,
                'fused_mlp_heuristic': 1,
                'drop_cls_token': False,
                'attn_pool_num_heads': 16,
                'clip_embed_dim': 768,
                'layerscale_no_force_fp32': True,
                'num_frames': num_frames,
                'tubelet_size': 1,
                'sep_pos_embed': False,
                'use_checkpoint': False,
                'checkpoint_num': 0,
            },
            'text_encoder': {
                'use_flash_attn': HAS_FLASH_ATTN,
                'transformer_width': 4096,
                'llama_path': paths['llama_path'],
                'use_lora': True,
            },
            'temp': 1 / 100.0,
            'temp_min': 1 / 100.0,
            'freeze_vision': True,
            'open_vision_clip_projector': True,
            'freeze_text': True,
            'open_text_projection': False,
            'open_text_lora': False,
            'tokenizer_path': paths['llama_path'],
            'vision_ckpt_path': paths['vision_ckpt_path'],
            'load_vision_ckpt_from_internvideo2_stage2': True,
            'vision_ckpt_t_size': 4,
            'text_ckpt_path': paths['text_ckpt_path'],
            'extra_ckpt_path': paths['extra_ckpt_path'],
        }
    })


def _verify_load(model: InternVideo2_CLIP) -> None:
    """Raise unless every model parameter/buffer received a checkpoint value.

    Raises:
        RuntimeError: If any key is missing, unexpectedly present, or covered by no
            checkpoint at all.
    """
    result = model.load_result
    bad_unexpected = [key for key in result.unexpected_keys if not _ALLOWED_UNEXPECTED.match(key)]
    if result.missing_keys or bad_unexpected:
        raise RuntimeError(
            f'checkpoint mismatch: missing={result.missing_keys[:_MAX_REPORTED_KEYS]} '
            f'(+{max(0, len(result.missing_keys) - _MAX_REPORTED_KEYS)} more), '
            f'unexpected={bad_unexpected[:_MAX_REPORTED_KEYS]} '
            f'(+{max(0, len(bad_unexpected) - _MAX_REPORTED_KEYS)} more)')
    model_keys = set(model.state_dict().keys())
    not_covered = sorted(model_keys - getattr(model, 'aligned_keys', model.loaded_keys))
    if not_covered:
        raise RuntimeError(f'{len(not_covered)} model keys not covered by any checkpoint: '
                           f'{not_covered[:_MAX_REPORTED_KEYS]}')
    logger.info('checkpoint load verified: %d keys, %d allowed-unexpected (rotary inv_freq)',
                len(model_keys), len(result.unexpected_keys))


def _align_lora_keys(model, checkpoint: dict) -> dict:
    """Rename checkpoint keys onto peft's wrapper layout when the installed peft needs it.

    peft >= 0.6 keeps a LoRA-wrapped Linear's original weight under `<module>.base_layer.weight`,
    while the 6B CLIP checkpoint was written with peft 0.4, which stored it as `<module>.weight`.
    Only keys the model does not already expect are moved, so an old peft changes nothing.
    """
    expected = set(model.state_dict())
    aligned, moved = {}, 0
    for key, value in checkpoint.items():
        if key not in expected:
            parent, _, leaf = key.rpartition('.')
            wrapped = f'{parent}.base_layer.{leaf}'
            if wrapped in expected:
                key, moved = wrapped, moved + 1
        aligned[key] = value
    if moved:
        logger.info('moved %d checkpoint keys onto peft base_layer wrappers', moved)
    return aligned


@contextmanager
def lora_key_compat():
    """Make the vendor's `load_state_dict` call speak whichever peft layout is installed."""
    original = InternVideo2_CLIP.load_state_dict

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        aligned = _align_lora_keys(self, state_dict)
        # _verify_load checks coverage against the names actually loaded, not the file's.
        self.aligned_keys = set(aligned)
        return original(self, aligned, strict=strict, **kwargs)

    with patch.object(InternVideo2_CLIP, 'load_state_dict', load_state_dict):
        yield


@contextmanager
def no_weight_init():
    """Skip checkpoint-replaced initialization; patches are process-wide."""
    def keep_tensor(tensor, *args, **kwargs):
        return tensor

    # The vision module holds its own imported reference to timm's initializer.
    with no_init_weights(), patch.object(internvideo2_clip_vision, 'trunc_normal_', keep_tensor):
        yield


class InternVideo2CLIP6BScorer:
    """Encode videos/text and compute cosine similarity with InternVideo2-CLIP-6B."""

    def __init__(self, device: str = 'cuda', num_frames: int = 8, dtype: torch.dtype = torch.bfloat16):
        print(f'[{MODEL_NAME}]')

        paths = _fetch_internvideo2clip_6b()

        self.device = device
        self.num_frames = num_frames
        self._text_cache: dict[str, torch.Tensor] = {}

        self.config = _build_config(paths, num_frames=num_frames)
        with no_weight_init(), lora_key_compat():
            self.model = InternVideo2_CLIP(config=self.config, is_pretrain=False)
        _verify_load(self.model)
        # Cast before the move: `.to(device).to(dtype)` would first put the fp32 weights
        # on the GPU (47 GiB) and, after casting, leave that space parked in the caching
        # allocator - memory the next model on the same card cannot use. Casting on the
        # CPU is bitwise identical and lands only the 23.5 GiB the model actually needs.
        self.model = self.model.to(dtype).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self._check_text_tower()
        self.feature_cache_identity = self._build_cache_identity(paths, dtype)

    def _build_cache_identity(self, paths: dict, dtype: torch.dtype) -> dict:
        """Everything a cached feature depends on; a feature that no longer matches is re-extracted."""
        return dict(
            model=MODEL_NAME, feature_dim=self.config.model.vision_encoder.clip_embed_dim,
            num_frames=self.num_frames, dtype=str(dtype), torch_version=torch.__version__,
            preprocessing=repr(self.model.transform),
            sampler='interval-midpoints-pad-last-v1', encoder_version=1,
            # Only recorded when it deviates, so caches extracted with flash attention
            # (every one written so far) keep their identity and stay valid.
            **({} if HAS_FLASH_ATTN else {'attention': 'naive'}),
            weights={name: file_identity(paths[name]) for name in ('vision_ckpt_path', 'extra_ckpt_path')},
        )

    @property
    def text_tokenizer(self) -> Any:
        """The tokenizer the text tower reads, for checking caption length."""
        return self.model.tokenizer.tokenizer

    def clear_text_cache(self) -> None:
        """Drop every cached text embedding."""
        self._text_cache.clear()

    @torch.no_grad()
    def _check_text_tower(self) -> None:
        """Verify the text tower separates two unrelated captions.

        A tokenizer staged with the wrong BOS/EOS/padding embeds anything almost identically,
        which would make every score meaningless rather than wrong in an obvious way.

        Raises:
            RuntimeError: If two unrelated captions embed almost identically.
        """
        dog = self.encode_text('A dog is running on the grass')
        cube = self.encode_text('A red cube is falling into a box')
        cosine = (dog @ cube.T).item()
        self.clear_text_cache()
        if cosine > _TEXT_COLLAPSE_COSINE:
            raise RuntimeError(f"text tower looks collapsed: cosine('dog', 'cube') = {cosine:.4f} "
                               f'> {_TEXT_COLLAPSE_COSINE} (check tokenizer BOS/EOS/padding)')
        logger.info('text tower sanity: cosine(unrelated prompts) = %.4f', cosine)

    @torch.no_grad()
    def encode_frames(self, frames: list) -> torch.Tensor:
        """Already-decoded BGR frames (e.g. a slice of a longer video) -> L2-normalized [1, D]."""
        sampled = downsample_frames(frames, self.num_frames)
        tensor = frames_to_tensor(sampled, self.model.transform, device=self.device)
        feature = self.model.encode_vision(tensor)
        return F.normalize(feature.float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """Returns an L2-normalized [1, D] embedding for the text (cached per-instance)."""
        if text not in self._text_cache:
            token_ids = self.model.tokenizer([text]).to(self.device)
            feature = self.model.encode_text(token_ids)
            self._text_cache[text] = F.normalize(feature.float(), dim=-1)
        return self._text_cache[text]
