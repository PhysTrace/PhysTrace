"""Standalone InternVideo2-Stage2-6B scorer: the stage-2 vision encoder and its own BERT text tower.

Everything comes from the one stage-2 checkpoint the CLIP scorer already reads its vision weights
from. That checkpoint also holds the BERT-large text encoder, `vision_proj` and `text_proj`, so
only the BERT tokenizer is staged beside it.

Upstream builds the text tower from its own `xbert.py`, which no longer imports under the
transformers vLLM pins. It is not needed: `encode_text` runs BERT in `mode="text"`, i.e. only the
first `fusion_layer` (19) layers, none of which has cross-attention, and takes the CLS token. That
is a stock `BertModel` with 19 layers and no pooler, whose parameter names match the checkpoint's.
The vision encoder is the vendored CLIP one: with no mask its forward (blocks, then the attention
pooling `clip_projector`) is the stage-2 `PretrainInternVideo2.forward` minus the distillation heads.
"""
from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
import re
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import BertConfig, BertModel, BertTokenizerFast

from PhysTrace.video_models.internvideo2_clip import (
    CLIP_MODEL_DIR,
    HAS_FLASH_ATTN,
    downsample_frames,
    frames_to_tensor,
    internvideo2_clip_vision,
    no_weight_init,
    sample_indices,  # noqa: F401 - ClipScorer reads the sampler from whichever loader it built
)

logger = logging.getLogger(__name__)

MODEL_NAME = 'InternVideo2-Stage2-6B'

CHECKPOINT_REPO = 'OpenGVLab/InternVideo2-Stage2_6B-224p-f4'
CHECKPOINT_FILE = 'internvideo2-s2_6b-224p-f4_with_audio_encoder.pt'
TOKENIZER_REPO = 'google-bert/bert-large-uncased'
# The checkpoint is shared with the CLIP scorer rather than copied: it is the same 29 GB file.
CHECKPOINT_DIR = CLIP_MODEL_DIR
TOKENIZER_DIR = CLIP_MODEL_DIR.parent / 'InternVideo2-Stage2' / 'bert-large-uncased'

# From the checkpoint's own training config (`model.text_encoder`, `inputs`).
TEXT_FUSION_LAYER = 19
MAX_TEXT_TOKENS = 40
CHECKPOINT_NUM_FRAMES = 4

_TEXT_PREFIX = 'text_encoder.bert.'
_TEXT_LAYER = re.compile(r'^encoder\.layer\.(\d+)\.')
# Distillation heads and the image-only positional table: training-time parts the scorer never runs.
_SKIPPED_VISION = ('clip_decoder', 'final_clip_decoder', 'clip_pos_embed', 'clip_img_pos_embed', 'img_pos_embed')
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_TEXT_COLLAPSE_COSINE = 0.97
_MAX_REPORTED_KEYS = 10

MODEL_SOURCES = {
    'vision + text (InternVideo2 stage2 6B)': f'{CHECKPOINT_REPO}/{CHECKPOINT_FILE}',
    'tokenizer (BERT-large uncased)': TOKENIZER_REPO,
}

# The 6B values of upstream `pretrain_internvideo2_6b_patch14_224`.
_VISION_KWARGS = dict(
    in_chans=3, patch_size=14, img_size=224, qkv_bias=False, drop_path_rate=0.0, head_drop_path_rate=0.0,
    embed_dim=3200, num_heads=25, mlp_ratio=4, init_values=0.00001, qk_normalization=True, depth=48,
    use_flash_attn=HAS_FLASH_ATTN, use_fused_rmsnorm=HAS_FLASH_ATTN, use_fused_mlp=HAS_FLASH_ATTN,
    fused_mlp_heuristic=1, attn_pool_num_heads=16, clip_embed_dim=768, layerscale_no_force_fp32=False,
    tubelet_size=1, sep_pos_embed=False, use_checkpoint=False, checkpoint_num=0,
)

_pos_embed = importlib.import_module('models.backbones.internvideo2.pos_embed')


def file_identity(path) -> dict:
    """Path, size and mtime: enough to notice a checkpoint swapped underneath a cache."""
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _checkpoint_path() -> Path:
    """The stage-2 checkpoint.

    Raises:
        FileNotFoundError: If it has not been staged locally.
    """
    path = Path(os.environ.get('PHYSTRACE_STAGE2_CKPT') or CHECKPOINT_DIR / CHECKPOINT_FILE)
    if not path.is_file():
        raise FileNotFoundError(f'Stage2 checkpoint missing: {path}; set PHYSTRACE_STAGE2_CKPT')
    return path


def _tokenizer_dir() -> Path:
    """The BERT tokenizer directory.

    Raises:
        FileNotFoundError: If its vocabulary has not been staged locally.
    """
    path = Path(os.environ.get('PHYSTRACE_STAGE2_TOKENIZER_DIR') or TOKENIZER_DIR)
    if not (path / 'vocab.txt').is_file():
        raise FileNotFoundError(f'BERT tokenizer missing: {path / "vocab.txt"}; '
                                f'download {TOKENIZER_REPO} there or set PHYSTRACE_STAGE2_TOKENIZER_DIR')
    return path


def _to_unit_range(tube: torch.Tensor) -> torch.Tensor:
    """uint8 pixels -> float in [0, 1]."""
    return tube.float().div(255.0)


def _build_transform(img_size: int) -> transforms.Compose:
    """The stage-2 test transform: bicubic resize, then ImageNet normalization."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.Lambda(_to_unit_range),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


class Stage2DualEncoder(nn.Module):
    """The two towers of InternVideo2-Stage2 and their contrastive projections."""

    def __init__(self, bert_config: BertConfig, num_frames: int):
        super().__init__()
        self.vision_encoder = internvideo2_clip_vision.InternVideo2(num_frames=num_frames, **_VISION_KWARGS)
        self.vision_proj = nn.Linear(_VISION_KWARGS['clip_embed_dim'], _VISION_KWARGS['clip_embed_dim'])
        self.text_encoder = BertModel(bert_config, add_pooling_layer=False)
        self.text_proj = nn.Linear(bert_config.hidden_size, _VISION_KWARGS['clip_embed_dim'])

    def encode_vision(self, tube: torch.Tensor) -> torch.Tensor:
        """[B, T, C, H, W] -> projected [B, D], not normalized."""
        pooled = self.vision_encoder(tube.permute(0, 2, 1, 3, 4), use_image=tube.shape[1] == 1)
        return self.vision_proj(pooled)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Token ids -> projected CLS embedding [B, D], not normalized."""
        hidden = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.text_proj(hidden[:, 0])


def _select_weights(checkpoint: dict) -> dict:
    """The checkpoint entries the dual encoder uses, renamed onto its modules."""
    selected = {}
    for key, value in checkpoint.items():
        if key.startswith('vision_encoder.'):
            if not any(part in key for part in _SKIPPED_VISION):
                selected[key] = value
        elif key.startswith(('vision_proj.', 'text_proj.')):
            selected[key] = value
        elif key.startswith(_TEXT_PREFIX):
            name = key[len(_TEXT_PREFIX):]
            layer = _TEXT_LAYER.match(name)
            # Layers from the fusion layer on only run in multimodal mode; position_ids is a buffer.
            if (layer is None or int(layer.group(1)) < TEXT_FUSION_LAYER) and name != 'embeddings.position_ids':
                selected[f'text_encoder.{name}'] = value
    return selected


def _load_weights(model: Stage2DualEncoder, path: Path) -> None:
    """Fill every parameter of `model` from the stage-2 checkpoint.

    Raises:
        RuntimeError: If any model key is missing from the checkpoint or vice versa.
    """
    logger.info('loading %s', path)
    # mmap keeps the 29 GB file out of RAM; the audio tower and heads are never read.
    checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)['model']
    _pos_embed.interpolate_pos_embed(checkpoint, model.vision_encoder, orig_t_size=CHECKPOINT_NUM_FRAMES)
    result = model.load_state_dict(_select_weights(checkpoint), strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f'stage2 checkpoint mismatch: missing={result.missing_keys[:_MAX_REPORTED_KEYS]} '
            f'(+{max(0, len(result.missing_keys) - _MAX_REPORTED_KEYS)} more), '
            f'unexpected={result.unexpected_keys[:_MAX_REPORTED_KEYS]} '
            f'(+{max(0, len(result.unexpected_keys) - _MAX_REPORTED_KEYS)} more)')
    logger.info('stage2 checkpoint load verified: %d keys', len(model.state_dict()))


class InternVideo2Stage2Scorer:
    """Encode videos/text and compute cosine similarity with InternVideo2-Stage2-6B."""

    def __init__(self, device: str = 'cuda', num_frames: int = 8, dtype: torch.dtype = torch.bfloat16):
        print(f'[{MODEL_NAME}]')
        for label, source in MODEL_SOURCES.items():
            print(f'  {label}: {source}')

        checkpoint_path, tokenizer_dir = _checkpoint_path(), _tokenizer_dir()
        self.device = device
        self.num_frames = num_frames
        self._text_cache: dict[str, torch.Tensor] = {}

        self.tokenizer = BertTokenizerFast.from_pretrained(str(tokenizer_dir), local_files_only=True)
        bert_config = BertConfig.from_pretrained(str(tokenizer_dir), local_files_only=True,
                                                 num_hidden_layers=TEXT_FUSION_LAYER)
        with no_weight_init():
            self.model = Stage2DualEncoder(bert_config, num_frames)
        _load_weights(self.model, checkpoint_path)
        # Cast on the CPU first, as the CLIP scorer does, so only the bf16 copy reaches the GPU.
        self.model = self.model.to(dtype).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.transform = _build_transform(_VISION_KWARGS['img_size'])

        self._check_text_tower()
        self.feature_cache_identity = dict(
            model=MODEL_NAME, feature_dim=_VISION_KWARGS['clip_embed_dim'], num_frames=num_frames,
            dtype=str(dtype), torch_version=torch.__version__, preprocessing=repr(self.transform),
            sampler='interval-midpoints-pad-last-v1', encoder_version=1,
            **({} if HAS_FLASH_ATTN else {'attention': 'naive'}),
            weights={'checkpoint': file_identity(checkpoint_path)},
        )

    @property
    def text_tokenizer(self) -> Any:
        """The tokenizer the text tower reads, for checking caption length."""
        return self.tokenizer

    def clear_text_cache(self) -> None:
        """Drop every cached text embedding."""
        self._text_cache.clear()

    @torch.no_grad()
    def _check_text_tower(self) -> None:
        """Verify the text tower separates two unrelated captions.

        Raises:
            RuntimeError: If two unrelated captions embed almost identically.
        """
        dog = self.encode_text('A dog is running on the grass')
        cube = self.encode_text('A red cube is falling into a box')
        cosine = (dog @ cube.T).item()
        self.clear_text_cache()
        if cosine > _TEXT_COLLAPSE_COSINE:
            raise RuntimeError(f"text tower looks collapsed: cosine('dog', 'cube') = {cosine:.4f} "
                               f'> {_TEXT_COLLAPSE_COSINE} (check the BERT weights and tokenizer)')
        logger.info('text tower sanity: cosine(unrelated prompts) = %.4f', cosine)

    @torch.no_grad()
    def encode_frames(self, frames: list) -> torch.Tensor:
        """Already-decoded BGR frames (e.g. a slice of a longer video) -> L2-normalized [1, D]."""
        sampled = downsample_frames(frames, self.num_frames)
        tensor = frames_to_tensor(sampled, self.transform, device=self.device)
        feature = self.model.encode_vision(tensor)
        return F.normalize(feature.float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """Returns an L2-normalized [1, D] embedding for the text (cached per-instance)."""
        if text not in self._text_cache:
            tokens = self.tokenizer([text], truncation=True, max_length=MAX_TEXT_TOKENS,
                                    return_tensors='pt').to(self.device)
            feature = self.model.encode_text(tokens['input_ids'], tokens['attention_mask'])
            self._text_cache[text] = F.normalize(feature.float(), dim=-1)
        return self._text_cache[text]

