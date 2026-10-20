"""The video-text scorer: one feature per window, one cosine per forecast caption.

The model is picked by name from `CLIP_MODELS`; this is the only interface the analysis sees.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib

import numpy as np
import torch


@dataclass(frozen=True)
class ClipModel:
    """One selectable scorer and the caption rules its text tower was trained with."""

    name: str
    loader_module: str
    loader_class: str
    max_caption_tokens: int     # the text tower truncates past this, silently
    caption_prefix: str         # what the model was evaluated with
    cache_tag: str              # joins the feature cache file name; '' keeps the original names
    reads_all_frames: bool = False   # every frame of a window, unless a caller opts in to sampling


CLIP_MODELS = {
    'pe-av-large': ClipModel(
        'PE-AV-Large', 'PhysTrace.video_models.pe_av', 'PEAVLargeScorer',
        max_caption_tokens=512, caption_prefix='', cache_tag='pe-av-l', reads_all_frames=True),
    'internvideo2-stage2-6b': ClipModel(
        'InternVideo2-Stage2-6B', 'PhysTrace.video_models.internvideo2_stage2', 'InternVideo2Stage2Scorer',
        max_caption_tokens=40, caption_prefix='', cache_tag='stage2-6b'),
    # The first scorer: its 1208 caches predate the tag, so it keeps the untagged file names.
    'internvideo2-clip-6b': ClipModel(
        'InternVideo2-CLIP-6B', 'PhysTrace.video_models.internvideo2_clip', 'InternVideo2CLIP6BScorer',
        max_caption_tokens=80, caption_prefix='summarize:', cache_tag=''),
}
DEFAULT_CLIP_MODEL = 'pe-av-large'


def _import_loader(spec: ClipModel):
    """The loader module, imported only once a scorer is built.

    Deferred on purpose: loaders install stub `flash_attn` modules that would reach vLLM's spawned engine process.
    """
    return importlib.import_module(spec.loader_module)


class ClipScorer:
    """Cosine similarity between a window's frames and a caption."""

    def __init__(self, device: str = 'cuda:0', num_frames: int = 8, model: str = DEFAULT_CLIP_MODEL, *,
                 sample_frames: bool = False):
        """Load one scorer; an all-frames model (PE-AV) ignores `num_frames` unless `sample_frames` is set."""
        if model not in CLIP_MODELS:
            raise ValueError(f'Unknown CLIP model {model!r}; choose one of {sorted(CLIP_MODELS)}')
        self.spec = CLIP_MODELS[model]
        self._loader = _import_loader(self.spec)
        loader_frames = num_frames if sample_frames or not self.spec.reads_all_frames else None
        self.model = getattr(self._loader, self.spec.loader_class)(device=device, num_frames=loader_frames)
        # The scorer's own count: None when it reads every frame (PE-AV without sample_frames).
        self.num_frames = self.model.num_frames

    @property
    def name(self) -> str:
        """The model's display name."""
        return self.spec.name

    @property
    def cache_tag(self) -> str:
        """What tells this model's feature cache files apart from another model's."""
        if self.spec.reads_all_frames and self.num_frames is not None:
            # Sampled features share the all-frames geometry, so without this they would share its
            # file and overwrite it on every switch (the identity check alone only rejects them).
            return f'{self.spec.cache_tag}-f{self.num_frames}'
        return self.spec.cache_tag

    def identity(self) -> dict:
        """What a feature produced here depends on: weights, frame count, preprocessing."""
        return dict(self.model.feature_cache_identity, num_frames=self.num_frames)

    def sample_indices(self, length: int) -> list[int]:
        """Which frames of a `length`-frame window this scorer reads (window-local)."""
        return self._loader.sample_indices(length, self.num_frames)

    def encode_window(self, frames: list) -> np.ndarray:
        """One window's BGR frames -> [D] float32, L2-normalized."""
        return self.model.encode_frames(frames).detach().cpu().numpy()[0].astype(np.float32)

    def count_tokens(self, caption: str) -> int:
        """How many tokens the text tower would spend on `caption`."""
        return len(self.model.text_tokenizer(self.spec.caption_prefix + caption, truncation=False)['input_ids'])

    def fit_caption(self, caption: str) -> str:
        """Shorten a caption by dropping words from the end until the text tower reads all of it."""
        words = caption.split()
        while words:
            candidate = ' '.join(words)
            if self.count_tokens(candidate) <= self.spec.max_caption_tokens:
                return candidate
            words.pop()
        raise ValueError(f'No prefix of the caption fits {self.spec.max_caption_tokens} tokens: {caption!r}')

    def score(self, feature: np.ndarray, caption: str) -> float:
        """Cosine in [-1, 1] between one window feature and a caption."""
        video = torch.as_tensor(feature, device=self.model.device, dtype=torch.float32).reshape(1, -1)
        try:
            return float((video @ self.model.encode_text(self.fit_caption(caption)).T).item())
        finally:
            # Every window is scored against a different caption, so the cache only grows.
            self.model.clear_text_cache()
