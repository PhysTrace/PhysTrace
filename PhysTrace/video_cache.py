"""Every window's CLIP feature, cached as one npz per video and window geometry.

A cache whose identity (video, geometry, encoder) does not match this run is re-extracted, never trusted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from PhysTrace.video_windows import Window

_NORM_TOLERANCE = 1e-4


def default_cache_dir(video_dir) -> Path:
    """The feature cache next to the videos: `.../Main/Videos` -> `.../Main/videos-npz`."""
    video_dir = Path(video_dir)
    return video_dir.parent / f'{video_dir.name.lower()}-npz'


def cache_path(video, window_seconds: float, hop_seconds: float, root, tag: str = '') -> Path:
    """Where one video's features for this geometry and encoder live."""
    # The identity alone would keep two encoders from trusting each other's features, but
    # sharing a file name they would overwrite each other's cache on every switch.
    stem = f'{Path(video).stem}_{tag}' if tag else Path(video).stem
    return Path(root) / f'{stem}_w{window_seconds:g}s_h{hop_seconds:g}s.npz'


def identity(video, windows: list[Window], encoder: dict,
             window_seconds: float, hop_seconds: float) -> str:
    """Everything a cached feature depends on, as a stable json string."""
    path = Path(video).resolve()
    stat = path.stat()
    return json.dumps(dict(video=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                           window_seconds=float(window_seconds), hop_seconds=float(hop_seconds),
                           windows=[[window.start_frame, window.end_frame] for window in windows],
                           encoder=encoder), sort_keys=True)


def _normalized(features, count: int) -> bool:
    """True when `features` is a finite, L2-normalized float32 block of `count` rows."""
    return bool(features is not None and features.ndim == 2 and len(features) == count
                and features.dtype == np.float32 and np.isfinite(features).all()
                and np.allclose(np.linalg.norm(features, axis=1), 1, atol=_NORM_TOLERANCE))


def load(path: Path, expected_identity: str, count: int) -> np.ndarray | None:
    """The cached features for exactly this run, or None when there are none to trust."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if str(archive['identity'].item()) != expected_identity:
                return None
            features = archive['feats']
    except (OSError, KeyError, ValueError, EOFError):
        # A missing, truncated or outdated archive is simply re-extracted.
        return None
    return features if _normalized(features, count) else None


def save(path: Path, features: np.ndarray, run_identity: str) -> None:
    """Write one geometry's features through an open file; np.savez would append a second '.npz' to a path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, feats=features, identity=np.array(run_identity))
    temporary.replace(path)


def load_or_extract(video, frames: list, windows: list[Window], encoder,  # noqa: PLR0913
                    window_seconds: float, hop_seconds: float, root) -> tuple[np.ndarray, bool]:
    """Features for every window, from the cache when it fits and from the encoder otherwise, and whether cached."""
    path = cache_path(video, window_seconds, hop_seconds, root, tag=encoder.cache_tag)
    run_identity = identity(video, windows, encoder.identity(), window_seconds, hop_seconds)

    cached = load(path, run_identity, len(windows))
    if cached is not None:
        return cached, True

    features = np.stack([encoder.encode_window(window.frames_of(frames))
                         for window in windows]).astype(np.float32)
    if not _normalized(features, len(windows)):
        raise ValueError('CLIP must return one finite L2-normalized feature per window')
    save(path, features, run_identity)
    return features, False
