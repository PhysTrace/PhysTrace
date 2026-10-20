"""Read a video and cut it into the windows the analysis walks."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import cv2


@dataclass(frozen=True)
class Window:
    """One half-open frame interval [start_frame, end_frame) and the seconds it spans."""

    index: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        """How many seconds the window spans."""
        return self.end_sec - self.start_sec

    def frames_of(self, frames: list) -> list:
        """This window's slice of a decoded video."""
        return frames[self.start_frame:self.end_frame]


def decode_video(path) -> tuple[list, float]:
    """Every frame (BGR uint8) of a video, and its FPS."""
    path = Path(path).expanduser().resolve()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open video: {path}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise ValueError(f'Video decoded to no frames: {path}')
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f'Video reports an unusable FPS ({fps}): {path}')
    return frames, fps


def split_windows(num_frames: int, fps: float, window_seconds: float, hop_seconds: float) -> list[Window]:
    """Windows of `window_seconds` starting every `hop_seconds`, up to the one that ends the video.

    Times are rounded per window, so a fractional FPS never accumulates drift.
    """
    if num_frames < 1:
        raise ValueError('Video contains no decoded frames')
    if not math.isfinite(window_seconds) or fps * window_seconds < 1:
        raise ValueError('Window length must cover at least one frame')
    if not math.isfinite(hop_seconds) or not 0 < hop_seconds <= window_seconds or fps * hop_seconds < 1:
        raise ValueError('hop_seconds must cover at least one frame and be <= the window length')

    windows: list[Window] = []
    end = 0
    while end < num_frames:
        index = len(windows)
        start = math.floor(index * hop_seconds * fps + 0.5)
        end = min(num_frames, math.floor((index * hop_seconds + window_seconds) * fps + 0.5))
        windows.append(Window(index, start, end, start / fps, end / fps))
    return windows
