"""The inference loop: one MLLM call and one CLIP score per window, no freeze.

The model's "trusted" only decides which windows join the history; the CLIP score alone picks the candidates.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from PhysTrace.video_cache import load_or_extract
from PhysTrace.video_models.mllm import Ask, Prediction, VideoPredictor
from PhysTrace.video_models.mllm_prompts import (
    NO_HISTORY,
    PromptConfig,
    history_block,
    history_line,
    system_prompt,
    window_request,
)
from PhysTrace.video_windows import Window, decode_video, split_windows


class TrustedHistory:
    """Two histories: every window observed, and the trusted ones the model is shown.

    A window is remembered by its state alone; `windows` and `trust_all` are PromptConfig's fields of those names.
    """

    def __init__(self, system_prompt: str, windows: int = 0, trust_all: bool = False):
        self.system = system_prompt
        self.windows = windows
        self.trust_all = trust_all
        self.observed: list[str] = []
        self.trusted: list[str] = []
        self.window: Window | None = None            # the last trusted window
        self.last_observed: Window | None = None     # the last window of all

    @property
    def blind(self) -> bool:
        """Whether the model is shown no earlier window at all."""
        return self.windows == NO_HISTORY

    @property
    def shown(self) -> list[str]:
        """The history later windows draw from: every window with `trust_all`, else the trusted ones."""
        return self.observed if self.trust_all else self.trusted

    @property
    def source(self) -> Window | None:
        """The latest window the model is shown as history; None when it is shown none."""
        if self.blind:
            return None
        return self.last_observed if self.trust_all else self.window

    def context(self) -> str:
        """The latest `windows` windows of `shown` as request text; 0 keeps all of them, NO_HISTORY none."""
        if self.blind:
            return ''
        return history_block(self.shown[-self.windows:] if self.windows else self.shown)

    def record(self, window: Window, answer: Prediction) -> None:
        """Every window is observed; only a trusted one joins `trusted`."""
        line = history_line(window, answer.state)
        self.observed.append(line)
        self.last_observed = window
        if not answer.trusted:
            return
        self.trusted.append(line)
        self.window = window

    def transcript(self) -> list[dict]:
        """The system turn and every window of `shown`, as the page shows what the model was given."""
        shown = '' if self.blind else history_block(self.shown)
        return [{'role': 'system', 'content': self.system}] + ([{'role': 'user', 'content': shown}] if shown else [])


@dataclass
class _Walk:
    """One pass over a video's windows."""

    frames: list
    fps: float
    windows: list[Window]
    features: np.ndarray
    predictor: VideoPredictor
    scorer: object
    prompts: PromptConfig
    ask: Ask

    def run(self, progress: Callable[[dict], None] | None = None) -> tuple[list[dict], TrustedHistory]:
        """Walk every window and return one row each, plus the histories they produced."""
        history = TrustedHistory(system_prompt(self.prompts), self.prompts.history_windows, self.prompts.trust_all)
        rows = []
        for window in self.windows:
            source = history.source
            request = window_request(window, self.prompts)
            shown = history.context()
            answer, raw_outputs, model_inputs = self.predictor.describe(
                window.frames_of(self.frames), self.fps, request, history.system, shown, ask=self.ask)
            history.record(window, answer)
            caption_scores = self._score_prediction(window, answer.prediction)
            row = self._row(window, answer, request, raw_outputs, caption_scores, source)
            row['score'] = max(caption_scores)
            # What the engine was given past the system turn, attempt by attempt, in the chat template's own text.
            rows.append(dict(row, model_inputs=model_inputs))
            if progress:
                progress(dict(rows[-1], total=len(self.windows)))
        return rows, history

    def _score_prediction(self, window: Window, captions: list[str]) -> list[float]:
        """The cosine of each of this window's own prediction captions against its video."""
        feature = self.features[window.index]
        return [self.scorer.score(feature, caption) for caption in captions]

    @staticmethod
    def _row(window: Window, answer: Prediction, request: str,  # noqa: PLR0913
             raw_outputs: list[str], caption_scores: list[float], source: Window | None) -> dict:
        """One window's record: what was asked, what came back, and how its prediction scored."""
        return dict(
            index=window.index, start_frame=window.start_frame, end_frame=window.end_frame,
            start_sec=window.start_sec, end_sec=window.end_sec,
            **answer.as_dict(),
            input_prompt=request, raw_outputs=raw_outputs,
            # The best-matching caption is the window's score.
            score=max(caption_scores), prediction_scores=caption_scores,
            # The latest trusted window the model was shown as history.
            history_source_window=source.index if source else None,
            # Set once every window is scored; see flag_lowest.
            is_anomaly=False)


def ask_for(prompts: PromptConfig, num_frames: int | None = None) -> Ask:
    """What every call of a run asks for: its caption count, its generation limits and its frames per window."""
    return Ask(prompts.num_prompts, prompts.max_new_tokens, prompts.thinking_budget, num_frames)


def flag_lowest(rows: list[dict], count: int) -> None:
    """Mark the `count` windows with the lowest CLIP scores as the physics-error candidates.

    The first window is never one (nothing precedes it); ties go to the earlier window.
    """
    scored = sorted((row for row in rows if row['index'] > 0), key=lambda row: (row['score'], row['index']))
    lowest = {row['index'] for row in scored[:count]}
    for row in rows:
        row['is_anomaly'] = row['index'] in lowest


def anomaly_intervals(rows: list[dict]) -> list[dict]:
    """Merge the flagged windows into intervals that start where the violation can first be seen.

    A flagged window after an unflagged one starts where that one ends, as the overlap was already judged normal.
    """
    intervals: list[dict] = []
    for position, row in enumerate(rows):
        if not row['is_anomaly']:
            continue
        previous = rows[position - 1] if position else None
        if previous is not None and previous['is_anomaly']:
            intervals[-1].update(end_frame=row['end_frame'], end_sec=row['end_sec'])
            intervals[-1]['windows'].append(row['index'])
            continue
        start_frame, start_sec = row['start_frame'], row['start_sec']
        if previous is not None:
            start_frame, start_sec = max(start_frame, previous['end_frame']), max(start_sec, previous['end_sec'])
        intervals.append(dict(start_frame=start_frame, end_frame=row['end_frame'],
                              start_sec=start_sec, end_sec=row['end_sec'], windows=[row['index']]))
    return intervals


def analyze_video(video, predictor: VideoPredictor, scorer, *,  # noqa: PLR0913
                  window_seconds: float = 1.0, hop_seconds: float = 0.5,
                  prompts: PromptConfig,
                  cache_root,
                  progress: Callable[[dict], None] | None = None) -> dict:
    """Analyze one video: cut it into windows, describe each, score the predictions, flag the lowest."""
    video = Path(video).expanduser().resolve()
    if not video.is_file():
        raise ValueError(f'Video file not found: {video}')
    # Checked before decoding, so an out-of-range limit fails at once and not at window 1.
    mllm_frames = predictor.frames_per_window(window_seconds, prompts.mllm_frames)
    ask = ask_for(prompts, mllm_frames)
    max_new_tokens, thinking_budget = predictor.limits(ask)

    frames, fps = decode_video(video)
    windows = split_windows(len(frames), fps, window_seconds, hop_seconds)
    features, cached = load_or_extract(video, frames, windows, scorer,
                                       window_seconds, hop_seconds, cache_root)
    rows, history = _Walk(frames, fps, windows, features, predictor, scorer, prompts, ask).run(progress)
    flag_lowest(rows, prompts.lowest_windows)
    intervals = anomaly_intervals(rows)
    return dict(
        video=str(video), fps=fps, num_frames=len(frames), duration_sec=len(frames) / fps,
        window_seconds=window_seconds, hop_seconds=hop_seconds,
        prompts=asdict(prompts), clip_cache_hit=cached,
        models=dict(mllm_backend=predictor.backend.name, mllm=predictor.model_path,
                    mllm_frames=mllm_frames, max_new_tokens=max_new_tokens,
                    thinking_budget=thinking_budget, max_model_len=predictor.max_model_len,
                    temperature=predictor.temperature, seed=predictor.sampling.seed,
                    clip=scorer.identity()),
        anomaly_rule=dict(lowest_windows=prompts.lowest_windows),
        has_anomaly=bool(intervals), anomaly_intervals=intervals, windows=rows,
        trusted_history=history.transcript(),
        # The system turn every window's `model_inputs` follows, stored once per video.
        model_input_system=predictor.render_system(system_prompt(prompts)))
