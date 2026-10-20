"""PhysTrace's result in the shape the page reads and the store keeps.

The one place that translates the analysis's "windows" into the page's "segments".
"""
from __future__ import annotations

SCHEMA_VERSION = 6
# window field -> what the page calls it
RENAMED = {'score': 'prediction_score',
           'history_source_window': 'history_source_segment'}
SCORE_SCOPE = 'max_over_this_window_prediction_captions_vs_this_window'


def _sampled_frames(row: dict, predictor, scorer, mllm_frames: int) -> dict:
    """Which original frames each model looked at, for the filmstrip."""
    length = row['end_frame'] - row['start_frame']
    return {'mllm': [row['start_frame'] + index for index in predictor.sample_indices(length, mllm_frames)],
            'clip': [row['start_frame'] + index for index in scorer.sample_indices(length)]}


def _segment(row: dict, predictor, scorer, mllm_frames: int) -> dict:
    """One window as the page's segment row."""
    segment = {RENAMED.get(key, key): value for key, value in row.items()}
    return dict(segment, status='trusted' if row['trusted'] else 'untrusted',
                sampled_frames=_sampled_frames(row, predictor, scorer, mllm_frames))


def for_page(result: dict, predictor, scorer) -> dict:
    """Rewrite one analyzed video into what the page draws and the store saves."""
    intervals = [dict(interval, segments=interval['windows'])
                 for interval in result['anomaly_intervals']]
    return {
        'schema_version': SCHEMA_VERSION,
        'video': result['video'],
        'fps': result['fps'],
        'num_frames': result['num_frames'],
        'duration_sec': result['duration_sec'],
        # The page reads the first, older records carry the second; they are the same number.
        'segment_seconds': result['window_seconds'],
        'window_seconds': result['window_seconds'],
        'hop_seconds': result['hop_seconds'],
        'prompts': result['prompts'],
        'clip_cache_hit': result['clip_cache_hit'],
        'prediction_score_scope': SCORE_SCOPE,
        # How the candidates were picked: the k lowest CLIP scores. Absent before 09-30, when a window
        # the model marked untrusted was the candidate.
        'anomaly_rule': result.get('anomaly_rule'),
        'has_anomaly': result['has_anomaly'],
        # The run's own count: the page's frames per window may differ from the predictor's default.
        'segments': [_segment(row, predictor, scorer, result['models']['mllm_frames']) for row in result['windows']],
        'anomaly_intervals': intervals,
        'trusted_history': result['trusted_history'],
        # The same history as the model's chat template renders it: role tokens instead of the
        # {"role", "content"} records the code passes around. Rendered with the run's own processor.
        'trusted_history_text': predictor.processor.apply_chat_template(result['trusted_history'], tokenize=False),
        'model_input_system': result['model_input_system'],
    }
