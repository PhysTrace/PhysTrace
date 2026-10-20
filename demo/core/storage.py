from __future__ import annotations

from collections.abc import Callable
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import uuid

csv.field_size_limit(sys.maxsize)

WINDOW_COLUMNS = ['index', 'start_sec', 'end_sec', 'prediction_score', 'state', 'prediction',
                  'trusted', 'history_source_segment']
VIDEO_FIELDS = ['result_id', 'saved_at', 'video', *WINDOW_COLUMNS, 'metadata_json', 'segment_json']
TRIAL_FIELDS = ['id', 'group', 'created_at', 'window_seconds', 'hop_seconds', 'trial_json']


def _dump(value) -> str:
    """Serialize one blob; NaN would not survive the JSON round trip."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _append(path: Path, fields: list[str], rows: list[dict]) -> None:
    """Add complete rows under an exclusive lock, so parallel runs cannot interleave."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+', encoding='utf-8', newline='') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0, os.SEEK_END)
        if stream.tell():
            stream.seek(0)
            if next(csv.reader(stream)) != fields:
                raise ValueError(f'CSV header mismatch: {path}')
            stream.seek(0, os.SEEK_END)
        writer = csv.DictWriter(stream, fieldnames=fields)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def _rows(path: Path) -> list[dict]:
    """Every row of a stored CSV, or nothing when it does not exist yet."""
    if not path.exists():
        return []
    with path.open(encoding='utf-8', newline='') as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        return list(csv.DictReader(stream))


def _remove_rows(path: Path, drop: Callable[[dict], bool]) -> list[dict]:
    """Rewrite one CSV without the rows `drop` picks and return them.

    In place, not replaced, so an appender waiting on the lock still writes to the same file.
    """
    if not path.exists():
        return []
    with path.open('r+', encoding='utf-8', newline='') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        reader = csv.DictReader(stream)
        rows = list(reader)
        dropped = [row for row in rows if drop(row)]
        if dropped:
            stream.seek(0)
            stream.truncate()
            writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(row for row in rows if not drop(row))
            stream.flush()
            os.fsync(stream.fileno())
    return dropped


def _window_rows(result: dict, result_id: str, saved_at: str) -> list[dict]:
    """One CSV row per window; only the first carries the video-level metadata blob."""
    metadata = {key: value for key, value in result.items() if key != 'segments'}
    rows = []
    for position, window in enumerate(result['segments']):
        # Caption lists as JSON, not as Python's repr of a list.
        row = {key: _dump(value) if isinstance(value := window.get(key), list) else value
               for key in WINDOW_COLUMNS}
        row.update(result_id=result_id, saved_at=saved_at, video=result.get('video', ''),
                   metadata_json=_dump(metadata) if position == 0 else '',
                   segment_json=_dump(window))
        rows.append(row)
    return rows


def save_result(result: dict, root) -> tuple[str, Path]:
    """Store one analyzed video. Returns its id and the file it went to."""
    result_id = uuid.uuid4().hex
    saved_at = datetime.now(timezone.utc).isoformat()
    path = Path(root) / 'results.csv'
    _append(path, VIDEO_FIELDS, _window_rows(result, result_id, saved_at))
    return result_id, path


def save_trial(trial: dict, result_ids: dict[str, str], root) -> None:
    """Store one pair run, referring to its videos by id instead of copying them."""
    payload = {key: value for key, value in trial.items() if key != 'results'}
    payload['result_ids'] = result_ids
    _append(Path(root) / 'trials.csv', TRIAL_FIELDS, [dict(
        id=trial['id'], group=trial['group'], created_at=trial['created_at'],
        window_seconds=trial['params']['window_seconds'],
        hop_seconds=trial['params']['hop_seconds'],
        trial_json=_dump(payload))])


def load_results(root) -> dict[str, dict]:
    """Every stored video, rebuilt from its metadata row and its window rows, in the current schema."""
    results: dict[str, dict] = {}
    for row in _rows(Path(root) / 'results.csv'):
        key = row['result_id']
        if row['metadata_json']:
            results[key] = json.loads(row['metadata_json'])
            results[key]['segments'] = []
        if key in results:
            results[key]['segments'].append(json.loads(row['segment_json']))
    return results


def load_trials(root):
    """Every stored pair run, with its videos attached."""
    results = load_results(root)
    for row in _rows(Path(root) / 'trials.csv'):
        trial = json.loads(row['trial_json'])
        ids = trial.pop('result_ids')
        trial['results'] = {side: results[key] for side, key in ids.items() if key in results}
        yield trial


def delete_trials(trial_ids: set[str], root) -> set[str]:
    """Remove stored pair runs and their videos' windows; returns the ids removed."""
    removed, result_ids = set(), set()
    for row in _remove_rows(Path(root) / 'trials.csv', lambda row: row['id'] in trial_ids):
        removed.add(row['id'])
        result_ids.update(json.loads(row['trial_json'])['result_ids'].values())
    if result_ids:
        _remove_rows(Path(root) / 'results.csv', lambda row: row['result_id'] in result_ids)
    return removed


def summarize(result: dict) -> dict:
    """What the pair list and the trial table show without opening the whole run."""
    scores = [row['prediction_score'] for row in result['segments'] if row['prediction_score'] is not None]
    return dict(min_score=min(scores) if scores else None,
                untrusted=sum(not row['trusted'] for row in result['segments']),
                candidates=sum(row['is_anomaly'] for row in result['segments']),
                has_anomaly=result['has_anomaly'], intervals=result['anomaly_intervals'])
