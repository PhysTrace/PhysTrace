"""The violation GT of each IntPhys2 pair, read from the CSV demo/gt/build_gt_csv.py writes.

The page draws it on the Impossible chart only: it is there to judge a run, never an input to it.
"""
from __future__ import annotations

import csv
from pathlib import Path

GT_CSV = Path(__file__).resolve().parents[2] / 'IntPhys2Timing' / 'intphys2_gt.csv'
GT_FIELDS = ['attribute', 'group', 'category', 'scene', 'variant', 'impossible_video', 'possible_video',
             'fps', 'frame_count', 'duration_sec', 'onset_sec', 'end_sec', 'onset_frame', 'end_frame',
             'kind', 'source', 'confirmed', 'description', 'in_category_subset']


def _read_rows(path: Path) -> list[dict]:
    """Every row of a GT CSV, after checking that it has the columns this module reads."""
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != GT_FIELDS:
            raise ValueError(f'{path} has columns {reader.fieldnames}; expected {GT_FIELDS}')
        return list(reader)


def load_gt_keys(path: Path = GT_CSV) -> set[tuple[str, str]]:
    """The (attribute, group) of every pair the CSV lists, labelled or not: it is what a run covers."""
    if not path.is_file():
        raise FileNotFoundError(f'GT CSV not found: {path}; pass --gt-csv with an existing file')
    return {(row['attribute'], row['group']) for row in _read_rows(path)}


def load_gt(path: Path = GT_CSV) -> dict[tuple[str, str], dict]:
    """GT by (attribute, group); pairs without a labelled onset are left out."""
    if not path.is_file():
        return {}
    rows = [row for row in _read_rows(path) if row['onset_sec']]
    return {(row['attribute'], row['group']): dict(
        onset_sec=float(row['onset_sec']), end_sec=float(row['end_sec']),
        onset_frame=int(row['onset_frame']), end_frame=int(row['end_frame']),
        kind=row['kind'], source=row['source'], confirmed=row['confirmed'] == 'yes',
        description=row['description']) for row in rows}
