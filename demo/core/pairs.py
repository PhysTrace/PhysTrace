"""The IntPhys2 Possible/Impossible pairs the browser lists.

Only complete pairs are listed, because a lone clip supports no comparison.
"""
from __future__ import annotations

import csv
from pathlib import Path
import sys

from demo.core.gt import load_gt_keys

SIDES = {'good': 'Possible', 'bad': 'Impossible'}
# Shown as chips and sortable table headers; a subset of them also filters the list.
META_COLUMNS = ['scene', 'variant', 'category', 'env', 'occluder', 'difficulty', 'camera']
FILTER_COLUMNS = ['category', 'env', 'occluder', 'difficulty', 'camera']
# Under the IntPhys2 root the Possible/Impossible clips and their metadata.csv live in this split.
SPLIT = 'Main'
VIDEOS_SUBDIR = 'Videos'


def main_videos_dir(intphys2_dir: Path) -> Path:
    """The Main/Videos folder of an IntPhys2 root, whose Main/metadata.csv lists its clips."""
    videos_dir = Path(intphys2_dir) / SPLIT / VIDEOS_SUBDIR
    if not videos_dir.is_dir():
        raise ValueError(f'{videos_dir} is not a folder; pass the IntPhys2 root (the one holding {SPLIT}/) '
                         f'as --intphys2-dir')
    return videos_dir


def _group_rows(metadata: Path, available: set[str]) -> dict[str, dict]:
    """Index the metadata rows of available videos by scene, keyed by possibility."""
    groups: dict[str, dict] = {}
    with metadata.open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if row['name'] not in available:
                continue
            variant, possibility = row['type'].split('_', 1)
            if possibility in SIDES.values():
                group = f"{row['game_name']}_scene{row['SceneIndex']}_v{variant}"
                groups.setdefault(group, {})[possibility] = row
    return groups


def _pair(group: str, good: dict, bad: dict) -> dict:
    """One row of the pair list; the Possible side supplies the shared metadata."""
    return dict(group=group, attribute=good['condition'], scene=good['SceneIndex'],
                variant=good['type'].split('_', 1)[0], category=good['game_name'],
                env=good.get('env', ''), occluder=good.get('occluder', ''),
                difficulty=good.get('Difficulty', ''), camera=good.get('Camera', ''),
                good_name=good['name'], bad_name=bad['name'])


def _keep_listed(pairs: list[dict], gt_csv: Path) -> list[dict]:
    """The pairs the GT CSV lists; listed pairs without a complete pair of videos are reported, not dropped silently."""
    keys = load_gt_keys(gt_csv)
    kept = [pair for pair in pairs if (pair['attribute'], pair['group']) in keys]
    missing = len(keys) - len(kept)
    if missing:
        print(f'[pairs] {missing} of the {len(keys)} pairs in {Path(gt_csv).name} have no complete pair of videos '
              f'under the video dir', file=sys.stderr)
    return kept


def load_pairs(video_dir: Path, metadata: Path | None = None, gt_csv: Path | None = None) -> list[dict]:
    """Group metadata.csv rows into complete Possible/Impossible pairs of available videos.

    `metadata` defaults to the metadata.csv next to `video_dir`. With `gt_csv`, only the pairs that CSV lists remain.
    """
    video_dir = Path(video_dir)
    metadata = Path(metadata) if metadata else video_dir.parent / 'metadata.csv'
    available = {path.stem for path in video_dir.glob('*.mp4')}

    pairs = []
    for group, members in _group_rows(metadata, available).items():
        if set(members) != set(SIDES.values()):
            continue  # a lone clip yields no comparison
        good, bad = members['Possible'], members['Impossible']
        pairs.append(_pair(group, good, bad))
    if gt_csv:
        pairs = _keep_listed(pairs, gt_csv)
    if not pairs:
        raise ValueError(f'No complete Possible/Impossible pairs under {video_dir}')
    pairs.sort(key=lambda pair: (pair['attribute'], pair['category'], int(pair['scene']), pair['variant']))
    return pairs
