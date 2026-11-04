"""The IntPhys2 Possible/Impossible pairs the browser lists.

The GT CSV names each pair's videos and metadata. The demo lists only complete pairs, because a lone clip supports
no comparison; scoring can do with Impossible alone.
"""
from __future__ import annotations

from pathlib import Path
import sys

from demo.core.gt import GT_CSV, load_gt_rows

SIDES = {'good': 'Possible', 'bad': 'Impossible'}
# Shown as chips and sortable table headers; a subset of them also filters the list.
META_COLUMNS = ['scene', 'variant', 'category', 'env', 'occluder', 'difficulty', 'camera']
FILTER_COLUMNS = ['category', 'env', 'occluder', 'difficulty', 'camera']
# Under the IntPhys2 root the Possible/Impossible clips live in this split.
SPLIT = 'Main'
VIDEOS_SUBDIR = 'Videos'


def main_videos_dir(intphys2_dir: Path) -> Path:
    """The Main/Videos folder of an IntPhys2 root."""
    videos_dir = Path(intphys2_dir) / SPLIT / VIDEOS_SUBDIR
    if not videos_dir.is_dir():
        raise ValueError(f'{videos_dir} is not a folder; pass the IntPhys2 root (the one holding {SPLIT}/) '
                         f'as --intphys2-dir')
    return videos_dir


def _pair(row: dict, available: set[str]) -> dict:
    """One row of the pair list; `good_name` is None when the Possible video is not on disk."""
    return dict(group=row['group'], attribute=row['attribute'], scene=row['scene'], variant=row['variant'],
                category=row['category'], env=row['env'], occluder=row['occluder'],
                difficulty=row['difficulty'], camera=row['camera'],
                good_name=row['possible_video'] if row['possible_video'] in available else None,
                bad_name=row['impossible_video'])


def load_pairs(video_dir: Path, gt_csv: Path = GT_CSV, paired: bool = True) -> list[dict]:
    """The pairs `gt_csv` lists whose videos are in `video_dir`.

    Without `paired`, a pair whose Possible video is missing is kept too, with `good_name` None.
    """
    video_dir = Path(video_dir)
    available = {path.stem for path in video_dir.glob('*.mp4')}
    rows = load_gt_rows(gt_csv)
    required = ['impossible_video', 'possible_video'] if paired else ['impossible_video']
    wanted = 'complete Possible/Impossible pair' if paired else 'Impossible video'
    pairs = [_pair(row, available) for row in rows if all(row[column] in available for column in required)]
    if missing := len(rows) - len(pairs):
        print(f'[pairs] {missing} of the {len(rows)} pairs in {Path(gt_csv).name} have no {wanted} '
              f'under the video dir', file=sys.stderr)
    if not pairs:
        raise ValueError(f'No {wanted} under {video_dir}')
    pairs.sort(key=lambda pair: (pair['attribute'], pair['category'], int(pair['scene']), pair['variant']))
    return pairs
