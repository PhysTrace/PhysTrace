from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]   # videotrace_v3/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.core.gt import GT_CSV, load_gt, load_gt_keys  # noqa: E402

PAIRED_SIDES = ('good', 'bad')   # Possible, Impossible, as the trials store them
IMPOSSIBLE_SIDES = ('bad',)
POSSIBLE_SIDES = ('good',)

csv.field_size_limit(sys.maxsize)   # segment_json holds whole prompts and raw outputs


def _hit_band(window: dict, gt: dict, touch: bool) -> bool:
    """The demo's candidate interval holding the window meets the GT [onset, end]; `touch` lets a shared edge count."""
    if touch:
        return window['band_start'] <= gt['end_sec'] and window['band_end'] >= gt['onset_sec']
    return window['band_start'] < gt['end_sec'] and window['band_end'] > gt['onset_sec']


def load_trials(result_dir: Path, sides: tuple[str, ...]) -> list[dict]:
    """Every stored trial run on exactly `sides`, one per pair (the latest when a pair was run twice)."""
    latest: dict[str, dict] = {}
    with (result_dir / 'trials.csv').open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            trial = json.loads(row['trial_json'])
            if tuple(trial['params'].get('sides', ())) != sides:
                continue
            if row['group'] not in latest or trial['created_at'] > latest[row['group']]['created_at']:
                latest[row['group']] = trial
    return sorted(latest.values(), key=lambda trial: trial['group'])


def load_paired_trials(result_dir: Path) -> list[dict]:
    """Each pair's trial on both sides: one run on both, else its Impossible-only and Possible-only trials joined."""
    paired = {trial['group']: trial for trial in load_trials(result_dir, PAIRED_SIDES)}
    impossible = {trial['group']: trial for trial in load_trials(result_dir, IMPOSSIBLE_SIDES)}
    possible = {trial['group']: trial for trial in load_trials(result_dir, POSSIBLE_SIDES)}
    for group in (impossible.keys() & possible.keys()) - paired.keys():
        result_ids = dict(impossible[group]['result_ids'], good=possible[group]['result_ids']['good'])
        paired[group] = dict(impossible[group], result_ids=result_ids)
    return sorted(paired.values(), key=lambda trial: trial['group'])


def load_windows(result_dir: Path, result_ids: set[str]) -> dict[str, list[dict]]:
    """The scored windows of each wanted result, in window order."""
    windows: dict[str, list[dict]] = {result_id: [] for result_id in result_ids}
    with (result_dir / 'results.csv').open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['result_id'] not in windows:
                continue
            segment = json.loads(row['segment_json'])
            windows[row['result_id']].append(dict(
                index=segment['index'], start_sec=segment['start_sec'], end_sec=segment['end_sec'],
                score=segment['prediction_score'], is_anomaly=bool(segment['is_anomaly'])))
    if missing := sorted(result_id for result_id, rows in windows.items() if not rows):
        raise ValueError(f'{len(missing)} results named in trials.csv are not in {result_dir}/results.csv, '
                         f'e.g. {missing[0]}')
    for rows in windows.values():
        rows.sort(key=lambda window: window['index'])
    return windows


def lowest_window(windows: list[dict]) -> dict:
    """The lowest-score window after the first, ties to the earlier one (the demo's rule)."""
    scored = [window for window in windows if window['index'] > 0 and window['score'] is not None]
    return min(scored, key=lambda window: (window['score'], window['index']))


def _band(windows: list[dict], window: dict) -> list[float]:
    """The demo's red interval of a lone candidate: from where the window before it ends to where it ends."""
    return [max(window['start_sec'], windows[window['index'] - 1]['end_sec']), window['end_sec']]


def judge_pair(trial: dict, windows: dict[str, list[dict]], gt: dict | None, touch: bool = True) -> dict:
    """The lowest Impossible window's hit and, when Possible ran, whether Possible's lowest is above it."""
    bad = windows[trial['result_ids']['bad']]
    bad_min = lowest_window(bad)
    band = _band(bad, bad_min)
    pair = trial['pair']
    row = dict(group=trial['group'], attribute=pair['attribute'], difficulty=pair.get('difficulty'),
               camera=pair.get('camera'), gt=None,
               candidates=[dict(index=bad_min['index'], start_sec=bad_min['start_sec'], end_sec=bad_min['end_sec'],
                                score=bad_min['score'], band=band)],
               impossible_min=bad_min['score'])
    if 'good' in trial['result_ids']:
        good_min = lowest_window(windows[trial['result_ids']['good']])
        row.update(possible_min=good_min['score'], pair_correct=good_min['score'] > bad_min['score'])
    if gt is not None:
        row['gt'] = dict(onset_sec=gt['onset_sec'], end_sec=gt['end_sec'], kind=gt['kind'],
                         confirmed=gt['confirmed'])
        row['hits'] = dict(band=dict(min=_hit_band(dict(bad_min, band_start=band[0], band_end=band[1]), gt, touch)))
    return row


def _count_hits(rows: list[dict], paired: bool) -> dict:
    """Correct counts and totals of one group of pairs."""
    with_gt = [row for row in rows if row['gt'] is not None]
    counts = dict(pairs=len(rows), with_gt=len(with_gt), band_min=sum(row['hits']['band']['min'] for row in with_gt))
    if paired:
        counts['pair_correct'] = sum(row['pair_correct'] for row in rows)
    return counts


def summarize(rows: list[dict], paired: bool) -> dict:
    """Tallies per condition and over all pairs, plus the split by whether the GT was confirmed."""
    attributes = sorted({row['attribute'] for row in rows})
    with_gt = [row for row in rows if row['gt'] is not None]
    return dict(
        by_attribute={attribute: _count_hits([row for row in rows if row['attribute'] == attribute], paired)
                      for attribute in attributes},
        total=_count_hits(rows, paired),
        by_confirmed={label: _count_hits([row for row in with_gt if row['gt']['confirmed'] == flag], paired)
                      for label, flag in (('confirmed', True), ('unconfirmed', False))})


def evaluate_run(result_dir: Path, gt_csv: Path, touch: bool = True, paired: bool = False) -> dict:
    """Every pair's answers and their hit counts, with the run's settings; only the pairs `gt_csv` lists count."""
    keys = load_gt_keys(gt_csv)
    sides = PAIRED_SIDES if paired else IMPOSSIBLE_SIDES
    stored = load_paired_trials(result_dir) if paired else load_trials(result_dir, IMPOSSIBLE_SIDES)
    trials = [trial for trial in stored if (trial['pair']['attribute'], trial['group']) in keys]
    if not trials:
        raise ValueError(f'No trial of {result_dir} run on {list(sides)} is listed in {gt_csv}; '
                         f'check --gt-csv, --result-dir and --paired')
    windows = load_windows(result_dir, {result_id for trial in trials for result_id in trial['result_ids'].values()})
    gt = load_gt(gt_csv)
    rows = [judge_pair(trial, windows, gt.get((trial['pair']['attribute'], trial['group'])), touch)
            for trial in trials]
    params = trials[0]['params']
    return dict(run=dict(result_dir=str(result_dir.relative_to(REPO_ROOT) if result_dir.is_relative_to(REPO_ROOT)
                                        else result_dir),
                         clip_model=params.get('clip_model'), clip_frames=params.get('clip_frames'),
                         mllm_backend=params.get('mllm_backend'), window_seconds=params['window_seconds'],
                         hop_seconds=params['hop_seconds'], run_k=params['prompts']['lowest_windows'],
                         gt_csv=gt_csv.name, edge_touch_is_hit=touch, paired=paired),
                summary=summarize(rows, paired), pairs=rows)


def _gt_source(text: str) -> tuple[str, Path]:
    """A `--gt-csv` value: `LABEL=PATH`, or a bare path labelled with its file name."""
    label, _, path = text.rpartition('=')
    return label or Path(path).name, Path(path)


def main(argv: list[str] | None = None) -> None:
    """Write the report as JSON; several GT files are listed in the order given."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--result-dir', type=Path, required=True, help='a run_test.py result directory')
    parser.add_argument('--gt-csv', type=_gt_source, action='append',
                        help=f'LABEL=PATH or PATH, repeatable, first one first (default: {GT_CSV.name})')
    parser.add_argument('--strict-edges', action='store_true',
                        help='a band that only touches a GT edge is not a hit (default: it is)')
    parser.add_argument('--paired', action='store_true',
                        help='score pairs with both sides, adding Possible vs Impossible: trials run on both, or '
                             'Impossible-only and --possible-only trials joined by pair (default: Impossible only)')
    parser.add_argument('--out', type=Path,
                        help='output stem (default: <result-dir>/eval_gt, plus _paired with --paired and _strict '
                             'with --strict-edges)')
    args = parser.parse_args(argv)
    result_dir = args.result_dir.expanduser().resolve()
    stem = args.out or result_dir / ('eval_gt' + '_paired' * args.paired + '_strict' * args.strict_edges)
    sources = args.gt_csv or [(GT_CSV.name, GT_CSV)]

    reports = [dict(evaluate_run(result_dir, path, touch=not args.strict_edges, paired=args.paired), label=label)
               for label, path in sources]
    # One GT keeps the single-report JSON earlier runs wrote; several are listed in the order given.
    stored = reports[0] if len(reports) == 1 else reports
    stem.with_suffix('.json').write_text(json.dumps(stored, ensure_ascii=False, indent=1), encoding='utf-8')
    for report in reports:
        total = report['summary']['total']
        paired_part = f" · 2 {total['pair_correct']}" if args.paired else ''
        print(f"{stem}.json [{report['label']}] · pairs {total['pairs']} (GT {total['with_gt']}) · "
              f"1-2 band {total['band_min']}{paired_part}")


if __name__ == '__main__':
    main()
