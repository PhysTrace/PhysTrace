"""Judge generated clips against their original pair by the lowest window score; writes a JSON report.

A run_test.py run that scored generated clips in the Impossible slot (`--result-dir`) is compared, pair by pair, with
runs that scored the original IntPhys2 clips under the same settings (`--reference-dir`, repeatable). Each video's
score is its lowest window (the demo's rule: the first window, which has no history, is left out). A generated clip
whose score is below the reference clip's (the pair's original Possible clip by default) is judged impossible, else
possible. When the generated clip has a meta.json next to it (the PhysTrace/gen_*.py layout), the side the generation
chose (P or I) is read too, and the report counts how often the judgement agrees with it (I -> impossible,
P -> possible).
E.g. `python score/run_compare.py --result-dir <generated run> --reference-dir <original P run> <original I run>`.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]   # videotrace_v7/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.core.gt import GT_CSV  # noqa: E402

REFERENCES = {'possible': 'possible_video', 'impossible': 'impossible_video'}   # --reference -> GT CSV column
VERDICTS = ('impossible', 'possible')
EXPECTED = {'I': 'impossible', 'P': 'possible'}   # what a judgement should say for the side the generation chose

csv.field_size_limit(sys.maxsize)   # segment_json holds whole prompts and raw outputs


def load_pair_videos(gt_csv: Path) -> dict[tuple[str, str], dict]:
    """The original Possible / Impossible file names of every pair the CSV lists, by (attribute, group)."""
    if not gt_csv.is_file():
        raise FileNotFoundError(f'GT CSV not found: {gt_csv}; pass --gt-csv with an existing file')
    with gt_csv.open(newline='', encoding='utf-8') as stream:
        return {(row['attribute'], row['group']): dict(possible_video=row['possible_video'],
                                                       impossible_video=row['impossible_video'])
                for row in csv.DictReader(stream)}


def lowest_window(windows: list[dict], all_windows: bool) -> dict | None:
    """The lowest-score window, ties to the earlier one; the first window counts only with `all_windows`."""
    scored = [window for window in windows if (all_windows or window['index'] > 0) and window['score'] is not None]
    return min(scored, key=lambda window: (window['score'], window['index'])) if scored else None


def load_results(result_dir: Path) -> dict[str, dict]:
    """Every stored result of a run: its video, when it was saved and its windows in order."""
    results: dict[str, dict] = {}
    with (result_dir / 'results.csv').open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            result = results.setdefault(row['result_id'], dict(video=row['video'], saved_at=row['saved_at'],
                                                                windows=[]))
            score = row['prediction_score']
            result['windows'].append(dict(index=int(row['index']), start_sec=float(row['start_sec']),
                                          end_sec=float(row['end_sec']), score=float(score) if score else None))
    for result in results.values():
        result['windows'].sort(key=lambda window: window['index'])
    return results


def reference_scores(result_dirs: list[Path], all_windows: bool) -> dict[str, dict]:
    """The lowest window of every original clip the reference runs scored, by file name (the latest result wins)."""
    scores: dict[str, dict] = {}
    for result_dir in result_dirs:
        for result in load_results(result_dir).values():
            name = Path(result['video']).resolve().stem
            low = lowest_window(result['windows'], all_windows)
            if low is not None and (name not in scores or result['saved_at'] > scores[name]['saved_at']):
                scores[name] = dict(low, saved_at=result['saved_at'], result_dir=str(result_dir))
    return scores


def load_generated(result_dir: Path) -> list[dict]:
    """The latest stored trial of each pair that scored the Impossible slot."""
    latest: dict[str, dict] = {}
    with (result_dir / 'trials.csv').open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            trial = json.loads(row['trial_json'])
            if 'bad' not in trial.get('result_ids', {}):
                continue
            if row['group'] not in latest or trial['created_at'] > latest[row['group']]['created_at']:
                latest[row['group']] = trial
    return sorted(latest.values(), key=lambda trial: trial['group'])


def local_path(video: str, roots: list[tuple[str, str]]) -> Path:
    """A stored video path as seen from this machine: the first `--video-root` prefix it starts with is swapped."""
    for old, new in roots:
        if video.startswith(old):
            return Path(new + video[len(old):])
    return Path(video)


def generation_choice(video: Path) -> str | None:
    """The side (P or I) the generation chose for this clip, from the meta.json next to it, if there is one."""
    path = video.resolve()
    meta = path.parent / 'meta.json'
    if not meta.is_file():
        return None
    output = json.loads(meta.read_text(encoding='utf-8')).get('outputs', {}).get(path.name, {})
    return output.get('choice')


def judge_pair(trial: dict, result: dict, videos: dict, references: dict, reference: str, all_windows: bool,
               roots: list[tuple[str, str]]) -> dict:
    """One generated clip against its pair's original clips: both scores, the judgement and the generation's choice."""
    pair = trial['pair']
    low = lowest_window(result['windows'], all_windows)
    originals = {side: references.get(videos[REFERENCES[side]]) for side in REFERENCES}
    row = dict(group=trial['group'], attribute=pair['attribute'], difficulty=pair.get('difficulty'),
               camera=pair.get('camera'), video=result['video'],
               generated_min=low['score'] if low else None, generated_window=low['index'] if low else None,
               possible_min=originals['possible']['score'] if originals['possible'] else None,
               impossible_min=originals['impossible']['score'] if originals['impossible'] else None,
               verdict=None, choice=generation_choice(local_path(result['video'], roots)), agrees=None)
    threshold = row[f'{reference}_min']
    if row['generated_min'] is not None and threshold is not None:
        row['verdict'] = 'impossible' if row['generated_min'] < threshold else 'possible'
        if row['choice'] in EXPECTED:
            row['agrees'] = row['verdict'] == EXPECTED[row['choice']]
    return row


def _count(rows: list[dict]) -> dict:
    """Judgement counts of one group of pairs, and the agreement with the generation's choice per side."""
    judged = [row for row in rows if row['verdict'] is not None]
    counts = dict(pairs=len(rows), judged=len(judged),
                  impossible=sum(row['verdict'] == 'impossible' for row in judged))
    with_choice = [row for row in judged if row['agrees'] is not None]
    if with_choice:
        counts.update(with_choice=len(with_choice), agrees=sum(row['agrees'] for row in with_choice),
                      by_choice={choice: dict(pairs=len(chosen), **{verdict: sum(row['verdict'] == verdict
                                                                                 for row in chosen)
                                                                       for verdict in VERDICTS})
                                 for choice in sorted(EXPECTED)
                                 for chosen in [[row for row in with_choice if row['choice'] == choice]]})
    return counts


def summarize(rows: list[dict]) -> dict:
    """Tallies per condition and over all pairs."""
    attributes = sorted({row['attribute'] for row in rows})
    return dict(by_attribute={attribute: _count([row for row in rows if row['attribute'] == attribute])
                              for attribute in attributes},
                total=_count(rows))


def compare_run(result_dir: Path, reference_dirs: list[Path], gt_csv: Path, reference: str = 'possible',
                all_windows: bool = False, roots: list[tuple[str, str]] = ()) -> dict:
    """Every listed pair's judgement and the tallies, with the settings; only the pairs `gt_csv` lists count."""
    videos = load_pair_videos(gt_csv)
    trials = [trial for trial in load_generated(result_dir) if (trial['pair']['attribute'], trial['group']) in videos]
    if not trials:
        raise ValueError(f'No stored trial of {result_dir} is listed in {gt_csv}; check --gt-csv and --result-dir')
    results = load_results(result_dir)
    references = reference_scores(reference_dirs, all_windows)
    rows = [judge_pair(trial, results[trial['result_ids']['bad']], videos[(trial['pair']['attribute'], trial['group'])],
                       references, reference, all_windows, list(roots))
            for trial in trials]
    if missing := [row['group'] for row in rows if row[f'{reference}_min'] is None]:
        print(f'[warn] {len(missing)} pairs have no scored original {reference} clip in the reference runs, '
              f'e.g. {missing[0]}; they are left unjudged', flush=True)
    params = trials[0]['params']
    return dict(run=dict(result_dir=str(result_dir), reference_dirs=[str(path) for path in reference_dirs],
                         reference=reference, all_windows=all_windows, clip_model=params.get('clip_model'),
                         clip_frames=params.get('clip_frames'), mllm_backend=params.get('mllm_backend'),
                         window_seconds=params['window_seconds'], hop_seconds=params['hop_seconds'],
                         gt_csv=gt_csv.name),
                summary=summarize(rows), pairs=rows)


def main(argv: list[str] | None = None) -> None:
    """Write the report as JSON and print the totals."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--result-dir', type=Path, required=True,
                        help='a run_test.py result directory of generated clips (Impossible slot)')
    parser.add_argument('--reference-dir', type=Path, nargs='+', required=True,
                        help='run_test.py result directories that scored the original clips, same settings')
    parser.add_argument('--reference', choices=sorted(REFERENCES), default='possible',
                        help="which original clip of the pair sets the threshold (default: %(default)s)")
    parser.add_argument('--all-windows', action='store_true',
                        help='let the first window count for the lowest score (default: left out, as the demo does)')
    parser.add_argument('--gt-csv', type=Path, default=GT_CSV,
                        help='the pairs and their original file names (default: %(default)s)')
    parser.add_argument('--video-root', nargs='+', default=[], metavar='OLD=NEW',
                        help='path prefixes to swap when the generated clips were scored on another machine, '
                             'to find their meta.json (e.g. /old/root=/new/root)')
    parser.add_argument('--out', type=Path, help='output file (default: <result-dir>/eval_compare.json)')
    args = parser.parse_args(argv)
    result_dir = args.result_dir.expanduser().resolve()
    report = compare_run(result_dir, [path.expanduser().resolve() for path in args.reference_dir],
                         args.gt_csv.expanduser().resolve(), args.reference, args.all_windows,
                         [tuple(root.split('=', 1)) for root in args.video_root])
    out = args.out or result_dir / 'eval_compare.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    total = report['summary']['total']
    line = f"{out} · pairs {total['pairs']} (judged {total['judged']}) · impossible {total['impossible']}"
    if 'with_choice' in total:
        line += f" · agrees with the generation's choice {total['agrees']}/{total['with_choice']}"
    print(line)


if __name__ == '__main__':
    main()
