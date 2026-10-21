"""Run the demo's pipeline on every IntPhys2 pair and store the trials; `--result-dir` opens in the demo.

Resumable; a CUDA or vLLM engine failure exits with 3. `--threads` pairs walk at once and share one vLLM engine.
E.g. `CUDA_VISIBLE_DEVICES=0 python score/run_test.py --intphys2-dir <IntPhys2 root> --limit 8`.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import signal
import sys
import threading
import time
import traceback
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]   # videotrace_v3/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Imported after the repo root is on sys.path, so the script runs from anywhere.
import torch  # noqa: E402
from vllm.v1.engine.exceptions import EngineDeadError  # noqa: E402

from demo.app import (  # noqa: E402
    ALL_FRAMES, DEFAULT_INTERNVIDEO2_CLIP_FRAMES, DEFAULT_MLLM_FRAMES, DEFAULT_PE_AV_CLIP_FRAMES)
from demo.core.gt import GT_CSV  # noqa: E402
from demo.core.pairs import SIDES, load_pairs, main_videos_dir  # noqa: E402
from demo.core.result import for_page  # noqa: E402
from demo.core.storage import save_result, save_trial, summarize  # noqa: E402
from PhysTrace.video_cache import default_cache_dir  # noqa: E402
from PhysTrace.video_models.clip import CLIP_MODELS, DEFAULT_CLIP_MODEL, ClipScorer  # noqa: E402
from PhysTrace.video_models.mllm import (  # noqa: E402
    BACKENDS,
    DEFAULT_BACKEND,
    DEFAULT_SEED,
    DEFAULT_THINKING_BUDGET,
    VideoPredictor,
)
from PhysTrace.video_models.mllm_prompts import ATTRIBUTES, PROMPT_SET_DIR, PromptConfig, load_prompt_set  # noqa: E402
from PhysTrace.video_pipeline import analyze_video  # noqa: E402
from score.batching import BatchLLM, Locked, Progress  # noqa: E402


# The demo's own defaults (demo/app.py), which it keeps as literals.
DEFAULT_WINDOW_SECONDS = 2.0
DEFAULT_HOP_SECONDS = 1.0
DEFAULT_MAX_NEW_TOKENS = 8000
DEFAULT_THREADS = 8
MAX_ATTEMPTS = 2                          # a pair that failed this often is left for --retry-failed
TRIAL_FILES = ['/results/results.csv']    # what the demo offers as a trial's download, as demo.server stores it
FAILED_LOG = 'failed.jsonl'

EXIT_FINISHED = 0     # nothing left for this shard
EXIT_FATAL = 3        # the GPU context or the engine is gone; the remaining pairs wait for a manual rerun
EXIT_TERMINATED = 143

# Once one of these is raised the process cannot do anything useful on the GPU again.
FATAL_ERRORS = (torch.AcceleratorError, torch.OutOfMemoryError, EngineDeadError)
_FATAL_MESSAGES = ('CUDA error', 'CUDA out of memory', 'NCCL error', 'EngineCore')



def parse_category_filter(text: str) -> dict[str, set[str]]:
    """One `--categories` value, `attribute[,attribute...]=category[,category...]`, as {attribute: categories}."""
    attributes, separator, categories = text.partition('=')
    names = [name for name in attributes.split(',') if name]
    chosen = {category for category in categories.split(',') if category}
    if not separator or not names or not chosen:
        raise argparse.ArgumentTypeError(f'{text!r}: expected attribute[,attribute...]=category[,category...]')
    if unknown := set(names) - set(ATTRIBUTES):
        raise argparse.ArgumentTypeError(
            f'{text!r}: unknown attributes {sorted(unknown)}; choose from {list(ATTRIBUTES)}')
    return {name: chosen for name in names}


def merge_category_filters(filters: list[dict[str, set[str]]]) -> dict[str, set[str]]:
    """The `--categories` values as one {attribute: categories} mapping."""
    merged: dict[str, set[str]] = {}
    for one in filters:
        for attribute, categories in one.items():
            merged.setdefault(attribute, set()).update(categories)
    return merged


def check_categories(filters: dict[str, set[str]], pairs: list[dict]) -> None:
    """Make sure every named category has pairs under its condition, before any model loads."""
    available: dict[str, set[str]] = {}
    for pair in pairs:
        available.setdefault(pair['attribute'], set()).add(pair['category'])
    for attribute, categories in filters.items():
        if missing := categories - available.get(attribute, set()):
            raise ValueError(f'--categories: no {attribute} pairs of {sorted(missing)}; '
                             f'{attribute} has {sorted(available.get(attribute, set()))}')


def _add_batch_arguments(parser: argparse.ArgumentParser) -> None:
    """Which pairs this process takes and where they are stored."""
    group = parser.add_argument_group('batch')
    group.add_argument('--shard', type=int, default=0)
    group.add_argument('--nshards', type=int, default=1)
    group.add_argument('--limit', type=int, help="only this shard's first N pending pairs (smoke runs)")
    group.add_argument('--force', action='store_true', help='re-analyze pairs already stored')
    group.add_argument('--retry-failed', action='store_true',
                       help=f'also take pairs that already failed {MAX_ATTEMPTS} times')
    group.add_argument('--status', action='store_true', help='print progress for this run and exit')
    group.add_argument('--threads', type=int, default=DEFAULT_THREADS,
                       help='pairs walked at once, i.e. the most requests in the vLLM batch (default: %(default)s)')
    group.add_argument('--log-windows', action='store_true',
                       help='one log line per finished window (always on with --threads 1)')
    group.add_argument('--status-interval', type=float, default=30,
                       help='seconds between the progress summary lines; 0 = off (default: %(default)s)')
    group.add_argument('--intphys2-dir', type=Path, required=True,
                       help='IntPhys2 root holding Main/ (Main/Videos and Main/metadata.csv give the pair list)')
    group.add_argument('--gt-csv', type=Path, default=GT_CSV,
                       help='the pairs this run covers are the ones this GT CSV lists (default: %(default)s)')
    group.add_argument('--attributes', nargs='+', choices=ATTRIBUTES, default=list(ATTRIBUTES),
                       help='only pairs of these conditions (default: all)')
    group.add_argument('--categories', nargs='+', type=parse_category_filter, metavar='ATTRIBUTES=CATEGORIES',
                       help='only these categories, per condition, e.g. continuity,immutability=Box,RotatingCup '
                            'solidity=BoxSoldity; a condition that is not named is left out (default: every category)')
    group.add_argument('--result-dir', type=Path, default=Path('output'),
                       help='results.csv / trials.csv in the demo schema')


def _add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    """The window geometry and both models, with the demo's defaults."""
    walk = parser.add_argument_group('windows')
    walk.add_argument('--window-seconds', type=float, default=DEFAULT_WINDOW_SECONDS)
    walk.add_argument('--hop-seconds', type=float, default=DEFAULT_HOP_SECONDS, help='0 < hop <= window')

    mllm = parser.add_argument_group('MLLM (vLLM; pick its GPU with CUDA_VISIBLE_DEVICES)')
    mllm.add_argument('--mllm-backend', choices=sorted(BACKENDS), default=DEFAULT_BACKEND)
    mllm.add_argument('--mllm-model-dir', type=Path, help="model directory, if it is not the backend's own")
    mllm.add_argument('--mllm-frames', type=int, default=DEFAULT_MLLM_FRAMES,
                      help='even frame count for every window, whatever its length; 0 = 4 per second of window')
    mllm.add_argument('--max-new-tokens', type=int, default=DEFAULT_MAX_NEW_TOKENS,
                      help='generation limit per call, reasoning included')
    mllm.add_argument('--max-model-len', type=int, help="engine context (default: the backend's own)")
    mllm.add_argument('--thinking-budget', type=int, default=DEFAULT_THINKING_BUDGET,
                      help='reasoning tokens per window before vLLM closes </think>; 0 = no cap')
    mllm.add_argument('--seed', type=int, default=DEFAULT_SEED)
    mllm.add_argument('--temperature', type=float, help="override the checkpoint's sampling temperature")
    mllm.add_argument('--gpu-fraction', type=float, help='GPU share vLLM reserves; the default is its own budget')

    clip = parser.add_argument_group('CLIP')
    clip.add_argument('--clip-model', choices=sorted(CLIP_MODELS), default=DEFAULT_CLIP_MODEL)
    clip.add_argument('--clip-device', default='cuda:0')
    clip.add_argument('--clip-frames', type=int,
                      help=f'frames encoded per window (default: {DEFAULT_PE_AV_CLIP_FRAMES} for pe-av-large, '
                           f'{DEFAULT_INTERNVIDEO2_CLIP_FRAMES} for InternVideo2); pe-av-large samples N > 0 evenly, '
                           f'{ALL_FRAMES} reads every frame')

    prompts = parser.add_argument_group('prompts')
    prompts.add_argument('--prompt-set', type=Path, default=PROMPT_SET_DIR,
                         help='a folder holding <condition>.system.txt and <condition>.segment.txt '
                              f'(default: {PROMPT_SET_DIR.relative_to(REPO_ROOT)})')
    prompts.add_argument('--num-prompts', type=int, default=PromptConfig.num_prompts,
                         help='captions in each of "state" and "prediction" (default: %(default)s)')
    prompts.add_argument('--history-windows', type=int, default=PromptConfig.history_windows,
                         help='latest trusted windows the model is shown; 0 shows all, -1 none (default: %(default)s)')
    prompts.add_argument('--lowest-windows', type=int, default=PromptConfig.lowest_windows,
                         help='windows with the lowest CLIP scores that become the error candidates; 0 flags none '
                              '(default: %(default)s)')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line and resolve the backend's checkpoint, as the demo does."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_batch_arguments(parser)
    _add_analysis_arguments(parser)
    args = parser.parse_args(argv)
    if not 0 <= args.shard < args.nshards:
        parser.error('--shard must be in [0, --nshards)')
    if args.threads < 1:
        parser.error('--threads must be >= 1')
    args.backend = BACKENDS[args.mllm_backend]
    args.mllm_model_dir = (args.mllm_model_dir.expanduser().resolve() if args.mllm_model_dir
                           else Path(args.backend.path(Path('AlignmentModels/weights'))))
    args.video_dir = main_videos_dir(args.intphys2_dir.expanduser().resolve())
    args.gt_csv = args.gt_csv.expanduser().resolve()
    args.npz_dir = default_cache_dir(args.video_dir)
    args.result_dir = args.result_dir.expanduser().resolve()
    args.categories = merge_category_filters(args.categories) if args.categories else None
    if args.categories:
        try:
            check_categories(args.categories, load_pairs(args.video_dir, gt_csv=args.gt_csv))
        except ValueError as error:
            parser.error(str(error))
    if args.clip_frames is None:
        args.clip_frames = (DEFAULT_PE_AV_CLIP_FRAMES if CLIP_MODELS[args.clip_model].reads_all_frames
                            else DEFAULT_INTERNVIDEO2_CLIP_FRAMES)
    args.prompt_set = args.prompt_set.expanduser().resolve()
    args.prompt_configs = load_prompt_set(args.prompt_set, num_prompts=args.num_prompts,
                                          history_windows=args.history_windows, lowest_windows=args.lowest_windows)
    return args


def run_key(args: argparse.Namespace) -> dict:
    """What makes two stored trials the same run, for resuming."""
    return dict(window_seconds=float(args.window_seconds), hop_seconds=float(args.hop_seconds),
                clip_model=args.clip_model, clip_frames=args.clip_frames, mllm_backend=args.mllm_backend,
                prompt_set=args.prompt_set.name, num_prompts=args.num_prompts, history_windows=args.history_windows,
                lowest_windows=args.lowest_windows)


def _trial_key(params: dict) -> dict:
    """The run key of one stored trial's params; keys older trials lack take the values those runs used."""
    return dict(window_seconds=float(params['window_seconds']), hop_seconds=float(params['hop_seconds']),
                clip_model=params.get('clip_model'), clip_frames=params.get('clip_frames'),
                mllm_backend=params.get('mllm_backend'), prompt_set=params.get('prompt_set'),
                num_prompts=params['prompts']['num_prompts'], history_windows=params['prompts']['history_windows'],
                lowest_windows=params['prompts']['lowest_windows'])


def stored_groups(result_dir: Path, key: dict) -> set[str]:
    """Pairs trials.csv already holds, both sides, for exactly this run."""
    path = result_dir / 'trials.csv'
    if not path.is_file():
        return set()
    done = set()
    with path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            params = json.loads(row['trial_json'])['params']
            if _trial_key(params) == key and set(params.get('sides', [])) == set(SIDES):
                done.add(row['group'])
    return done


def failure_counts(result_dir: Path, key: dict) -> dict[str, int]:
    """How often each pair has failed in this run."""
    path = result_dir / FAILED_LOG
    counts: dict[str, int] = {}
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            record = json.loads(line)
            if record['run'] == key:
                counts[record['group']] = counts.get(record['group'], 0) + 1
    return counts


def record_failure(result_dir: Path, key: dict, pair: dict, error: BaseException) -> None:
    """Append one failed pair, with its traceback, so the next run can decide to skip it."""
    record = dict(run=key, group=pair['group'], at=datetime.now(timezone.utc).isoformat(),
                  error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    with (result_dir / FAILED_LOG).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')


def run_pairs(args: argparse.Namespace) -> list[dict]:
    """The sorted pairs this run covers: the chosen conditions and categories."""
    pairs = [pair for pair in load_pairs(args.video_dir, gt_csv=args.gt_csv)
             if pair['attribute'] in args.attributes]
    if args.categories:
        pairs = [pair for pair in pairs if pair['category'] in args.categories.get(pair['attribute'], ())]
    return pairs


def pending_pairs(args: argparse.Namespace) -> list[dict]:
    """This shard's share of the run's pairs, minus the stored ones and those that keep failing."""
    pairs = run_pairs(args)[args.shard::args.nshards]
    if not args.force:
        key = run_key(args)
        done = stored_groups(args.result_dir, key)
        failures = {} if args.retry_failed else failure_counts(args.result_dir, key)
        pairs = [pair for pair in pairs
                 if pair['group'] not in done and failures.get(pair['group'], 0) < MAX_ATTEMPTS]
    return pairs[:args.limit] if args.limit else pairs


def print_status(args: argparse.Namespace) -> None:
    """Stored, failed and pending pairs of this run, over every shard."""
    key = run_key(args)
    groups = {pair['group'] for pair in run_pairs(args)}
    done = stored_groups(args.result_dir, key) & groups
    failures = failure_counts(args.result_dir, key)
    given_up = {group for group in groups - done if failures.get(group, 0) >= MAX_ATTEMPTS}
    print(f'run {key} · {args.result_dir}')
    print(f'  pairs {len(groups)} · stored {len(done)} · failed {MAX_ATTEMPTS}x (skipped) {len(given_up)} · '
          f'pending {len(groups) - len(done) - len(given_up)}')


def load_models(args: argparse.Namespace) -> tuple[VideoPredictor, ClipScorer]:
    """The models, in the order and with the settings the demo loads them."""
    started = time.perf_counter()
    scorer = ClipScorer(args.clip_device, args.clip_frames, args.clip_model,
                        sample_frames=args.clip_frames != ALL_FRAMES)
    print(f'[load] {args.clip_model}: {time.perf_counter() - started:.2f}s', flush=True)
    started = time.perf_counter()
    predictor = VideoPredictor(
        args.backend, str(args.mllm_model_dir), window_seconds=args.window_seconds, num_frames=args.mllm_frames,
        max_new_tokens=args.max_new_tokens, seed=args.seed, temperature=args.temperature,
        gpu_fraction=args.gpu_fraction, thinking_budget=args.thinking_budget, max_model_len=args.max_model_len)
    print(f'[load] mllm (vllm): {time.perf_counter() - started:.2f}s', flush=True)
    return predictor, scorer


def default_prompts(attribute: str, args: argparse.Namespace):
    """The condition's prompts with the limits the UI sends by default (its form shows the server's)."""
    return replace(args.prompt_configs[attribute], max_new_tokens=args.max_new_tokens,
                   thinking_budget=args.thinking_budget)


def describe_models(models: dict, args: argparse.Namespace) -> dict:
    """Provenance stored with every side, in the flat shape the demo's table reads."""
    return dict(models, clip=CLIP_MODELS[args.clip_model].name, clip_identity=models['clip'],
                clip_frames=args.clip_frames, clip_device=args.clip_device)


def analyze_side(video: Path, prompts, models: tuple, args: argparse.Namespace, report) -> tuple[dict, str]:
    """One video through the walk, appended to results.csv as the demo stores it."""
    predictor, scorer = models
    result = analyze_video(video, predictor, scorer, window_seconds=args.window_seconds,
                           hop_seconds=args.hop_seconds, prompts=prompts, cache_root=args.npz_dir,
                           progress=report)
    page = for_page(result, predictor, scorer)
    page['models'] = describe_models(result['models'], args)
    result_id, _ = save_result(page, args.result_dir)
    return page, result_id


def analyze_pair(pair: dict, models: tuple, args: argparse.Namespace, progress: Progress, position: int,
                 total: int) -> dict:
    """Both sides, then the pair as the trial demo.server stores for a two-sided run.

    A side that raises leaves the pair unstored, so it is analyzed again on a later run.
    """
    prompts = default_prompts(pair['attribute'], args)
    results, ids = {}, {}
    for side, label in SIDES.items():
        video = args.video_dir / f"{pair[f'{side}_name']}.mp4"
        report = progress.start(pair['group'], label[:4], position, total)
        results[side], ids[side] = analyze_side(video, prompts, models, args, report)
    trial = dict(id=uuid.uuid4().hex, group=pair['group'], pair=pair,
                 created_at=datetime.now(timezone.utc).isoformat(),
                 params=dict(window_seconds=args.window_seconds, hop_seconds=args.hop_seconds,
                             sides=list(SIDES), prompts=asdict(prompts),
                             clip_model=args.clip_model, clip_frames=args.clip_frames,
                             mllm_backend=args.mllm_backend, prompt_set=run_key(args)['prompt_set']),
                 results=results, files=dict.fromkeys(results, TRIAL_FILES),
                 summary={side: summarize(result) for side, result in results.items()})
    save_trial(trial, ids, args.result_dir)
    return trial


def is_fatal(error: BaseException) -> bool:
    """True when the process's GPU context or engine is gone, whatever wrapped the error."""
    seen = error
    while seen is not None:
        if isinstance(seen, FATAL_ERRORS) or any(text in str(seen) for text in _FATAL_MESSAGES):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def _rss_gib() -> float:
    """This process's resident memory (vLLM's engine is a separate process)."""
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) / 2**20
    return float('nan')


def _memory_line(device: str) -> str:
    """Host RSS and the GPU's free memory, logged after every pair to catch a slow leak early."""
    free, total = torch.cuda.mem_get_info(torch.device(device))
    return f'rss {_rss_gib():.1f}GiB · gpu free {free / 2**30:.1f}/{total / 2**30:.0f}GiB'


class _ShardRun:
    """What the worker threads of one shard share: the pairs still to claim and how many have finished."""

    def __init__(self, pending: list[dict]):
        self.pending, self.lock, self.fatal = pending, threading.Lock(), threading.Event()
        self.todo = iter(enumerate(pending, 1))
        self.stored = self.failed = 0
        self.started = time.monotonic()

    def claim(self) -> tuple[int, dict] | None:
        """The next (position, pair), or None when none is left or the shard has hit a fatal error."""
        with self.lock:
            return None if self.fatal.is_set() else next(self.todo, None)


def _worker(run: _ShardRun, models: tuple, args: argparse.Namespace, progress: Progress, in_flight) -> None:
    """Analyze claimed pairs until none is left or a fatal error stops the shard."""
    total = len(run.pending)
    while (item := run.claim()) is not None:
        position, pair = item
        one = time.monotonic()
        try:
            analyze_pair(pair, models, args, progress, position, total)
        except Exception as error:   # sorted below: fatal ends the shard, anything else skips the pair
            progress.finish(pair['group'])
            with run.lock:
                if is_fatal(error):
                    print(f"[fatal] {pair['group']}: {type(error).__name__}: {error}\n{traceback.format_exc()}",
                          flush=True)
                    run.fatal.set()
                    return
                run.failed += 1
                record_failure(args.result_dir, run_key(args), pair, error)
                print(f"[{position}/{total}] {pair['group']} failed: {type(error).__name__}: {error}", flush=True)
            continue
        progress.finish(pair['group'])
        with run.lock:
            run.stored += 1
            hours = (time.monotonic() - run.started) / 3600
            print(f"[{position}/{total}] {pair['group']} stored · {time.monotonic() - one:.0f}s · "
                  f"done {run.stored + run.failed} (~{(run.stored + run.failed) / max(1e-9, hours):.0f} pairs/h) · "
                  f"in flight {in_flight()} · {_memory_line(args.clip_device)}", flush=True)
        gc.collect()


def _print_summaries(run: _ShardRun, progress: Progress, interval: float, in_flight) -> None:
    """Every `interval` seconds, one line with the finished count and what each thread is walking."""
    while interval > 0:
        time.sleep(interval)
        with run.lock:
            done = run.stored + run.failed
            print(f'[progress] done {done}/{len(run.pending)} (failed {run.failed}) · {time.strftime("%H:%M:%S")} · '
                  f'in flight {in_flight()} · {progress.summary()}', flush=True)


def run_shard(args: argparse.Namespace) -> int:
    """Walk every pending pair of this shard, `--threads` at a time; returns the exit code."""
    pending = pending_pairs(args)
    print(f'[shard {args.shard}/{args.nshards}] pairs left {len(pending)} · threads {args.threads} · '
          f'{run_key(args)} · {args.result_dir}', flush=True)
    if not pending:
        return EXIT_FINISHED

    args.result_dir.mkdir(parents=True, exist_ok=True)
    predictor, scorer = load_models(args)
    batcher = None
    if args.threads > 1:
        # One thread keeps the engine and the scorer to itself; several share them through these wrappers.
        batcher = BatchLLM(predictor.llm)
        predictor.llm, scorer = batcher, Locked(scorer)

    def in_flight() -> int:
        return len(batcher.pending) if batcher else 0

    run, progress = _ShardRun(pending), Progress(args.log_windows or args.threads == 1)
    threading.Thread(target=_print_summaries, args=(run, progress, args.status_interval, in_flight),
                     daemon=True).start()
    # Daemon threads, so a SIGTERM (SystemExit in the main thread) ends the process without waiting for the walks.
    workers = [threading.Thread(target=_worker, args=(run, (predictor, scorer), args, progress, in_flight),
                                daemon=True) for _ in range(args.threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    print(f'[shard {args.shard}] stored {run.stored} pairs · failed {run.failed} pairs · '
          f'{(time.monotonic() - run.started) / 3600:.2f}h', flush=True)
    return EXIT_FATAL if run.fatal.is_set() else EXIT_FINISHED


def main(argv: list[str] | None = None) -> None:
    """Resume this shard's run, or report progress with --status."""
    # A plain `kill` would skip atexit and orphan vLLM's engine process on the GPU.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(EXIT_TERMINATED))
    args = parse_args(argv)
    if args.status:
        print_status(args)
        return
    sys.exit(run_shard(args))


if __name__ == '__main__':
    main()
