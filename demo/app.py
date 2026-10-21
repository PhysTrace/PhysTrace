"""Entry point: load the two models once, then analyze videos from the CLI (`python -m demo.app --video clip.mp4`)
or serve the browser UI.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import signal
import sys
import threading
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Imported after the repo root is on sys.path, so `python demo/app.py` also works.
from demo.core.gt import GT_CSV  # noqa: E402
from demo.core.pairs import main_videos_dir  # noqa: E402
from demo.core.result import for_page  # noqa: E402
from demo.core.storage import save_result  # noqa: E402
from demo.server import serve  # noqa: E402
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

# InternVideo2: a fixed frame count per window.
DEFAULT_INTERNVIDEO2_CLIP_FRAMES = 8
MIN_INTERNVIDEO2_CLIP_FRAMES = 2
MAX_INTERNVIDEO2_CLIP_FRAMES = 32

# PE-AV: every frame of a window (ALL_FRAMES) or N sampled evenly.
ALL_FRAMES = 0
DEFAULT_PE_AV_CLIP_FRAMES = 60
MAX_PE_AV_CLIP_FRAMES = 1024

# MLLM: a fixed frame count per window.
DEFAULT_MLLM_FRAMES = 60


class Demo:
    """The loaded models, driven one video at a time and stored as each one finishes."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.Lock()
        self.predictor = self.scorer = None
        self.load_seconds: dict[str, float] = {}
        # What the loaded (or next loaded) scorer reads per window; a run may ask for another.
        self.clip_frames = args.clip_frames

    def load_models(self) -> None:
        """Load both models once; every later run reuses them."""
        with self.lock:
            self._load()

    def _load(self) -> None:
        if self.scorer is None:
            self.scorer = self._timed(self.args.clip_model, ClipScorer,
                                      self.args.clip_device, self.clip_frames, self.args.clip_model,
                                      sample_frames=self.clip_frames != ALL_FRAMES)
        if self.predictor is None:
            self.predictor = self._timed(
                'mllm (vllm)', VideoPredictor, self.args.backend, str(self.args.mllm_model_dir),
                window_seconds=self.args.window_seconds, num_frames=self.args.mllm_frames,
                max_new_tokens=self.args.max_new_tokens, seed=self.args.seed,
                temperature=self.args.temperature, gpu_fraction=self.args.gpu_fraction,
                thinking_budget=self.args.thinking_budget, max_model_len=self.args.max_model_len)

    def _timed(self, name: str, factory, *arguments, **options):
        """Build one model and report how long it took."""
        started = time.perf_counter()
        model = factory(*arguments, **options)
        self.load_seconds[name] = time.perf_counter() - started
        print(f'[load] {name}: {self.load_seconds[name]:.2f}s', flush=True)
        return model

    def clip_frames_range(self) -> tuple[int, int]:
        """The "CLIP input frames" values the selected scorer takes, both ends included."""
        if CLIP_MODELS[self.args.clip_model].reads_all_frames:
            return ALL_FRAMES, MAX_PE_AV_CLIP_FRAMES
        return MIN_INTERNVIDEO2_CLIP_FRAMES, MAX_INTERNVIDEO2_CLIP_FRAMES

    def check_clip_frames(self, frames) -> int:
        """A "CLIP input frames" value the selected scorer can take."""
        lowest, highest = self.clip_frames_range()
        if isinstance(frames, bool) or not isinstance(frames, int) or not lowest <= frames <= highest:
            every = f' ({ALL_FRAMES} = every frame)' if lowest == ALL_FRAMES else ''
            raise ValueError(f'CLIP frames for {CLIP_MODELS[self.args.clip_model].name} must be an integer in '
                             f'{lowest}..{highest}{every}; got {frames!r}')
        return frames

    def _use_clip_frames(self, frames: int) -> None:
        """Drop the scorer when a run asks for another frame count; `_load` builds the new one."""
        if frames == self.clip_frames:
            return
        print(f'[load] CLIP frames {self.clip_frames} -> {frames}: reloading {self.args.clip_model}', flush=True)
        self.clip_frames = frames
        if self.scorer is not None:
            self.scorer = None
            # Return the old weights to the card before the new copy is allocated next to vLLM.
            gc.collect()
            torch.cuda.empty_cache()

    def run(self, video, window_seconds: float, hop_seconds: float, progress=None, *,
            prompts: PromptConfig | dict, clip_frames: int | None = None) -> tuple[dict, str, Path]:
        """Analyze one video and append it to the result store; `prompts` may be the plain dict the browser sends."""
        prompts = PromptConfig(**prompts) if isinstance(prompts, dict) else prompts
        with self.lock:
            if clip_frames is not None:
                self._use_clip_frames(self.check_clip_frames(clip_frames))
            self._load()
            result = analyze_video(video, self.predictor, self.scorer,
                                   window_seconds=window_seconds, hop_seconds=hop_seconds,
                                   prompts=prompts, cache_root=self.args.npz_dir, progress=progress)
            page = for_page(result, self.predictor, self.scorer)
            page['models'] = self.describe_models(result['models'])
            result_id, path = save_result(page, self.args.output)
        return page, result_id, path

    def describe_models(self, models: dict) -> dict:
        """Provenance stored with every run, in the flat shape the page's table reads."""
        return dict(models, clip=CLIP_MODELS[self.args.clip_model].name, clip_identity=models['clip'],
                    clip_frames=self.clip_frames, clip_device=self.args.clip_device)


def _add_demo_arguments(parser: argparse.ArgumentParser) -> None:
    """The demo's own inputs: what to analyze, how to cut it, and where results go."""
    group = parser.add_argument_group('demo')
    group.add_argument('--video', nargs='+', help='CLI mode: analyze these videos and exit')
    group.add_argument('--attribute', choices=ATTRIBUTES, help='CLI mode: the IntPhys2 condition whose prompts '
                       '--video asks with')
    group.add_argument('--prompt-set', type=Path, default=PROMPT_SET_DIR,
                       help='a folder holding <condition>.system.txt and <condition>.segment.txt, the '
                            'prompts the page starts from (default: IntPhys2Timing/prompts)')
    group.add_argument('--window-seconds', type=float, default=2.0)
    group.add_argument('--hop-seconds', type=float, default=1.0, help='0 < hop <= window')
    group.add_argument('--intphys2-dir', type=Path, required=True,
                       help='IntPhys2 root holding Main/ (Main/Videos and Main/metadata.csv give the pair list)')
    group.add_argument('--result-dir', dest='output', type=Path, default=ROOT / 'demo' / 'results',
                       help='where results.csv and trials.csv live')
    group.add_argument('--gt-csv', type=Path, default=GT_CSV,
                       help='violation GT drawn on the Impossible chart (demo/gt/build_gt_csv.py writes it)')
    group.add_argument('--host', default='0.0.0.0')
    group.add_argument('--port', type=int, default=7874)


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Which models to load and how much they may generate."""
    group = parser.add_argument_group('MLLM (vLLM; pick its GPU with CUDA_VISIBLE_DEVICES)')
    group.add_argument('--mllm-backend', choices=sorted(BACKENDS), default=DEFAULT_BACKEND)
    group.add_argument('--mllm-model-dir', type=Path,
                       help="model directory, if it is not the backend's own under AlignmentModels/weights/")
    group.add_argument('--mllm-frames', type=int, default=DEFAULT_MLLM_FRAMES,
                       help='even frame count for every window, whatever its length; 0 = 4 per second of window. '
                            'The page\'s "MLLM input frames" overrides it per run')
    # Thinking backends reason before answering; 8000 leaves the answer room past the thinking budget.
    group.add_argument('--max-new-tokens', type=int, default=8000, help='generation limit per call, reasoning included')
    group.add_argument('--max-model-len', type=int,
                       help="engine context, prompt and output together (default: the backend's own, "
                            '8B 49152 = the official 40960-token output + 8192 for the prompt, 32B 16384)')
    group.add_argument('--thinking-budget', type=int, default=DEFAULT_THINKING_BUDGET,
                       help='reasoning tokens per window before vLLM closes </think> (reasoning backends); 0 = no cap')
    group.add_argument('--seed', type=int, default=DEFAULT_SEED, help='sampling seed')
    group.add_argument('--temperature', type=float,
                       help="override the checkpoint's sampling temperature; 0 decodes greedily")
    group.add_argument('--gpu-fraction', type=float,
                       help='GPU share vLLM reserves; the default is its own budget')

    clip = parser.add_argument_group('CLIP (video-text scorer)')
    clip.add_argument('--clip-model', choices=sorted(CLIP_MODELS), default=DEFAULT_CLIP_MODEL,
                      help='pe-av-large: every frame of a window, ModernBERT text tower, captions up to 512 tokens; '
                           'internvideo2-stage2-6b: BERT-large text tower, up to 40 tokens; '
                           'internvideo2-clip-6b: InternVL-C/LLaMA text tower, up to 80 tokens')
    # The 32B backends' 62GiB of weights leave no room for CLIP's 25GiB: give them two devices and cuda:1.
    clip.add_argument('--clip-device', default='cuda:0',
                      help='cuda:1 puts CLIP on the second visible device (needed for the 32B backends)')
    clip.add_argument('--clip-frames', type=int,
                      help=f'frames encoded per window, the page\'s starting "CLIP input frames" (default: '
                           f'{DEFAULT_INTERNVIDEO2_CLIP_FRAMES} for InternVideo2, '
                           f'which takes {MIN_INTERNVIDEO2_CLIP_FRAMES}..{MAX_INTERNVIDEO2_CLIP_FRAMES}; '
                           f'{DEFAULT_PE_AV_CLIP_FRAMES} for pe-av-large, which samples 1..{MAX_PE_AV_CLIP_FRAMES} '
                           f'evenly or reads every frame at {ALL_FRAMES}); part of what a cached feature means')



def parse_args(argv=None) -> argparse.Namespace:
    """Read the command line and resolve the selected backend's defaults."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_demo_arguments(parser)
    _add_model_arguments(parser)

    args = parser.parse_args(argv)
    args.backend = BACKENDS[args.mllm_backend]
    args.mllm_model_dir = (args.mllm_model_dir.expanduser().resolve() if args.mllm_model_dir
                           else Path(args.backend.path(Path('AlignmentModels/weights'))))
    args.output = args.output.expanduser().resolve()
    args.video_dir = main_videos_dir(args.intphys2_dir.expanduser().resolve())
    args.npz_dir = default_cache_dir(args.video_dir)
    if args.video and not args.attribute:
        parser.error('--video needs --attribute, the condition whose prompts it asks with')
    args.prompt_set = args.prompt_set.expanduser().resolve()
    try:
        args.prompt_configs = load_prompt_set(args.prompt_set)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.clip_frames is None:
        args.clip_frames = (DEFAULT_PE_AV_CLIP_FRAMES if CLIP_MODELS[args.clip_model].reads_all_frames
                            else DEFAULT_INTERNVIDEO2_CLIP_FRAMES)
    return args


def _print_window(row: dict) -> None:
    """One progress line per finished window, for CLI mode."""
    # The first window has no earlier forecast to score.
    score = 'none' if row['score'] is None else f"{row['score']:.4f}"
    print(f"window {row['index'] + 1}/{row['total']}: score={score} trusted={row['trusted']}", flush=True)


def _analyze_from_cli(demo: Demo, args: argparse.Namespace) -> None:
    """Analyze each --video and print the result as JSON."""
    for video in args.video:
        result, _, path = demo.run(video, args.window_seconds, args.hop_seconds, progress=_print_window,
                                   prompts=args.prompt_configs[args.attribute])
        print(json.dumps({'video': result['video'], 'has_anomaly': result['has_anomaly'],
                          'anomaly_intervals': result['anomaly_intervals'],
                          'saved_to': str(path)}, ensure_ascii=False, indent=2))


def main() -> None:
    """Load both models, then either analyze the --video list or serve the browser UI."""
    # A plain `kill` would skip atexit and orphan vLLM's engine process on the GPU.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    args = parse_args()
    print(f'[MLLM] loading {args.mllm_model_dir} (vllm) and {CLIP_MODELS[args.clip_model].name}…', flush=True)
    demo = Demo(args)
    demo.load_models()
    print('[MLLM] models ready', flush=True)

    if args.video:
        _analyze_from_cli(demo, args)
        return

    serve(demo, args)


if __name__ == '__main__':
    main()
