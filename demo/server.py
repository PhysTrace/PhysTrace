"""The browser side: pair list, one analysis job at a time, and the stored trials."""
from __future__ import annotations

import contextlib
from dataclasses import asdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, unquote, urlparse
import uuid

import cv2

from demo.core.gt import load_gt
from demo.core.pairs import FILTER_COLUMNS, META_COLUMNS, SIDES, load_pairs
from demo.core.result import SCHEMA_VERSION
from demo.core.storage import delete_trials, load_trials, save_trial, summarize
from PhysTrace.video_models.clip import CLIP_MODELS
from PhysTrace.video_models.mllm import FRAMES_PER_SECOND
from PhysTrace.video_models.mllm_prompts import PromptConfig
from PhysTrace.video_windows import split_windows

STATIC = Path(__file__).resolve().parent / 'static'
STATIC_ROUTES = {'/': 'index.html', '/index.html': 'index.html', '/app.js': 'app.js',
                 '/style.css': 'style.css', '/mllm.css': 'mllm.css'}
RESULTS_PREFIX = '/results/'
# A score request carries three edited prompts, so it is larger than a typical POST.
MAX_REQUEST_BYTES = 262144
FRAME_MAX_EDGE = 480
KEPT_JOBS = 40
PROGRESS_BAR_WIDTH = 24


def progress_bar(label: str, done: int, total: int, elapsed: float) -> None:
    """One line per finished window, rewritten in place, so the terminal tracks the GPU."""
    ratio = done / total if total else 0
    per_window = elapsed / done if done else 0
    filled = round(PROGRESS_BAR_WIDTH * ratio)
    left = f' · remaining {per_window * (total - done):.0f}s' if total else ''
    print(f'\r  {label} {done:>3}/{total or "?"} [{"#" * filled}'
          f'{"." * (PROGRESS_BAR_WIDTH - filled)}] {ratio * 100:5.1f}% · {per_window:.1f}s/window{left}   ',
          end='\n' if total and done >= total else '', flush=True)


class Browser:
    """Everything the HTTP layer reads: the pairs, the stored trials and the running job."""

    def __init__(self, demo, args):
        self.demo, self.args = demo, args
        self.output = args.output
        self.video_dir = args.video_dir.expanduser().resolve()
        self.pairs = load_pairs(self.video_dir, self.video_dir.parent / 'metadata.csv', args.gt_csv)
        # Drawn on the Impossible chart to judge a run; the analysis never sees it.
        gt = load_gt(args.gt_csv)
        for pair in self.pairs:
            pair['gt'] = gt.get((pair['attribute'], pair['group']))
        self.by_group = {pair['group']: pair for pair in self.pairs}
        self.names = {pair[f'{side}_name'] for pair in self.pairs for side in SIDES}
        # Each IntPhys2 condition has its own wording; the pair's condition picks the defaults.
        self.prompts = {attribute: asdict(args.prompt_configs[attribute])
                        for attribute in sorted({pair['attribute'] for pair in self.pairs})}
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        self.active: str | None = None
        self.geometries: dict[str, dict] = {}
        # A stored trial is kept only while its pair still names the same two videos. Older schemas
        # arrive already converted by load_trials, so the schema check only drops what it could not convert.
        self.trials = {trial['id']: trial for trial in load_trials(self.output)
                       if (pair := self.by_group.get(trial['group']))
                       and all(trial['pair'][key] == pair[key] for key in ('good_name', 'bad_name'))
                       and all(result.get('schema_version') == SCHEMA_VERSION
                               for result in trial['results'].values())}

    def video_path(self, name: str) -> Path:
        """The file behind a listed video name."""
        if name not in self.names:
            raise KeyError('Unknown video')
        return self.video_dir / f'{name}.mp4'

    def geometry(self, name: str) -> dict:
        """FPS and frame count, read from the container once per video."""
        if name not in self.geometries:
            capture = cv2.VideoCapture(str(self.video_path(name)))
            try:
                if not capture.isOpened():
                    raise ValueError('Cannot decode video')
                fps, count = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                capture.release()
            if not math.isfinite(fps) or fps <= 0 or count <= 0:
                raise ValueError('Invalid video FPS or frame count')
            self.geometries[name] = {'fps': fps, 'n_frames': count}
        return self.geometries[name]

    def summary(self, trial: dict) -> dict:
        """The trial row shown in lists; a one-sided run carries only the side that ran."""
        ran = next(iter(trial['results'].values()), {})
        return {**{key: trial[key] for key in ('id', 'group', 'created_at', 'params', 'summary')},
                'models': ran.get('models', {})}

    def _validated_run(self, payload: dict) -> dict:
        """Turn a request body into a run, so a bad setting is a 400 now, not a failed job later."""
        if not isinstance(payload, dict):
            raise ValueError('Expected a JSON object')
        group = payload.get('group', '')
        if group not in self.by_group:
            raise KeyError('Unknown pair')
        window_seconds = float(payload.get('window_seconds', self.args.window_seconds))
        hop_seconds = float(payload.get('hop_seconds', self.args.hop_seconds))
        split_windows(100, 10.0, window_seconds, hop_seconds)  # same rules the analysis applies
        sides = payload.get('sides', list(SIDES))
        if not isinstance(sides, list) or not sides or not set(sides) <= set(SIDES):
            raise ValueError(f'sides must be a non-empty subset of {list(SIDES)}')
        prompts, defaults = payload.get('prompts', {}), self.prompts[self.by_group[group]['attribute']]
        if not isinstance(prompts, dict):
            raise ValueError('prompts must be a JSON object')
        clip_frames = self.demo.check_clip_frames(payload.get('clip_frames', self.demo.clip_frames))
        return dict(group=group, window_seconds=window_seconds, hop_seconds=hop_seconds,
                    sides=[side for side in SIDES if side in sides],
                    prompts=asdict(PromptConfig(**(defaults | prompts))), clip_frames=clip_frames)

    def start(self, payload: dict) -> str:
        """Validate a request and run it on a worker thread; one pair at a time."""
        run = self._validated_run(payload)
        with self.lock:
            if self.active is not None:
                raise RuntimeError('Another pair is being analyzed; wait for it to finish')
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = dict(id=job_id, group=run['group'], state='running',
                                     progress='preparing analysis', completed={}, error=None)
            self.active = job_id
            for stale in list(self.jobs)[:-KEPT_JOBS]:   # trials stay on disk; job payloads do not
                del self.jobs[stale]
        threading.Thread(target=self._run, args=(job_id, run), daemon=True).start()
        return job_id

    def delete(self, payload: dict) -> list[str]:
        """Remove stored trials from the list and from disk; returns the ids removed."""
        ids = payload.get('ids') if isinstance(payload, dict) else None
        if not isinstance(ids, list) or not ids or not all(isinstance(trial_id, str) for trial_id in ids):
            raise ValueError('ids must be a non-empty list of trial ids')
        with self.lock:
            if unknown := set(ids) - set(self.trials):
                raise KeyError(f'Unknown trial: {sorted(unknown)}')
            removed = delete_trials(set(ids), self.output)
            for trial_id in ids:
                del self.trials[trial_id]
        print(f'[trials] removed {len(ids)} · {self.output}', flush=True)
        return sorted(removed)

    def _reporter(self, job_id: str, label: str, started: float):
        """A per-side progress callback that updates the job and redraws the terminal bar."""
        def report(row: dict) -> None:
            with self.lock:
                self.jobs[job_id]['progress'] = f"{label.strip()} · window {row['index'] + 1}/{row['total']} done"
            progress_bar(label, row['index'] + 1, row['total'], time.monotonic() - started)

        return report

    def _run_side(self, job_id: str, run: dict, side: str) -> tuple[dict, str]:
        """Analyze one side of the pair and publish it to the job."""
        pair = self.by_group[run['group']]
        label, started = f'{SIDES[side]:<10}', time.monotonic()
        print(f'  {label} {pair[f"{side}_name"][:12]}… analysis started', flush=True)
        result, result_id, _ = self.demo.run(
            self.video_path(pair[f'{side}_name']), run['window_seconds'], run['hop_seconds'],
            progress=self._reporter(job_id, label, started), prompts=run['prompts'],
            clip_frames=run['clip_frames'])
        with self.lock:
            self.jobs[job_id]['completed'][side] = result
        return result, result_id

    def _run(self, job_id: str, run: dict) -> None:
        """The worker thread: run every requested side, then store the trial."""
        results, files, ids = {}, {}, {}
        try:
            print(f"[analysis] {run['group']} · window {run['window_seconds']:g}s · "
                  f"hop {run['hop_seconds']:g}s · MLLM frames {run['prompts']['mllm_frames'] or 'auto'} · "
                  f"CLIP frames {run['clip_frames'] or 'all'} · "
                  f"{'+'.join(SIDES[s] for s in run['sides'])}", flush=True)
            for side in run['sides']:
                results[side], ids[side] = self._run_side(job_id, run, side)
                files[side] = ['/results/results.csv']

            trial = dict(id=job_id, group=run['group'], pair=self.by_group[run['group']],
                         created_at=datetime.now(timezone.utc).isoformat(),
                         params={key: run[key] for key in
                                 ('window_seconds', 'hop_seconds', 'sides', 'prompts', 'clip_frames')},
                         results=results, files=files,
                         summary={side: summarize(result) for side, result in results.items()})
            save_trial(trial, ids, self.output)
            with self.lock:
                self.trials[job_id] = trial
                self.jobs[job_id].update(state='done', trial_id=job_id, progress='done', completed={})
            print(f"[analysis] {run['group']} done · {self.output / 'results.csv'}", flush=True)
        except Exception as error:   # whatever went wrong is reported to the UI, not swallowed
            print(f"\n[analysis] {run['group']} failed: {error}", flush=True)
            with self.lock:
                self.jobs[job_id].update(state='error', error=str(error))
        finally:
            with self.lock:
                self.active = None


def make_handler(browser: Browser):
    """Build the request handler bound to one Browser."""

    class Handler(BaseHTTPRequestHandler):
        """Serves the static UI, the read-only API and the POSTs that start a run or delete trials."""

        def log_message(self, fmt, *args):
            """Log only the 4xx/5xx responses; a polled UI would drown the terminal."""
            if args and str(args[1] if len(args) > 1 else '').startswith(('4', '5')):
                super().log_message(fmt, *args)

        # --- responses ----------------------------------------------------------------
        def send_json(self, value, status=200):
            """Send one JSON body; NaN would not survive the round trip."""
            self.send_bytes(json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
                            'application/json; charset=utf-8', status)

        def send_bytes(self, data: bytes, content_type: str, status=200, filename: str | None = None):
            """Send a complete body, tolerating a browser that navigated away mid-write."""
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            if filename:
                self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(data)

        def send_file(self, path: Path, download=False):
            """Send one file from disk, or 404 when it is not there."""
            if not path.is_file():
                return self.send_json({'error': 'File not found'}, 404)
            return self.send_bytes(path.read_bytes(),
                                   mimetypes.guess_type(path.name)[0] or 'application/octet-stream',
                                   filename=path.name if download else None)

        # --- routing ------------------------------------------------------------------
        def param(self, key: str) -> str:
            """One query parameter, or the empty string."""
            return self.query.get(key, [''])[0]

        def do_GET(self):
            """Dispatch a read request; every route reports failures the same way."""
            parsed = urlparse(self.path)
            route, self.query = unquote(parsed.path), parse_qs(parsed.query)
            try:
                if route in STATIC_ROUTES:
                    return self.send_file(STATIC / STATIC_ROUTES[route])
                if route.startswith(RESULTS_PREFIX):
                    return self.send_result_file(route)
                handler = self.ROUTES.get(route)
                if handler is None:
                    return self.send_json({'error': 'Not found'}, 404)
                return handler(self)
            except KeyError:
                return self.send_json({'error': 'Unknown pair, video, job or trial'}, 404)
            except (ValueError, OSError) as error:
                return self.send_json({'error': str(error)}, 400)

        def do_POST(self):
            """The writes: start an analysis run, or delete stored trials."""
            handler = self.POST_ROUTES.get(self.path)
            if handler is None:
                return self.send_json({'error': 'Not found'}, 404)
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= MAX_REQUEST_BYTES:
                    raise ValueError('Invalid request size')
                return handler(self, json.loads(self.rfile.read(size)))
            except RuntimeError as error:
                return self.send_json({'error': str(error)}, 409)
            except (ValueError, KeyError, TypeError) as error:
                return self.send_json({'error': str(error)}, 400)

        # --- API routes ---------------------------------------------------------------
        def api_config(self):
            """Everything the UI needs to draw its controls before anything has run."""
            return self.send_json(dict(
                attributes=sorted({pair['attribute'] for pair in browser.pairs}),
                meta_cols=META_COLUMNS, filter_cols=FILTER_COLUMNS,
                # A run that leaves its MLLM frames empty gets --mllm-frames, else the window at this rate.
                model=str(browser.args.mllm_model), mllm_frames=browser.args.mllm_frames,
                mllm_frames_per_second=FRAMES_PER_SECOND,
                clip_frames=browser.args.clip_frames, default_prompts=browser.prompts,
                clip_model=CLIP_MODELS[browser.args.clip_model].name,
                clip_reads_all_frames=CLIP_MODELS[browser.args.clip_model].reads_all_frames,
                clip_frames_range=browser.demo.clip_frames_range(),
                defaults={'segment_seconds': browser.args.window_seconds,
                          'hop_seconds': browser.args.hop_seconds,
                          'max_new_tokens': browser.args.max_new_tokens,
                          'thinking_budget': browser.args.thinking_budget,
                          # The engine decides how far a run may raise them; None before it loads.
                          'max_new_tokens_limit': getattr(browser.demo.predictor, 'output_limit', None)}))

        def api_pairs(self):
            """The whole pair list; the UI filters and sorts it client-side."""
            return self.send_json({'pairs': browser.pairs})

        def api_status(self):
            """Whether a job is running and whether the models are loaded."""
            with browser.lock:
                return self.send_json({'active_job': browser.active,
                                       'model': 'ready' if browser.demo.predictor else 'idle'})

        def api_trials(self):
            """The stored trials, newest first, optionally narrowed to one pair."""
            group = self.param('group')
            with browser.lock:
                trials = [browser.summary(trial) for trial in browser.trials.values()
                          if not group or trial['group'] == group]
            return self.send_json({'trials': sorted(trials, key=lambda t: t['created_at'], reverse=True)})

        def api_trial(self):
            """One stored trial, with both sides' windows."""
            with browser.lock:
                trial = browser.trials[self.param('id')]
            return self.send_json(trial)

        def api_job(self):
            """A running job's state. Polled every 1.5s, so it carries only side names."""
            with browser.lock:
                job = browser.jobs[self.param('id')]
                return self.send_json({**job, 'completed': sorted(job['completed'])})

        def api_job_result(self):
            """One finished side of a running job, fetched once rather than polled."""
            with browser.lock:
                return self.send_json(browser.jobs[self.param('id')]['completed'][self.param('side')])

        def api_score(self, payload: dict):
            """Start an analysis run."""
            return self.send_json({'id': browser.start(payload)}, 202)

        def api_delete_trials(self, payload: dict):
            """Delete stored trials, both from the list and from the CSV store."""
            return self.send_json({'deleted': browser.delete(payload)})

        def api_video_info(self):
            """One video's FPS and frame count."""
            return self.send_json(browser.geometry(self.param('name')))

        def api_frame(self):
            """One decoded frame as a JPEG, shrunk to fit the filmstrip."""
            return self.send_bytes(self.frame(self.param('name'), int(self.param('i'))), 'image/jpeg')

        def send_result_file(self, route: str):
            """Download one stored CSV, refusing a path that escapes the result directory."""
            target = (browser.output / route[len(RESULTS_PREFIX):]).resolve()
            if browser.output not in target.parents:
                return self.send_json({'error': 'Invalid path'}, 400)
            return self.send_file(target, download=True)

        def frame(self, name: str, index: int) -> bytes:
            """Decode, shrink and encode one frame of a listed video."""
            if not 0 <= index < browser.geometry(name)['n_frames']:
                raise ValueError('Frame index out of bounds')
            capture = cv2.VideoCapture(str(browser.video_path(name)))
            try:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError('Cannot decode frame')
            finally:
                capture.release()
            height, width = frame.shape[:2]
            scale = min(1, FRAME_MAX_EDGE / max(height, width))
            frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
            ok, encoded = cv2.imencode('.jpg', frame)
            if not ok:
                raise ValueError('Cannot encode frame')
            return encoded.tobytes()

        # Routes are looked up, not chained through, so adding one cannot change another.
        ROUTES = {
            '/api/config': api_config,
            '/api/pairs': api_pairs,
            '/api/status': api_status,
            '/api/trials': api_trials,
            '/api/trial': api_trial,
            '/api/job': api_job,
            '/api/job_result': api_job_result,
            '/api/video_info': api_video_info,
            '/api/frame': api_frame,
        }
        POST_ROUTES = {
            '/api/score': api_score,
            '/api/delete_trials': api_delete_trials,
        }

    return Handler


def serve(demo, args) -> None:
    """Run the UI until interrupted."""
    browser = Browser(demo, args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(browser))
    print(f'[MLLM UI] {len(browser.pairs)} pairs · http://{args.host}:{server.server_port}', flush=True)
    try:
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()
    finally:
        server.server_close()
