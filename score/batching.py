"""Pieces that let several pairs walk at once against one vLLM engine (continuous batching)."""
from __future__ import annotations

from concurrent.futures import Future
import os
import queue
import threading

DEBUG = bool(os.environ.get('SCORE_DEBUG'))


class BatchLLM:
    """Stands in for vllm.LLM: generate() calls from several threads join ONE running engine.

    A request is enqueued as soon as a thread asks and the engine keeps stepping; only the loop thread touches it.
    """

    def __init__(self, llm):
        self.llm, self.requests, self.pending, self.alias = llm, queue.Queue(), {}, {}
        threading.Thread(target=self._loop, daemon=True).start()

    def __getattr__(self, name):   # llm_engine (close), get_default_sampling_params, ...
        return getattr(self.llm, name)

    def generate(self, request, sampling, use_tqdm=False):
        """The engine's output for one request, once that request finishes."""
        future = Future()
        self.requests.put((request, sampling, future))
        return [future.result()]

    def _admit(self, block: bool) -> None:
        """Move the queued requests into the engine; wait for the first one only when `block`."""
        while True:
            try:
                request, sampling, future = self.requests.get(block=block)
            except queue.Empty:
                return
            block = False
            try:
                (request_id,) = self.llm.enqueue([request], [sampling], use_tqdm=False)
                self.pending[request_id] = future
                self.alias[request_id.split('-', 1)[0]] = request_id
                if DEBUG:
                    print(f'[dbg] enqueued {request_id}; pending {len(self.pending)}', flush=True)
            except Exception as error:   # the caller decides whether it is fatal
                future.set_exception(error)

    def _finish(self, output) -> None:
        """Hand a finished output to the thread waiting for it."""
        # enqueue() returns the engine's internal id ("<n>-<suffix>"); outputs carry the external one ("<n>").
        future = self.pending.pop(output.request_id, None)
        if future is None:
            future = self.pending.pop(self.alias.pop(output.request_id, None), None)
        if future is not None:
            future.set_result(output)
        elif DEBUG:
            print(f'[dbg] unmatched output id {output.request_id!r}', flush=True)

    def _loop(self) -> None:
        """Admit requests and step the engine forever."""
        engine = self.llm.llm_engine
        while True:
            self._admit(block=not self.pending)
            try:
                outputs = engine.step()
            except Exception as error:   # every waiting caller gets it
                for future in self.pending.values():
                    future.set_exception(error)
                self.pending.clear()
                continue
            for output in outputs:
                if output.finished:
                    self._finish(output)


class Locked:
    """Serialises every call into an object that one GPU model shares between threads (the CLIP scorer)."""

    def __init__(self, obj):
        self._obj, self._lock = obj, threading.Lock()

    def __getattr__(self, name):
        attribute = getattr(self._obj, name)
        if not callable(attribute):
            return attribute

        def call(*args, **kwargs):
            with self._lock:
                return attribute(*args, **kwargs)
        return call


class Progress:
    """Which side of which pair each thread is walking and at which window."""

    def __init__(self, log_windows: bool):
        self.lock, self.active, self.log_windows = threading.Lock(), {}, log_windows

    def start(self, group: str, tag: str, position: int, total: int):
        """Log the start of one side and return the walk's per-window callback."""
        with self.lock:
            self.active[group] = [tag, 0, 0]
        print(f'start [{position}/{total}] {group} {tag}', flush=True)

        def report(row: dict) -> None:
            with self.lock:
                self.active[group] = [tag, row['index'] + 1, row['total']]
            if self.log_windows:
                score = '-' if row['score'] is None else f"{row['score']:.4f}"
                print(f"    {group} {tag} {row['index'] + 1}/{row['total']} score {score}", flush=True)
        return report

    def finish(self, group: str) -> None:
        """Forget a pair that is no longer being walked."""
        with self.lock:
            self.active.pop(group, None)

    def summary(self) -> str:
        """One line naming every active pair, its side and its window."""
        with self.lock:
            return ' | '.join(f'{group} {tag} {index}/{total}'
                              for group, (tag, index, total) in sorted(self.active.items()))
