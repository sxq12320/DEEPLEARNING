"""Bounded CPU input preparation; model inference stays in the consuming thread."""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor


class OrderedInputPredictor:
    """Overlap deterministic CPU preparation with unchanged single-image inference.

    ``prepare`` must only read/prepare CPU data. ``infer`` runs in the caller's
    thread, so model state, CUDA transfers and forward execution never enter the
    pool. The bound includes the input currently being inferred, rather than
    allowing an extra full-size input outside the prefetch queue.
    """

    def __init__(self, prepare, infer, prefetch_workers=2, max_prefetch=4):
        if int(prefetch_workers) < 0 or int(max_prefetch) < 1:
            raise ValueError("Input workers must be nonnegative and max_prefetch must be positive")
        self.prepare = prepare
        self.infer = infer
        self.max_prefetch = min(4, int(max_prefetch))
        self.prefetch_workers = min(int(prefetch_workers), self.max_prefetch)
        self.last_timing = {}

    def _prepare(self, path):
        started = time.perf_counter()
        prepared = self.prepare(path)
        return prepared, time.perf_counter() - started

    def _infer(self, prepared, prepare_seconds, input_wait_seconds):
        started = time.perf_counter()
        prediction = self.infer(prepared)
        self.last_timing = dict(
            prepare_seconds=prepare_seconds,
            input_wait_seconds=input_wait_seconds,
            inference_seconds=time.perf_counter() - started,
        )
        return prediction

    def __call__(self, path):
        """Keep the existing synchronous ``predictor(path)`` API."""
        prepared, elapsed = self._prepare(path)
        return self._infer(prepared, elapsed, elapsed)

    def iter_predictions(self, paths):
        """Yield predictions in path order and release the pool on close/failure."""
        if not self.prefetch_workers:
            for path in paths:
                yield self(path)
            return

        pending = deque()
        source = iter(paths)
        executor = ThreadPoolExecutor(max_workers=self.prefetch_workers, thread_name_prefix="baseline-input")
        try:
            exhausted = False
            while True:
                while not exhausted and len(pending) < self.max_prefetch:
                    try:
                        path = next(source)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.append(executor.submit(self._prepare, path))
                if not pending:
                    break
                started = time.perf_counter()
                future = pending.popleft()
                prepared, elapsed = future.result()
                waited = time.perf_counter() - started
                del future
                prediction = self._infer(prepared, elapsed, waited)
                del prepared
                # Refill only after the consumer resumes: no extra original-size
                # input is retained beside the bounded set during inference.
                yield prediction
                del prediction
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            pending.clear()
