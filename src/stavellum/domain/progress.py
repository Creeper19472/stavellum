"""Serializable export snapshots and estimates based on completed work."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExportProgress:
    phase: str
    message: str
    elapsed_seconds: float
    attempt: int = 1
    completed: int = 0
    total: int = 0
    unit: str = ""
    fps: float = 0
    render_backend: str = ""
    video_encoder: str = ""
    fallback_reason: str = ""
    part_name: str = ""
    page: int = 0
    pages: int = 0


class ProgressReporter:
    """Report phase changes immediately and bound repetitive worker messages."""

    def __init__(self, callback: Callable[[ExportProgress], None] | None,
                 *, clock: Callable[[], float] | None = None) -> None:
        self.callback = callback
        self._clock = clock or time.perf_counter
        self._started = self._clock()
        self._reported_at = float("-inf")
        self._key = None
        self._fields: dict = {}

    def emit(self, phase: str, message: str = "", *, force: bool = False,
             **fields) -> ExportProgress | None:
        self._fields.update(fields)
        if self.callback is None:
            return None
        now = self._clock()
        key = (phase, self._fields.get("attempt", 1), self._fields.get("part_name", ""))
        if not force and key == self._key and now - self._reported_at < .25:
            return None
        snapshot = ExportProgress(phase, message, max(0.0, now - self._started),
                                  **self._fields)
        self._key = key
        self._reported_at = now
        self.callback(snapshot)
        return snapshot


@dataclass(frozen=True, slots=True)
class ProgressEstimate:
    speed: float | None = None
    remaining_seconds: float | None = None


class ExportEstimator:
    """Use worker time for throughput and local time to detect stale samples."""

    def __init__(self, operation: str = "video") -> None:
        self.operation = operation
        self._detail: ExportProgress | None = None
        self._samples: deque[tuple[float, int]] = deque()
        self._received_at = 0.0
        self._activity_at = 0.0
        self._parts_started: float | None = None

    def update(self, detail: ExportProgress, now: float) -> ProgressEstimate:
        previous = self._detail
        reset = (previous is None or previous.attempt != detail.attempt
                 or detail.completed < previous.completed)
        if reset:
            self._samples.clear()
            self._parts_started = None
        activity = (reset or previous.completed != detail.completed
                    or previous.part_name != detail.part_name or previous.page != detail.page)
        if activity:
            self._activity_at = now
        if detail.phase == "frames":
            if previous is None or previous.phase != "frames":
                self._samples.clear()
                self._activity_at = now
            if self._samples and detail.elapsed_seconds < self._samples[-1][0]:
                self._samples.clear()
            if self._samples and detail.elapsed_seconds == self._samples[-1][0]:
                self._samples[-1] = (detail.elapsed_seconds, detail.completed)
            else:
                self._samples.append((detail.elapsed_seconds, detail.completed))
            cutoff = detail.elapsed_seconds - 5
            while len(self._samples) > 2 and self._samples[1][0] <= cutoff:
                self._samples.popleft()
        elif detail.unit == "parts" and detail.phase in {"xml", "pdf", "save"}:
            if self._parts_started is None:
                self._parts_started = detail.elapsed_seconds
        self._detail = detail
        self._received_at = now
        return self.estimate(now)

    def estimate(self, now: float) -> ProgressEstimate:
        detail = self._detail
        if detail is None or now - self._activity_at >= 5:
            return ProgressEstimate()
        elapsed = detail.elapsed_seconds + max(0.0, now - self._received_at)
        if detail.phase == "frames" and self.operation == "video":
            if not self._samples:
                return ProgressEstimate()
            cutoff = elapsed - 5
            samples = list(self._samples)
            start, completed = samples[0]
            for (left, left_count), (right, right_count) in zip(samples, samples[1:]):
                if left <= cutoff < right:
                    start = cutoff
                    completed = left_count + (right_count - left_count) * (cutoff - left) / (right - left)
                    break
                if right <= cutoff:
                    start, completed = right, right_count
            if elapsed - start < 1:
                return ProgressEstimate()
            speed = (detail.completed - completed) / (elapsed - start)
        elif (self.operation == "parts" and detail.phase in {"xml", "pdf", "save"}
              and self._parts_started is not None and detail.completed > 0):
            if elapsed <= self._parts_started:
                return ProgressEstimate()
            speed = detail.completed / (elapsed - self._parts_started)
        else:
            return ProgressEstimate()
        if speed <= 0:
            return ProgressEstimate()
        remaining = ((detail.total - detail.completed) / speed
                     if detail.total > detail.completed else None)
        return ProgressEstimate(speed, remaining)
