"""Shared render effects, bounded batch scheduling and measurement deltas."""

from __future__ import annotations

from collections import deque

from PySide6.QtGui import QColor

from stavellum.domain.models import RenderSettings
from stavellum.presentation.curves import ease
from stavellum.presentation.types import ScenePart


class _ExportTimes:
    """Keep at most one lookahead time when a tile boundary splits a batch."""

    def __init__(self, times, cancel):
        self._times, self.cancel = iter(times), cancel
        self._lookahead = deque()
        self.budget_splits = 0
        self.tile_splits = 0

    def check_cancel(self):
        if self.cancel():
            raise InterruptedError("已取消导出。")

    def next_time(self):
        self.check_cancel()
        return self._lookahead.popleft() if self._lookahead else next(self._times)

    def take_batch(self, renderer):
        times = [self.next_time()]
        limit = renderer.export_batch_limit
        if limit == 1:
            return times
        plan = renderer.batch_plan(times[0])
        self.check_cancel()
        if plan is None:
            self.budget_splits += 1
            return times
        while len(times) < limit:
            try:
                seconds = self.next_time()
            except StopIteration:
                break
            following = renderer.batch_plan(seconds)
            self.check_cancel()
            if following != plan:
                self._lookahead.append(seconds)
                if following is None:
                    self.budget_splits += 1
                else:
                    self.tile_splits += 1
                break
            times.append(seconds)
        return times

    def close(self):
        self._lookahead.clear()
        self._times = iter(())


def _export_native_delta(current, baseline):
    counters = ("gpu_submit_seconds", "fence_wait_seconds", "memory_copy_seconds",
                "scene_evaluation_count", "scene_evaluation_seconds", "scene_evaluation_cache_hits",
                "memory_copy_bytes", "gpu_submitted_frame_count", "owned_readback_frame_count",
                "copied_readback_frame_count", "inplace_format_conversion_seconds",
                "inplace_format_conversion_bytes", "gpu_submission_count",
                "synchronous_readback_seconds")
    values = {key: max(0, current.get(key, 0) - baseline.get(key, 0)) for key in counters}
    before = baseline.get("gpu_batch_size_histogram", {})
    values["gpu_batch_size_histogram"] = {
        size: max(0, count - before.get(size, 0))
        for size, count in current.get("gpu_batch_size_histogram", {}).items()}
    return values


def overlay_opacity(time: float, settings: RenderSettings, *, hide: bool, hold: float) -> float:
    """Absolute presentation-time opacity, including the fully visible hold."""
    opacity = ease(time / settings.overlay_enter_seconds)
    if hide:
        opacity *= 1 - ease((time - settings.overlay_enter_seconds - hold)
                           / settings.overlay_exit_seconds)
    return opacity


def activity_lamp_color(part: ScenePart, activity: tuple[float, float]) -> QColor | None:
    """Brighten the Rack hue, with headroom for a brief note-on highlight."""
    level, attack = activity
    alpha = round(level * 255)
    if alpha <= 0:
        return None
    flash = min(1.0, attack / level)
    hue, saturation, value, _ = QColor(part.activity_color).getHsvF()
    held_value = min(0.82, value * 1.5 + 0.10)
    peak_value = min(1.0, held_value + 0.22)
    color = QColor.fromHsvF(
        hue, saturation * (1 - 0.15 * flash),
        held_value + (peak_value - held_value) * flash,
    )
    color.setAlpha(alpha)
    return color


def logo_opacity(time: float, settings: RenderSettings) -> float:
    """Absolute presentation-time logo state, independent of playback history."""
    if not settings.logo_enabled:
        return 0.0
    opacity = settings.logo_opacity
    if settings.logo_display_mode != "persistent":
        opacity *= ease(time / settings.logo_enter_seconds)
    if settings.logo_display_mode == "intro":
        if time >= settings.logo_enter_seconds + settings.logo_hold_seconds + settings.logo_exit_seconds:
            return 0.0
        opacity *= 1 - ease((time - settings.logo_enter_seconds - settings.logo_hold_seconds)
                           / settings.logo_exit_seconds)
    return max(0.0, min(settings.logo_opacity, opacity))
