"""Renderer-owned, bounded materialization of the mandatory Rust scene core."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from ._core import CoreBackendError, CoreTarget
from .layout import FrameLayout, RowLayout

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .scene import CompiledScene


@dataclass(frozen=True, slots=True)
class FrameState:
    presentation_time: float
    audio_time: float
    world_x: float
    camera_speed: float
    layout: FrameLayout
    activity: Mapping[str, tuple[float, float]]
    tile_level: int
    tile_raster_scale: float
    tile_first: int
    tile_last: int
    tile_working_bytes: int


class FrameEvaluator:
    """Compile once and retain at most a batch plus its lookahead (five states)."""

    def __init__(self, scene: CompiledScene):
        started = time.perf_counter()
        self._settings = replace(scene.settings)
        self._part_ids = tuple(part.part_id for part in scene.parts)
        self._target = CoreTarget(scene)
        self._states: OrderedDict[float, FrameState] = OrderedDict()
        self._closed = False
        self.evaluation_count = 0
        self.evaluation_seconds = 0.0
        self.cache_hits = 0
        self.initialization_seconds = time.perf_counter() - started

    def evaluate(self, presentation_time: float) -> FrameState:
        if self._closed:
            raise CoreBackendError("场景计算器已经关闭。")
        if not math.isfinite(presentation_time):
            raise CoreBackendError("场景计算时间必须为有限数值。")
        if presentation_time in self._states:
            self.cache_hits += 1
            self._states.move_to_end(presentation_time)
            return self._states[presentation_time]
        started = time.perf_counter()
        try:
            audio_time = self._settings.audio_time(presentation_time)
            frame, native_rows = self._target.frame(presentation_time, audio_time)
            rows = {}
            activity = {}
            for part_id, row in zip(self._part_ids, native_rows, strict=True):
                rows[part_id] = RowLayout(
                    row.top, row.opacity,
                    (row.indicator_x, row.indicator_y, row.indicator_w, row.indicator_h),
                    (row.bounds_top, row.bounds_bottom), row.icon_size,
                )
                activity[part_id] = (row.activity_level, row.activity_attack)
            state = FrameState(
                presentation_time, audio_time, frame.world_x,
                0.0 if presentation_time < self._settings.intro_delay_seconds else frame.camera_speed,
                FrameLayout(frame.scale, MappingProxyType(rows),
                            (frame.bounds_top, frame.bounds_bottom),
                            (frame.region_top, frame.region_bottom)),
                MappingProxyType(activity), frame.tile_level, frame.tile_raster_scale,
                frame.tile_first, frame.tile_last, int(frame.tile_working_bytes),
            )
            self._states[presentation_time] = state
            if len(self._states) > 5:
                self._states.popitem(last=False)
            self.evaluation_count += 1
            return state
        finally:
            self.evaluation_seconds += time.perf_counter() - started

    def report(self) -> dict:
        return {
            "scene_compute_backend": "rust",
            "scene_compute_abi": 1,
            "scene_compute_library": str(self._target.path),
            "scene_compute_initialization_seconds": self.initialization_seconds,
            "scene_evaluation_count": self.evaluation_count,
            "scene_evaluation_seconds": self.evaluation_seconds,
            "scene_evaluation_cache_hits": self.cache_hits,
        }

    def close(self) -> None:
        if not self._closed:
            self._target.close()
            self._states.clear()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
