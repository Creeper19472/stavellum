"""Serializable layout contracts and absolute-time timeline sampling."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from .curves import _Track, ease


@dataclass(slots=True, frozen=True)
class RowLayout:
    top: float
    opacity: float
    indicator_rect: tuple[float, float, float, float]
    bounds: tuple[float, float]
    icon_size: float


@dataclass(slots=True, frozen=True)
class FrameLayout:
    scale: float
    rows: Mapping[str, RowLayout]
    bounds: tuple[float, float]
    region_bounds: tuple[float, float]


@dataclass(slots=True)
class LayoutKeyframe:
    time: float
    scale: float
    tops: dict[str, float]
    opacities: dict[str, float]


@dataclass(slots=True)
class LayoutTimeline:
    keyframes: list[LayoutKeyframe]
    part_dimensions: dict[str, tuple[float, float]]
    indicator_right: float
    indicator_source_width: float
    indicator_source_height: float
    icon_source_size: float
    icon_source_gap: float
    center: float
    region_top: float
    region_bottom: float
    expanded_top: float
    expanded_bottom: float
    expansion_start: float | None
    expansion_duration: float
    zoom: _Track
    tops: dict[str, _Track]
    opacities: dict[str, _Track]
    tempo_owner_id: str
    tempo_padding: float
    tempo_exit_time: float = math.inf
    times: list[float] = field(default_factory=list)
    urgent_intervals: list[tuple[float, float]] = field(default_factory=list)
    recovery_intervals: list[tuple[float, float]] = field(default_factory=list)

    def region_at(self, time: float) -> tuple[float, float]:
        fraction = (ease((time - self.expansion_start) / self.expansion_duration)
                    if self.expansion_start is not None else 0.0)
        return (self.region_top + (self.expanded_top - self.region_top) * fraction,
                self.region_bottom + (self.expanded_bottom - self.region_bottom) * fraction)

    def scale_at(self, time: float) -> float:
        return math.exp(self.zoom.sample(time).value)

    def opacity_keys_for(self, part_id: str) -> list[tuple[float, float]]:
        return self.opacities[part_id].opacity_keys()

    def padding_at(self, part_id: str, time: float) -> float:
        return self.tempo_padding if part_id == self.tempo_owner_id and time < self.tempo_exit_time else 0.0

    def finish(self) -> None:
        times = sorted({key.time for track in [self.zoom, *self.tops.values(), *self.opacities.values()]
                        for key in track.keys})
        self.times = times
        self.keyframes = [LayoutKeyframe(time, self.scale_at(time),
                          {part_id: track.sample(time).value for part_id, track in self.tops.items()},
                          {part_id: max(0.0, min(1.0, track.sample(time).value)) for part_id, track in self.opacities.items()})
                          for time in times]
