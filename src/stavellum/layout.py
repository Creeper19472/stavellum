"""Compile presentation-time layout, including seekable, continuous zoom curves."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scene import CompiledScene, ScenePart

ZOOM_PAN_FRACTION = 0.5


def ease(value: float) -> float:
    """A clamped quintic transition with zero endpoint velocity and acceleration."""
    value = max(0.0, min(1.0, value))
    return value * value * value * (10 + value * (-15 + 6 * value))


def _polynomial(coefficients: tuple[float, ...], value: float) -> float:
    result = 0.0
    for coefficient in reversed(coefficients):
        result = result * value + coefficient
    return result


def _derivative(coefficients: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(index * value for index, value in enumerate(coefficients) if index)


def _roots(coefficients: tuple[float, ...], left: float = 0.0, right: float = 1.0) -> list[float]:
    """Isolate polynomial roots between its derivative's extrema."""
    while coefficients and abs(coefficients[-1]) < 1e-13:
        coefficients = coefficients[:-1]
    if len(coefficients) <= 1:
        return []
    if len(coefficients) == 2:
        root = -coefficients[0] / coefficients[1]
        return [root] if left < root < right else []
    boundaries = [left, *_roots(_derivative(coefficients), left, right), right]
    roots = []
    for first, last in zip(boundaries, boundaries[1:]):
        low_value = _polynomial(coefficients, first)
        high_value = _polynomial(coefficients, last)
        if abs(low_value) < 1e-11 and left < first < right:
            roots.append(first)
        if low_value * high_value >= 0:
            continue
        low, high = first, last
        for _ in range(45):
            middle = (low + high) / 2
            if _polynomial(coefficients, middle) * low_value > 0:
                low = middle
            else:
                high = middle
        roots.append((low + high) / 2)
    return roots


def _range(coefficients: tuple[float, ...], left: float = 0.0, right: float = 1.0) -> tuple[float, float]:
    values = [_polynomial(coefficients, value)
              for value in (left, *_roots(_derivative(coefficients), left, right), right)]
    return min(values), max(values)


@dataclass(slots=True, frozen=True)
class _CurveKey:
    time: float
    value: float
    velocity: float = 0.0
    acceleration: float = 0.0


def _coefficients(first: _CurveKey, last: _CurveKey) -> tuple[float, ...]:
    duration = last.time - first.time
    first_velocity = first.velocity * duration
    first_acceleration = first.acceleration * duration * duration / 2
    displacement = last.value - first.value - first_velocity - first_acceleration
    velocity = last.velocity * duration - first_velocity - 2 * first_acceleration
    acceleration = last.acceleration * duration * duration - 2 * first_acceleration
    return (first.value, first_velocity, first_acceleration,
            10 * displacement - 4 * velocity + acceleration / 2,
            -15 * displacement + 7 * velocity - acceleration,
            6 * displacement - 3 * velocity + acceleration / 2)


@dataclass(slots=True)
class _Track:
    keys: list[_CurveKey]
    times: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.times = [key.time for key in self.keys]

    def sample(self, time: float) -> _CurveKey:
        index = max(0, bisect_right(self.times, time) - 1)
        first = self.keys[index]
        if index + 1 == len(self.keys) or time < first.time:
            return _CurveKey(time, first.value)
        last = self.keys[index + 1]
        duration = last.time - first.time
        fraction = max(0.0, min(1.0, (time - first.time) / duration))
        coefficients = _coefficients(first, last)
        return _CurveKey(time, _polynomial(coefficients, fraction),
                         _polynomial(_derivative(coefficients), fraction) / duration,
                         _polynomial(_derivative(_derivative(coefficients)), fraction) / duration**2)

    def transition(self, start: float, end: float, target: float) -> None:
        # Restrict the previous polynomial at start; its entire past stays intact.
        current = self.sample(start)
        index = bisect_left(self.times, start)
        self.keys[index:] = [current, _CurveKey(end, target)]
        self.times[index:] = [start, end]

    def opacity_keys(self) -> list[tuple[float, float]]:
        return [(key.time, key.value) for key in self.keys]

    def value_range(self, start: float, end: float) -> tuple[float, float]:
        """Exact bounds of existing segments, including interrupted extrema."""
        values = [self.sample(start).value, self.sample(end).value]
        first_index = max(0, bisect_right(self.times, start) - 1)
        for first, last in zip(self.keys[first_index:], self.keys[first_index + 1:]):
            if first.time >= end:
                break
            length = last.time - first.time
            low = max(0.0, (start - first.time) / length)
            high = min(1.0, (end - first.time) / length)
            if low < high:
                values.extend(_range(_coefficients(first, last), low, high))
        return min(values), max(values)


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


@dataclass(slots=True)
class _PartActivity:
    bars: list[tuple[float, float]]
    bar_lefts: list[float]
    bar_rights: list[float]
    keep: list[tuple[float, float]]
    keep_starts: list[float]
    playing: list[tuple[float, float]]
    playing_starts: list[float]

    def in_view(self, left: float, right: float) -> bool:
        index = bisect_left(self.bar_rights, left)
        return index < len(self.bars) and self.bar_lefts[index] <= right + 1e-8

    def retained(self, time: float, gap: float = 0.0) -> bool:
        index = bisect_right(self.keep_starts, time) - 1
        if index >= 0 and time <= self.keep[index][1] + 1e-9:
            return True
        return index + 1 < len(self.keep) and self.keep[index + 1][0] <= time + gap + 1e-9

    def active(self, time: float) -> bool:
        index = bisect_right(self.playing_starts, time) - 1
        return index >= 0 and time < self.playing[index][1]

    def due(self, time: float, lead: float) -> bool:
        index = bisect_right(self.playing_starts, time + lead) - 1
        return index >= 0 and self.playing[index][1] > time


def _activity(scene: CompiledScene, part: ScenePart) -> _PartActivity:
    seconds_per_beat = 60 / scene.bpm
    offset = scene.settings.score_start_in_audio_sec
    occupied = set()
    intervals = []
    lookahead = 4 * scene.bar_beats * seconds_per_beat
    for note in part.notes:
        first = max(0, math.floor((note.start - offset) / seconds_per_beat / scene.bar_beats))
        last = max(first, math.ceil(((note.end - offset) / seconds_per_beat - 1e-9) / scene.bar_beats) - 1)
        occupied.update(range(first, min(last + 1, len(scene.measure_bounds))))
        intervals.append((note.start - lookahead, note.end))
    keep = []
    for start, end in sorted(intervals):
        if keep and start <= keep[-1][1]:
            keep[-1] = (keep[-1][0], max(keep[-1][1], end))
        else:
            keep.append((start, end))
    bars = [scene.measure_bounds[index] for index in sorted(occupied)]
    playing = []
    for note in sorted(part.notes, key=lambda note: note.start):
        if playing and note.start <= playing[-1][1]:
            playing[-1] = (playing[-1][0], max(playing[-1][1], note.end))
        else:
            playing.append((note.start, note.end))
    return _PartActivity(bars, [item[0] for item in bars], [item[1] for item in bars],
                         keep, [item[0] for item in keep], playing, [item[0] for item in playing])


def _subtract(first: tuple[float, ...], last: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(left - right for left, right in zip(first, last, strict=True))


def _curve_fits(timeline: LayoutTimeline, start: float, end: float, scale: float,
                targets: dict[str, float], visible: list[str], cap: float) -> bool:
    """Check interior extrema, including score-scaled indicators and decorations."""
    log_curve = _coefficients(timeline.zoom.sample(start), _CurveKey(end, math.log(scale)))
    if _range(log_curve)[1] > math.log(cap) + 1e-10:
        return False
    top_curves = {part_id: _coefficients(timeline.tops[part_id].sample(start), _CurveKey(end, top))
                  for part_id, top in targets.items()}
    # Each constraint is polynomial + source_offset * exp(log_zoom) + pixels >= 0.
    constraints = []
    half = timeline.indicator_source_height / 2
    bands = {}
    for part_id in visible:
        curve = top_curves[part_id]
        height, midpoint = timeline.part_dimensions[part_id]
        above = min(-timeline.padding_at(part_id, start), midpoint - half)
        below = max(height, midpoint + half)
        bands[part_id] = (above, below)
        constraints.extend(((curve, above, -1),
                            (tuple(-value for value in curve), -below, 1)))
    for first, last in zip(visible, visible[1:]):
        difference = _subtract(top_curves[last], top_curves[first])
        constraints.append((difference, bands[last][0] - bands[first][1], 0))
    duration = end - start

    def valid_interval(left: float, right: float, depth: int) -> bool:
        log_low, log_high = _range(log_curve, left, right)
        scale_low, scale_high = math.exp(log_low), math.exp(log_high)
        uncertain = False
        region = timeline.region_at(start + duration * left)
        for curve, source, boundary in constraints:
            minimum = _range(curve, left, right)[0]
            minimum += source * (scale_low if source >= 0 else scale_high)
            # Both boundaries expand monotonically, so the first region gives
            # the tightest bound on this interval.
            if boundary:
                minimum += -region[0] if boundary == -1 else region[1]
            if minimum >= -1e-7:
                continue
            for value in (left, (left + right) / 2, right):
                actual = _polynomial(curve, value) + source * math.exp(_polynomial(log_curve, value))
                if boundary:
                    actual_region = timeline.region_at(start + duration * value)
                    actual += -actual_region[0] if boundary == -1 else actual_region[1]
                if actual < -1e-7:
                    return False
            uncertain = True
        if not uncertain or depth == 14:
            return True
        middle = (left + right) / 2
        return valid_interval(left, middle, depth + 1) and valid_interval(middle, right, depth + 1)

    return valid_interval(0.0, 1.0, 0)


def compile_layout(scene: CompiledScene) -> LayoutTimeline:
    """Find viewport events offline; render only evaluates absolute-time curves."""
    settings = scene.settings
    stable_seconds = settings.animation_stable_seconds
    order = [part.part_id for part in scene.parts]
    dimensions = {part.part_id: (part.source_height,
                  (part.staff_centers[0] + part.staff_centers[-1]) / 2 - part.source_top)
                  for part in scene.parts}
    top, bottom = settings.score_top * settings.height, settings.score_bottom * settings.height
    center = (top + bottom) / 2
    populated = [part for part in scene.parts if part.notes] or scene.parts[:1]
    indicator_right = max(1.0, settings.score_left * settings.width - 11 * settings.width / 1920)
    # One source-space size follows every zoom; a small custom left margin may
    # reduce the whole icon/lamp group once, never clamp it on individual frames.
    indicator_ratio = min(1.0, indicator_right / (1207.5 * scene.scale))
    mark = scene.tempo_mark
    if mark is None:
        raise ValueError("场景缺少首小节速度记号。")
    expansion_start = (settings.overlay_enter_seconds + settings.announcement_hold_seconds + settings.overlay_exit_seconds
                       if settings.announcement_auto_hide else None)
    margin = min(0.05, settings.score_top, 1 - settings.score_bottom)
    timeline = LayoutTimeline(
        keyframes=[], part_dimensions=dimensions, indicator_right=indicator_right,
        indicator_source_width=195 * indicator_ratio, indicator_source_height=720 * indicator_ratio,
        icon_source_size=450 * indicator_ratio, icon_source_gap=450 * indicator_ratio,
        center=center, region_top=top, region_bottom=bottom,
        expanded_top=margin * settings.height, expanded_bottom=(1 - margin) * settings.height,
        expansion_start=expansion_start, expansion_duration=settings.reflow_seconds,
        zoom=_Track([_CurveKey(0, math.log(scene.scale))]), tops={},
        opacities={part_id: _Track([_CurveKey(0, 0.0)]) for part_id in order},
        tempo_owner_id=mark.owner_id, tempo_padding=mark.padding,
    )
    activity = {part.part_id: _activity(scene, part) for part in scene.parts}

    def band(part_id: str, scale: float, time: float) -> tuple[float, float]:
        height, midpoint = dimensions[part_id]
        half = timeline.indicator_source_height / 2
        return (min(-timeline.padding_at(part_id, time), midpoint - half) * scale - 1,
                max(height, midpoint + half) * scale + 1)

    def geometry(members: set[str], time: float, cap: float | None = None) -> tuple[float, dict[str, float]]:
        selected = [part_id for part_id in order if part_id in members]
        region = timeline.region_at(time)
        region_center = sum(region) / 2
        if not selected:
            return scene.scale, {}

        def extent(scale: float) -> float:
            return (sum(band(part_id, scale, time)[1] - band(part_id, scale, time)[0] for part_id in selected)
                    + 220 * scale * (len(selected) - 1))

        scale = scene.scale if cap is None else min(scene.scale, cap)
        if extent(scale) > region[1] - region[0]:
            low, high = 0.0, scale
            for _ in range(45):
                middle = (low + high) / 2
                if extent(middle) > region[1] - region[0]:
                    high = middle
                else:
                    low = middle
            scale = max(low, 1e-9)
        y = region_center - extent(scale) / 2
        tops = {}
        for part_id in selected:
            above, below = band(part_id, scale, time)
            tops[part_id] = y - above
            y += below - above + 220 * scale
        return scale, tops

    def viewport(time: float, scale: float) -> tuple[float, float]:
        world_x = scene.camera_x_at(time)
        return world_x - (scene.play_x - scene.body_left) / scale, world_x + (scene.body_right - scene.play_x) / scale

    members = {mark.owner_id}
    while len(order) > 1:
        scale, _ = geometry(members, 0.0)
        left, right = viewport(0.0, scale)
        additions = {part_id for part_id in order if activity[part_id].in_view(left, right)} - members
        if stable_seconds:
            additions.update(part_id for part_id in order if part_id not in members
                             and activity[part_id].due(0.0, settings.enter_seconds + settings.reflow_seconds))
        if not additions:
            break
        members.update(additions)
    scale, initial_tops = geometry(members, 0.0)
    timeline.zoom = _Track([_CurveKey(0, math.log(scale))])
    timeline.tops = {part_id: _Track([_CurveKey(0, initial_tops.get(part_id, center))]) for part_id in order}
    timeline.opacities = {part_id: _Track([_CurveKey(0, float(part_id in members))]) for part_id in order}
    allocated = set(members)
    # The opening state is already stationary; no animation just completed.
    visible_ready = {part_id: -math.inf for part_id in members}

    minimum_scale, _ = geometry({part.part_id for part in populated} | {mark.owner_id}, 0.0)
    clearance = (scene.play_x - scene.body_left) / minimum_scale
    last_right = max((bounds[1] for item in activity.values() for bounds in item.bars), default=scene.axis.xs[-1])
    geometry_end = scene.axis.beat_at(last_right + clearance) * 60 / scene.bpm + settings.score_start_in_audio_sec
    duration = settings.presentation_time(max(0.0, geometry_end, scene.score_duration + settings.score_start_in_audio_sec))
    duration += 3 * (settings.enter_seconds + settings.exit_seconds + settings.reflow_seconds)
    events = [settings.intro_delay_seconds]
    if expansion_start is not None:
        events.extend((expansion_start, expansion_start + settings.reflow_seconds))
        duration = max(duration, expansion_start + settings.reflow_seconds)
    for item in activity.values():
        events.extend(settings.presentation_time(time) for interval in item.keep for time in interval if time >= 0)
    for part in scene.parts:
        events.extend(settings.presentation_time(time) for note in part.notes for time in (note.start, note.end) if time >= 0)
        if stable_seconds:
            events.extend(settings.presentation_time(max(0.0, note.start - settings.enter_seconds - settings.reflow_seconds))
                          for note in part.notes if note.end > 0)
    events = sorted({time for time in events if time > 0})
    pending: set[str] = set()
    departing: dict[str, float] = {}
    motion_end = -stable_seconds
    deferred_reflow = False
    retry_at = math.inf
    radius = max(scene.play_x - scene.body_left, scene.body_right - scene.play_x)

    def decision(time: float) -> tuple[set[str], set[str]]:
        current_scale = timeline.scale_at(time)
        left, right = viewport(time, current_scale)
        visible = {part_id for part_id in order if activity[part_id].in_view(left, right)}
        if stable_seconds:
            visible.update(part_id for part_id in order if activity[part_id].due(
                settings.audio_time(time), settings.enter_seconds + settings.reflow_seconds))
        gap = max(settings.enter_seconds, settings.exit_seconds, settings.reflow_seconds)
        _, future_right = viewport(time + gap, current_scale)
        retained = visible | {part_id for part_id in members
                    if activity[part_id].retained(settings.audio_time(time), gap)
                    or activity[part_id].in_view(left, future_right)}
        if len(order) == 1:
            retained.add(order[0])
        if time < timeline.tempo_exit_time:
            visible.add(mark.owner_id)
            retained.add(mark.owner_id)
        return visible - members, members - retained

    def tempo_right(time: float) -> float:
        return scene.play_x + (mark.x + mark.width - scene.camera_x_at(time)) * timeline.scale_at(time)

    def future_members(start: float, end: float, minimum_scale: float) -> set[str]:
        """A conservative envelope of the moving viewport, without frame scans."""
        left = viewport(start, minimum_scale)[0]
        right = viewport(end, minimum_scale)[1]
        return {part_id for part_id in order if activity[part_id].in_view(left, right)}

    def fade_in(part_id: str, start: float, ready: float, length: float) -> None:
        nonlocal duration
        track = timeline.opacities[part_id]
        if ready > start + 1e-9:
            track.transition(start, ready, 0.0)
        track.transition(ready, ready + length, 1.0)
        visible_ready[part_id] = ready + length
        heappush(events, ready + length)
        if stable_seconds:
            heappush(events, ready + length + stable_seconds)
            duration = max(duration, ready + length + stable_seconds
                           + settings.exit_seconds + settings.reflow_seconds)

    def normal_duration(start: float, target_scale: float) -> tuple[float, bool]:
        length = settings.reflow_seconds
        current = timeline.zoom.sample(start)
        instant_pan = scene.camera_speed_at(start) * math.exp(current.value)
        if (start >= settings.intro_delay_seconds
                and abs(current.velocity) * radius > ZOOM_PAN_FRACTION * instant_pan * (1 + 1e-8)):
            # A deadline can leave a faster initial derivative. C2 continuity
            # requires a bounded recovery, not an impossible slower initial value.
            return length, False
        local_speed = scene.camera_min_speed(max(start, settings.intro_delay_seconds + 1e-8),
                                            max(start + length, settings.intro_delay_seconds + length))
        maximum_length = max(length, 64 * radius * (math.exp(-current.value) + 1 / target_scale)
                             / max(1e-9, local_speed))
        for _ in range(28):
            end = start + length
            moving_start = max(start, settings.intro_delay_seconds + 1e-8)
            if end <= moving_start:
                return length, True
            coefficients = _coefficients(current, _CurveKey(end, math.log(target_scale)))
            first_fraction = (moving_start - start) / length
            low, _ = _range(coefficients, first_fraction, 1.0)
            if low < -40:
                return settings.reflow_seconds, False
            derivative_range = _range(_derivative(coefficients), first_fraction, 1.0)
            peak = max(abs(value) for value in derivative_range) / length
            pan = scene.camera_min_speed(moving_start, end) * math.exp(low)
            ratio = peak * radius / max(1e-9, ZOOM_PAN_FRACTION * pan)
            if ratio <= 1 + 1e-9:
                return length, True
            length *= max(1.05, ratio * 1.02)
            if not math.isfinite(length) or length > maximum_length:
                return settings.reflow_seconds, False
        return settings.reflow_seconds, False

    def reflow(start: float, *, urgent: bool, region_change: bool = False) -> None:
        nonlocal motion_end, duration, deferred_reflow, allocated, retry_at
        if stable_seconds and not urgent and not region_change and start < motion_end + stable_seconds - 1e-9:
            retry_at = motion_end + stable_seconds
            heappush(events, retry_at)
            return
        future = max(start, (expansion_start + settings.reflow_seconds) if expansion_start is not None and start >= expansion_start else start)
        waiting = {part_id for part_id in pending if timeline.opacities[part_id].sample(start).value <= 1e-9}
        deadlines = [settings.presentation_time(note.start) for part in scene.parts if part.part_id in waiting
                     for note in part.notes if settings.presentation_time(note.end) >= start]
        visible = [part_id for part_id in order if timeline.opacities[part_id].sample(start).value > 1e-9]
        # Admissions keep existing slots. Releases may reclaim them only if the
        # resulting geometry will survive its full motion and stable interval.
        target_members = (members | allocated if stable_seconds and urgent else set(members))
        position_only = False
        for _ in range(len(order) + 2):
            cap = timeline.scale_at(start) if position_only else None
            target_scale, _ = geometry(target_members, future, cap)
            length, speed_limited = normal_duration(start, target_scale)
            if not speed_limited and not urgent and start < motion_end - 1e-9:
                deferred_reflow = True
                return
            normal_length = length
            available = max(2e-5, min(deadlines, default=start + length + settings.enter_seconds) - start)
            fade_length = min(settings.enter_seconds, available * settings.enter_seconds / (settings.enter_seconds + settings.reflow_seconds))
            if waiting:
                length = min(length, max(1e-5, available - fade_length))
            for _ in range(35):
                end = start + length
                target_scale, target_tops = geometry(target_members, end, cap)
                full_targets = {part_id: target_tops.get(part_id, timeline.tops[part_id].sample(start).value) for part_id in order}
                if _curve_fits(timeline, start, end, target_scale, full_targets, visible, scene.scale):
                    break
                if not urgent and start < motion_end - 1e-9:
                    deferred_reflow = True
                    return
                # Only an admission with a musical deadline may accelerate a
                # constrained transition; optional motion keeps its speed cap.
                length *= 0.7 if urgent else 1.2
            else:
                raise ValueError("无法在谱区内安排连续的分谱重排。")
            if not stable_seconds:
                break
            log_curve = _coefficients(timeline.zoom.sample(start), _CurveKey(end, math.log(target_scale)))
            minimum = max(1e-9, math.exp(max(-40, _range(log_curve)[0])))
            upcoming = future_members(start, end + stable_seconds - 1e-8, minimum) - target_members
            if not urgent:
                # A release whose completion would outlast other imminent
                # releases is another transient target. Keep all current slots
                # and reconsider the combined removal at their real events.
                gap = max(settings.enter_seconds, settings.exit_seconds, settings.reflow_seconds)
                unstable_release = False
                for check_time in (end, end + stable_seconds - 1e-8):
                    left, right = viewport(check_time, target_scale)
                    if any(not activity[part_id].in_view(left, right)
                           and not activity[part_id].retained(settings.audio_time(check_time), gap)
                           and not (part_id == mark.owner_id and check_time < timeline.tempo_exit_time)
                           for part_id in members):
                        unstable_release = True
                        break
                if unstable_release:
                    if not position_only:
                        # Skipping an unstable zoom must not leave a small
                        # visible group at the edge of old, hidden slots. Try
                        # centering at the existing scale with a short reflow.
                        position_only = True
                        continue
                    if not region_change:
                        retry_at = math.inf
                        return
            if not upcoming:
                break
            if not urgent:
                if not position_only:
                    position_only = True
                    continue
                # A nearby admission needs these complete slots soon. At the
                # unchanged scale its envelope only covers the short reflow
                # and hold, so preallocation cannot linger far into the song.
                target_members.update(upcoming)
                continue
            target_members.update(upcoming)
        else:
            raise ValueError("分谱预留槽位未能收敛。")
        if stable_seconds and not visible and not waiting:
            # No motion was committed: prospective slots are not ready and
            # must go through admission validation if music resumes later.
            allocated = set(members)
            retry_at = math.inf
            return
        unchanged = (abs(math.log(target_scale) - timeline.zoom.sample(start).value) < 1e-10
                     and abs(timeline.zoom.sample(start).velocity) < 1e-10
                     and abs(timeline.zoom.sample(start).acceleration) < 1e-10
                     and all(abs(value - timeline.tops[part_id].sample(start).value) < 1e-7
                             and abs(timeline.tops[part_id].sample(start).velocity) < 1e-7
                             and abs(timeline.tops[part_id].sample(start).acceleration) < 1e-7
                             for part_id, value in target_tops.items()))
        if stable_seconds and unchanged:
            allocated = target_members
            for part_id in waiting:
                fade_in(part_id, start, start, min(settings.enter_seconds, available))
            retry_at = math.inf
            return
        # Never-visible rows can be positioned in their eventual slot directly;
        # their first fade still waits until that complete slot is ready.
        for part_id in target_members:
            if timeline.opacities[part_id].value_range(0.0, start)[1] <= 1e-9:
                timeline.tops[part_id] = _Track([_CurveKey(0, full_targets[part_id])])
        timeline.zoom.transition(start, end, math.log(target_scale))
        for part_id, value in full_targets.items():
            timeline.tops[part_id].transition(start, end, value)
        for part_id in waiting:
            fade_in(part_id, start, end, fade_length)
        timeline.urgent_intervals = [(first, min(last, start)) for first, last in timeline.urgent_intervals
                                     if first < start]
        timeline.recovery_intervals = [(first, min(last, start)) for first, last in timeline.recovery_intervals
                                       if first < start]
        if not speed_limited or (urgent and length < normal_length - 1e-9):
            timeline.urgent_intervals.append((start, end))
        if not speed_limited:
            timeline.recovery_intervals.append((start, end))
        motion_end = end
        heappush(events, end)
        if stable_seconds:
            heappush(events, end + stable_seconds)
        allocated = target_members
        retry_at = math.inf
        duration = max(duration, end + fade_length)
        deferred_reflow = False

    previous = 0.0
    sample_index = 1
    while previous < duration - 1e-9:
        # Deadline shortening can finish inside the interval whose admission
        # was just bisected. Absolute-time sampling has already passed these
        # events; leaving them at the heap head would prevent any progress.
        while events and events[0] <= previous + 1e-10:
            heappop(events)
        regular = sample_index / 240
        sample = min(regular, events[0] if events else math.inf, duration)
        if regular <= sample + 1e-10:
            sample_index += 1
        if sample <= previous + 1e-10:
            continue
        tempo_exit_event = False
        if math.isinf(timeline.tempo_exit_time) and tempo_right(sample) <= scene.body_left:
            low, high = previous, sample
            if tempo_right(low) > scene.body_left:
                while high - low > 1e-8:
                    middle = (low + high) / 2
                    if tempo_right(middle) > scene.body_left:
                        low = middle
                    else:
                        high = middle
            timeline.tempo_exit_time = high
            # Restart at the exact geometric exit; preserve the existing C2
            # curves there and reclaim only the now-invisible decoration band.
            sample = high
            tempo_exit_event = True
        explicit_event = bool(events and events[0] <= sample + 1e-10)
        while events and events[0] <= sample + 1e-10:
            heappop(events)
        entering, leaving = decision(sample)
        start = sample
        if entering and not tempo_exit_event and not decision(previous)[0]:
            low, high = previous, sample
            while high - low > 1e-6:
                middle = (low + high) / 2
                if decision(middle)[0]:
                    high = middle
                else:
                    low = middle
            start = high
            entering, leaving = decision(start)
        removed = {part_id for part_id, end in departing.items() if end <= start + 1e-9 and part_id in leaving}
        for part_id in removed:
            members.remove(part_id)
            pending.discard(part_id)
            del departing[part_id]
        for part_id in set(departing) - leaving:
            fade_in(part_id, start, start, settings.enter_seconds)
            del departing[part_id]
        for part_id in leaving - removed - departing.keys():
            if part_id in pending:
                continue
            if stable_seconds:
                # Stability belongs to each opacity track independently of
                # layout motion. A short hidden interval is omitted entirely.
                if start < visible_ready.get(part_id, 0.0) + stable_seconds - 1e-9:
                    continue
                horizon = start + settings.exit_seconds + stable_seconds - 1e-8
                minimum = math.exp(max(-40, timeline.zoom.value_range(start, horizon)[0]))
                if part_id in future_members(start, horizon, minimum):
                    continue
            end = start + settings.exit_seconds
            timeline.opacities[part_id].transition(start, end, 0.0)
            departing[part_id] = end
            heappush(events, end)
        pending = {part_id for part_id in pending if timeline.opacities[part_id].sample(start).value < 1 - 1e-9}
        if entering:
            members.update(entering)
            pending.update(entering)
        expansion_event = expansion_start is not None and abs(start - expansion_start) < 1e-8
        expansion_complete = expansion_start is not None and abs(start - expansion_start - settings.reflow_seconds) < 1e-8
        needs_geometry = bool(entering)
        if stable_seconds and entering and entering <= allocated and not expansion_event and not tempo_exit_event:
            ready = max(start, motion_end)
            deadline = min((settings.presentation_time(note.start) for part in scene.parts if part.part_id in entering
                            for note in part.notes if settings.presentation_time(note.end) >= start),
                           default=ready + settings.enter_seconds)
            if ready < deadline - 1e-5:
                length = min(settings.enter_seconds, deadline - ready)
                for part_id in entering:
                    fade_in(part_id, start, ready, length)
                needs_geometry = False
        reservation_event = (stable_seconds and explicit_event and not entering and allocated != members
                             and start >= motion_end + stable_seconds - 1e-9)
        if (needs_geometry or removed or expansion_event or tempo_exit_event or reservation_event
                or start >= retry_at - 1e-9 or (deferred_reflow and start >= motion_end - 1e-9)):
            reflow(start, urgent=bool(needs_geometry or pending), region_change=expansion_event)
        elif expansion_complete and start >= motion_end - 1e-9:
            reflow(start, urgent=False, region_change=True)
        previous = sample
    timeline.finish()
    return timeline
