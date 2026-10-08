"""Infer hairpins from the effective gain of the source channels and mixer path."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass

from stavellum.domain.models import (
    AutomationPoint,
    Diagnostic,
    PartMapping,
    ProjectIR,
    VolumeAutomation,
    VolumeRoute,
)


@dataclass(frozen=True, slots=True)
class DynamicSpan:
    start_tick: float
    end_tick: float
    direction: str
    change_db: float


class UnsupportedCurve(ValueError):
    """A curve or routing ambiguity cannot safely be interpreted as dynamics."""


def interpolate_point(first: AutomationPoint, last: AutomationPoint, fraction: float) -> float:
    """Evaluate the original FL point shape; unknown shapes remain explicit."""
    fraction = max(0.0, min(1.0, fraction))
    # FL stores the preceding segment's shape on its right-hand endpoint.
    mode = last.metadata & 0xFF
    flags = last.metadata & ~0xFF
    if flags not in (0, 0x01000000, 0x02000000, 0xFF000000) or mode not in (0, 2):
        raise UnsupportedCurve("尚未验证的自动化曲线形状")
    if mode == 2:
        return first.value if fraction < 1 else last.value
    if not -1 <= last.tension <= 1:
        raise UnsupportedCurve("自动化曲线张力超出已验证范围")
    # FL native renders at positive and negative tension confirm this shape.
    # High-byte flags are cached direction state, not the curve-mode byte.
    exponent = -math.log(1_000_000) * last.tension
    shape = fraction if abs(exponent) < 1e-8 else math.expm1(exponent * fraction) / math.expm1(exponent)
    return first.value + (last.value - first.value) * shape


def volume_gain(value: float) -> float:
    """FL's native volume knob: 0 is silence, 1 is unity, 1.25 is +5.6 dB."""
    return max(0.0, math.expm1(value * math.log(11)) / 10)


def _recorded_trends(clip: VolumeAutomation, ppq: int) -> list[tuple[float, float]]:
    """Identify densely recorded fader motion without changing its held samples."""
    if clip.source_kind != "pattern" or not clip.supported:
        return []
    result = []
    run_start = steps = direction = 0
    for index, (first, last) in enumerate(zip(clip.points, clip.points[1:])):
        current = _sign(last.value - first.value)
        eligible = (0 < last.tick - first.tick <= ppq / 16 + 1e-8 and current
                    and last.metadata & 0xFF == 2
                    and last.metadata & ~0xFF in (0, 0x01000000, 0x02000000, 0xFF000000))
        if eligible and (not steps or current == direction):
            if not steps:
                run_start = index
            steps += 1
            direction = current
            continue
        if steps >= 4:
            result.append((clip.points[run_start].tick, first.tick))
        steps, direction = (1, current) if eligible else (0, 0)
        run_start = index
    if steps >= 4:
        result.append((clip.points[run_start].tick, clip.points[-1].tick))
    return result


def _clip_value(clip: VolumeAutomation, tick: float,
                trend_runs: list[tuple[float, float]] | None = None) -> float:
    if not clip.points:
        raise UnsupportedCurve("音量自动化没有控制点")
    local = tick - clip.start_tick + clip.source_offset_tick
    points = clip.points
    index = bisect_right([point.tick for point in points], local) - 1
    if index < 0:
        value = points[0].value
    elif index + 1 == len(points):
        value = points[-1].value
    else:
        first, last = points[index:index + 2]
        fraction = (local - first.tick) / (last.tick - first.tick)
        if trend_runs and any(start <= first.tick and last.tick <= end for start, end in trend_runs):
            # This is an annotation trend through dense recorded steps. The
            # preserved source curve still plays the actual held values.
            value = first.value + (last.value - first.value) * fraction
        else:
            value = interpolate_point(first, last, fraction)
    return (clip.minimum + (clip.maximum - clip.minimum) * value) * clip.value_scale


class _Envelope:
    def __init__(self, controls: dict[str, list[VolumeAutomation]], route: VolumeRoute,
                 recorded_trends: dict[str, list[tuple[float, float]]] | None = None):
        self.controls = controls
        self.route = route
        self.recorded_trends = recorded_trends or {}

    def gain_db(self, tick: float) -> float:
        if not self.route.supported:
            raise UnsupportedCurve(self.route.unsupported_reason or "音量路由尚未确认")
        result = 0.0
        for target in self.route.control_ids:
            clips = self.controls.get(target, [])
            active = [clip for clip in clips if clip.start_tick <= tick < clip.end_tick]
            if len(active) > 1:
                raise UnsupportedCurve("同一音量目标有重叠自动化片段，播放优先级尚未确认")
            if active:
                clip = active[0]
                if not clip.supported:
                    raise UnsupportedCurve(clip.unsupported_reason or "音量自动化尚未确认")
                value = _clip_value(clip, tick, self.recorded_trends.get(clip.automation_id))
            else:
                previous = [clip for clip in clips if clip.end_tick <= tick]
                if previous:
                    last_end = max(clip.end_tick for clip in previous)
                    finished = [clip for clip in previous if clip.end_tick == last_end]
                    if len(finished) != 1:
                        raise UnsupportedCurve("同一音量目标的片段结束顺序不明确")
                    if not finished[0].supported:
                        raise UnsupportedCurve(finished[0].unsupported_reason or "已结束的音量自动化尚未确认")
                    value = _clip_value(finished[0], finished[0].end_tick)
                elif target in self.route.initial_values:
                    value = self.route.initial_values[target]
                else:
                    raise UnsupportedCurve("音量控制器的初始值未知")
            gain = volume_gain(value)
            if gain == 0:
                return -60.0
            result += 20 * math.log10(gain)
        return max(-60.0, result)


@dataclass(frozen=True, slots=True)
class _Section:
    start: float
    end: float
    direction: int
    reason: str = ""


def _sign(value: float) -> int:
    return 1 if value > 1e-7 else -1 if value < -1e-7 else 0


def _inside(tick: float, start: float, end: float) -> float:
    margin = min((end - start) / 1000, max(1e-7, (end - start) * 1e-8))
    return min(end - margin, max(start + margin, tick))


def _sections(envelope: _Envelope, knots: list[float]) -> list[_Section]:
    """Split continuous gain at changes of direction, including composed curves."""
    result = []
    for start, end in zip(knots, knots[1:]):
        if end - start <= 1e-8:
            continue
        try:
            # Restrict derivatives to this segment; reset jumps are not ramps.
            epsilon = min((end - start) / 1000, max(1e-7, (end - start) * 1e-6))

            def derivative(tick: float) -> float:
                left = _inside(tick - epsilon, start, end)
                right = _inside(tick + epsilon, start, end)
                return ((envelope.gain_db(right) - envelope.gain_db(left)) / (right - left)
                        if right > left else 0.0)

            samples = [start + (end - start) * index / 16 for index in range(17)]
            values = [derivative(tick) for tick in samples]
            turning = [samples[index] for index in range(1, 16)
                       if abs(values[index]) < 1e-9 and values[index - 1] * values[index + 1] < 0]
            for left, right, first, last in zip(samples, samples[1:], values, values[1:]):
                if first * last >= 0:
                    continue
                for _ in range(45):
                    middle = (left + right) / 2
                    current = derivative(middle)
                    if first * current > 0:
                        left, first = middle, current
                    else:
                        right = middle
                turning.append((left + right) / 2)
            boundaries = [start, *sorted(set(turning)), end]
            for left, right in zip(boundaries, boundaries[1:]):
                difference = (envelope.gain_db(_inside(right, start, end))
                              - envelope.gain_db(_inside(left, start, end)))
                result.append(_Section(left, right, _sign(difference)))
        except UnsupportedCurve as exc:
            result.append(_Section(start, end, 0, str(exc)))
    return result


def _continuous(envelope: _Envelope, tick: float, span: float) -> bool:
    epsilon = min(max(1e-7, span * 1e-8), span / 1000)
    try:
        return abs(envelope.gain_db(tick - epsilon) - envelope.gain_db(tick + epsilon)) < 1e-4
    except UnsupportedCurve:
        return False


def _merge_sections(envelope: _Envelope, sections: list[_Section]) -> list[_Section]:
    result = []
    for section in sections:
        if section.direction == 0 or section.reason:
            continue
        if (result and result[-1].end == section.start and result[-1].direction == section.direction
                and _continuous(envelope, section.start, min(result[-1].end - result[-1].start,
                                                             section.end - section.start))):
            result[-1] = _Section(result[-1].start, section.end, section.direction)
        else:
            result.append(section)
    return result


def infer_dynamics(project: ProjectIR, mapping: PartMapping,
                   activity: list[tuple[str, float, float]] | None = None) -> tuple[list[DynamicSpan], list[Diagnostic]]:
    """Keep source endpoints; remove only conflicts between currently sounding sources."""
    if not mapping.auto_dynamics or not project.volume_automations:
        return [], []
    if activity is None:
        activity = [(event.track_id, float(event.start_tick), float(event.end_tick))
                    for event in project.notes if event.track_id in mapping.track_ids
                    and event.pitch not in mapping.keyswitches and event.duration_tick > 0
                    and 0 <= (event.pitch if mapping.percussion else event.pitch + mapping.transpose) <= 127]
    if not activity:
        return [], []
    source_ids = sorted({track_id for track_id, _, _ in activity})
    routes_by_id = {route.track_id: route for route in project.volume_routes}
    controls = defaultdict(list)
    for clip in project.volume_automations:
        controls[clip.target_id].append(clip)
    relevant_targets = {target for track_id in source_ids if track_id in routes_by_id
                        for target in routes_by_id[track_id].control_ids}
    relevant_clips = [clip for clip in project.volume_automations if clip.target_id in relevant_targets]
    if not relevant_clips:
        unknown = [track_id for track_id in source_ids if track_id not in routes_by_id
                   or not routes_by_id[track_id].supported]
        return [], ([Diagnostic("warning", "volume-dynamics-unsupported",
                                 f"{mapping.name}：来源音量路由尚未确认，未推断力度渐变。", mapping.part_id)]
                    if unknown else [])
    first = min(clip.start_tick for clip in relevant_clips)
    last = max(clip.end_tick for clip in relevant_clips)
    knots = {first, last}
    for clip in relevant_clips:
        knots.update((clip.start_tick, clip.end_tick))
        knots.update(clip.start_tick + point.tick - clip.source_offset_tick for point in clip.points
                     if clip.start_tick < clip.start_tick + point.tick - clip.source_offset_tick < clip.end_tick)
    knots = sorted(knots)
    envelopes = {}
    sections = {}
    candidates = []
    recorded_trends = {clip.automation_id: _recorded_trends(clip, project.ppq) for clip in relevant_clips}
    for track_id in source_ids:
        route = routes_by_id.get(track_id, VolumeRoute(track_id, [], {}, False, "音量路由未知"))
        envelope = envelopes[track_id] = _Envelope(controls, route, recorded_trends)
        sections[track_id] = _sections(envelope, knots)
        candidates.extend(_merge_sections(envelope, sections[track_id]))
    unsupported = {section.reason for track_id, values in sections.items() for section in values
                   if section.reason and any(source == track_id and left < section.end and right > section.start
                                             for source, left, right in activity)}
    if not candidates:
        reasons = sorted(unsupported)
        return [], ([Diagnostic("warning", "volume-dynamics-unsupported",
                                 f"{mapping.name}：跳过音量渐变记谱：{'；'.join(reasons)}。", mapping.part_id)]
                    if reasons else [])

    # Union candidates with the same direction, while retaining control resets.
    merged = []
    for candidate in sorted(candidates, key=lambda item: (item.direction, item.start, item.end)):
        if (merged and merged[-1].direction == candidate.direction and candidate.start < merged[-1].end):
            merged[-1] = _Section(merged[-1].start, max(merged[-1].end, candidate.end), candidate.direction)
        elif merged and merged[-1] == candidate:
            continue
        else:
            merged.append(candidate)
    conflicts = set()
    spans = []
    minimum_span = project.ppq * 4 / mapping.quantization * (2 / 3 if mapping.triplets else 1)
    for candidate in merged:
        boundaries = {candidate.start, candidate.end}
        boundaries.update(value for _, start, end in activity for value in (start, end)
                          if candidate.start < value < candidate.end)
        boundaries.update(value for values in sections.values() for section in values
                          for value in (section.start, section.end) if candidate.start < value < candidate.end)
        boundaries = sorted(boundaries)
        blocks = []
        for start, end in zip(boundaries, boundaries[1:]):
            midpoint = (start + end) / 2
            active = {track_id for track_id, left, right in activity if left < end and right > start}
            blocked = False
            directions = set()
            for track_id in active:
                section = next((section for section in sections[track_id]
                                if section.start <= midpoint < section.end), None)
                if section is None or section.reason:
                    unsupported.add(section.reason if section else "音量曲线范围未知")
                    blocked = True
                elif section.direction != candidate.direction:
                    blocked = True
                if section is not None and not section.reason:
                    directions.add(section.direction)
            if len(directions) > 1:
                conflicts.update(active)
            if blocked:
                blocks.append((start, end))
        for boundary in boundaries[1:-1]:
            active = {track_id for track_id, left, right in activity if left < boundary < right}
            if any(not _continuous(envelopes[track_id], boundary, candidate.end - candidate.start)
                   for track_id in active):
                blocks.append((boundary, boundary))
        remaining = [(candidate.start, candidate.end)]
        for left, right in blocks:
            remaining = [(start, end) for first_start, first_end in remaining
                         for start, end in ((first_start, min(first_end, left)), (max(first_start, right), first_end))
                         if end - start > 1e-8]
        for start, end in remaining:
            active = {track_id for track_id, left, right in activity if left < end and right > start}
            if not active or end - start < minimum_span - 1e-8:
                continue
            changes = []
            for track_id in active:
                envelope = envelopes[track_id]
                try:
                    changes.append(abs(envelope.gain_db(_inside(end, start, end))
                                       - envelope.gain_db(_inside(start, start, end))))
                except UnsupportedCurve:
                    pass
            change = max(changes, default=0)
            if change >= 1 - 1e-6:
                spans.append(DynamicSpan(start, end, "crescendo" if candidate.direction > 0 else "diminuendo", change))
    if conflicts:
        diagnostics = [Diagnostic("warning", "volume-dynamics-conflict",
                                  f"{mapping.name}：合并分谱的活跃来源音量变化不一致，已跳过冲突区间。", mapping.part_id)]
    else:
        diagnostics = []
    if unsupported:
        diagnostics.append(Diagnostic("warning", "volume-dynamics-unsupported",
                                      f"{mapping.name}：跳过音量渐变记谱：{'；'.join(sorted(unsupported))}。", mapping.part_id))
    spans = sorted(set(spans), key=lambda span: (span.start_tick, span.end_tick, span.direction))
    if spans:
        diagnostics.append(Diagnostic("info", "auto-volume-dynamics",
                                      f"{mapping.name}：从音量自动化识别渐强 {sum(span.direction == 'crescendo' for span in spans)} 处、渐弱 {sum(span.direction == 'diminuendo' for span in spans)} 处。", mapping.part_id))
    return spans, diagnostics
