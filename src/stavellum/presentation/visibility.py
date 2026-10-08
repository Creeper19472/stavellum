"""Compile and restore visibility from shared absolute-time timelines."""

from __future__ import annotations

import math
from bisect import bisect_right

from stavellum.domain.models import RenderSettings
from stavellum.graphics.musicfont import metronome_renderer
from stavellum.graphics.typography import _layout

from .camera import compile_camera
from .curves import ease
from .layout import compile_layout
from .types import CompiledScene, ScenePart, TempoMark, VisibleSpan


def smoothstep(value: float) -> float:
    return ease(value)


def part_state(part: ScenePart, time: float, settings: RenderSettings) -> tuple[float, float]:
    """Opacity and occupancy at presentation time, independent of frame history."""
    if part.opacity_curve is not None:
        opacity = max(0.0, min(1.0, part.opacity_curve.sample(time).value))
        return opacity, 1.0 if opacity > 0 else 0.0
    if part.opacity_keys:
        index = max(0, bisect_right(part.opacity_keys, (time, math.inf)) - 1)
        start, first = part.opacity_keys[index]
        end, last = part.opacity_keys[min(index + 1, len(part.opacity_keys) - 1)]
        fraction = ease((time - start) / (end - start)) if end > start else 0.0
        opacity = first + (last - first) * fraction
        # A fading row retains a complete slot; FrameLayout owns reflow geometry.
        return opacity, 1.0 if opacity > 0 else 0.0
    opacity = occupancy = 0.0
    for span in part.spans:
        if time < span.start or time > span.end + max(settings.exit_seconds, settings.reflow_seconds):
            continue
        enter = 1.0 if span.initial else smoothstep((time - span.start) / settings.enter_seconds)
        occupy_enter = 1.0 if span.initial else smoothstep((time - span.start) / settings.reflow_seconds)
        leave = 1 - smoothstep((time - span.end) / settings.exit_seconds)
        occupy_leave = 1 - smoothstep((time - span.end) / settings.reflow_seconds)
        opacity = max(opacity, enter * leave)
        occupancy = max(occupancy, occupy_enter * occupy_leave)
    return opacity, occupancy


def compile_visibility(scene: CompiledScene) -> None:
    """Visibility and zoom share absolute presentation time with the camera."""

    glyph = metronome_renderer()
    glyph_width = 360 * glyph.viewBoxF().width() / glyph.viewBoxF().height()
    label = "= " + f"{scene.bpm:.6f}".rstrip("0").rstrip(".")
    text_layout = _layout(label, 270)
    label_width = text_layout.lineAt(0).naturalTextWidth()
    scene.tempo_mark = TempoMark(scene.parts[0].part_id, scene.axis.x_at(0), label,
                                glyph_width + 90 + label_width + 2, glyph_width)
    if scene.camera is None:
        left, right = scene.play_corridor
        scene.camera = compile_camera(scene.axis, scene.bpm, scene.settings.score_start_in_audio_sec,
                                      (right - left) / (2 * scene.scale))
    scene.layout = compile_layout(scene)
    restore_visibility(scene)


def restore_visibility(scene):
    for part in scene.parts:
        part.opacity_curve = scene.layout.opacities[part.part_id]
        part.opacity_keys = scene.layout.opacity_keys_for(part.part_id)
        part.spans = []
        opened = None
        for (start, first), (end, last) in zip(part.opacity_keys, part.opacity_keys[1:]):
            if opened is None and (first > 0 or last > 0):
                opened = start
            if opened is not None and first > 0 and last == 0:
                part.spans.append(VisibleSpan(opened, start, opened == 0.0))
                opened = None
        if opened is not None or part.opacity_keys[-1][1] > 0:
            start = opened if opened is not None else 0.0
            part.spans.append(VisibleSpan(start, 1e12, start == 0.0))
