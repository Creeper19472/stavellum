"""Native frame queries with explicit per-test ownership, independent of drawing."""

from dataclasses import replace
from types import SimpleNamespace

from stavellum._frame import FrameEvaluator
from stavellum.axis import TimeAxis
from stavellum.camera import CameraTimeline
from stavellum.layout import _CurveKey, _Track
from stavellum.models import RenderSettings

_evaluators = {}
_activity_scenes = {}


def frame_state(scene, time):
    key = id(scene)
    if key in _evaluators and _evaluators[key][1] is not scene.layout:
        _evaluators.pop(key)[2].close()
    if key not in _evaluators:
        _evaluators[key] = (scene, scene.layout, FrameEvaluator(scene))
    return _evaluators[key][2].evaluate(time)


def frame_layout(scene, time):
    return frame_state(scene, time).layout


def activity_levels(part, time):
    # Evaluate isolated envelopes through the same production ABI, without engraving.
    key = id(part)
    if key not in _activity_scenes:
        axis = TimeAxis([0.0, 1.0], [0.0, 1.0])
        track = _Track([_CurveKey(0.0, 0.0)])
        layout = SimpleNamespace(
            tops={part.part_id: track}, opacities={part.part_id: track}, zoom=track,
            indicator_right=1.0, indicator_source_width=1.0, indicator_source_height=1.0,
            icon_source_size=1.0, icon_source_gap=0.0, region_top=0.0, region_bottom=100.0,
            expanded_top=0.0, expanded_bottom=100.0, expansion_start=None,
            expansion_duration=1.0, tempo_padding=0.0, tempo_exit_time=float('inf'),
            tempo_owner_id='',
        )
        _activity_scenes[key] = SimpleNamespace(
            parts=[part], axis=axis, camera=CameraTimeline(axis, 1.0, 0.0, 0.0),
            layout=layout, settings=RenderSettings(intro_delay_seconds=0),
            scale=1.0, play_x=1.0, body_left=0.0, body_right=2.0,
        )
    return frame_state(_activity_scenes[key], time).activity[part.part_id]


def activity_level(part, time):
    return activity_levels(part, time)[0]


def raster_state(scene, scale):
    import math

    layout = replace(scene.layout, zoom=_Track([_CurveKey(0, math.log(scale))]))
    with FrameEvaluator(replace(scene, layout=layout)) as evaluator:
        return evaluator.evaluate(0)


def close_evaluators():
    try:
        for _, _, evaluator in _evaluators.values():
            evaluator.close()
    finally:
        _evaluators.clear()
        _activity_scenes.clear()
