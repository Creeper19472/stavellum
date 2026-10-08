"""A lookahead camera moves smoothly while the exact musical axis stays intact."""

from __future__ import annotations

import math
import pickle

import pytest

from stavellum.presentation.axis import TimeAxis
from stavellum.presentation.camera import CameraTimeline, compile_camera


def test_linear_motion_is_unchanged_including_silent_leadin_and_tail():
    axis = TimeAxis([0, 4], [100, 2100])
    camera = compile_camera(axis, 120, 1.25, 50)
    assert camera.half_window_seconds == .5
    for time in (-10, 0, 1.25, 2, 10):
        assert camera.x_at(time) == pytest.approx(axis.x_at((time - 1.25) * 2))
        assert camera.speed_at(time) == pytest.approx(1000)
        assert camera.acceleration_at(time) == pytest.approx(0)
    assert camera.min_speed(-10, 10) == pytest.approx(1000)


def test_camera_is_c2_monotone_and_certifies_between_samples():
    axis = TimeAxis([0, .25, .5, 1, 2, 4], [100, 600, 900, 1100, 2500, 3900])
    camera = compile_camera(axis, 120, .75, 100)
    assert camera.certified_drift_source <= 100
    values = []
    for index in range(-120, 1081):
        time = index / 360
        values.append(camera.x_at(time))
        assert abs(camera.x_at(time) - axis.x_at((time - .75) * 2)) <= camera.certified_drift_source + 1e-7
        assert camera.speed_at(time) > 0
        assert abs(camera.acceleration_at(time)) <= axis.max_abs_acceleration * 4 + 1e-6
    assert all(right > left for left, right in zip(values, values[1:]))
    for beat in axis.beats:
        for shift in (-camera.half_window_seconds, camera.half_window_seconds, 0):
            time = beat / 2 + .75 + shift
            epsilon = 1e-7
            assert camera.acceleration_at(time - epsilon) == pytest.approx(camera.acceleration_at(time + epsilon), abs=.2)
            observed = (camera.x_at(time + epsilon) - camera.x_at(time - epsilon)) / (2 * epsilon)
            assert observed == pytest.approx(camera.speed_at(time), rel=1e-5)


def test_tighter_corridor_selects_a_shorter_uniform_window():
    axis = TimeAxis([0, .25, 1, 4], [100, 600, 900, 3000])
    wide = compile_camera(axis, 120, 0, 200)
    narrow = compile_camera(axis, 120, 0, 10)
    assert narrow.half_window_seconds < wide.half_window_seconds
    assert narrow.certified_drift_source <= 10
    assert axis.x_at(.25) == 600


def test_minimum_speed_finds_interior_extrema_and_supports_reversed_ranges():
    axis = TimeAxis([0, .25, 1, 4], [100, 600, 900, 3000])
    camera = compile_camera(axis, 90, -.5, 70)
    low = camera.min_speed(-.5, 2)
    samples = [camera.speed_at(-.5 + 2.5 * index / 10000) for index in range(10001)]
    assert low <= min(samples) + 1e-8
    assert low == pytest.approx(min(samples), rel=1e-5)
    assert camera.min_speed(2, -.5) == low


def test_camera_pickle_and_random_queries_have_no_frame_history():
    camera = compile_camera(TimeAxis([0, .5, 2, 4], [100, 700, 1400, 4000]), 118, 1.25, 50)
    restored = pickle.loads(pickle.dumps(camera))
    expected = {time: camera.x_at(time) for time in (-2, 0, .75, 1.25, 2, 8)}
    for time in (8, .75, -2, 1.25, 0, 2, .75):
        assert restored.x_at(time) == expected[time]
        assert math.isfinite(restored.speed_at(time))


def test_no_drift_room_uses_exact_position_without_dividing_by_zero():
    axis = TimeAxis([0, .5, 2, 4], [100, 700, 1400, 4000])
    camera = compile_camera(axis, 120, 0, 0)
    assert camera.half_window_seconds == 0
    assert camera.x_at(.25) == axis.x_at(.5)
    assert camera.min_speed(0, 2) > 0


def test_narrow_box_does_not_amplify_a_large_absolute_position():
    axis = TimeAxis([0, 1, 2, 10], [1e9, 1e9 + 1000, 1e9 + 3000, 1e9 + 7000])
    camera = CameraTimeline(axis, 2, 0, 1e-6)
    time = 4.56789
    assert camera.x_at(time) == pytest.approx(axis.x_at(time * 2), abs=5e-7, rel=0)
