"""Position curves preserve musical timing without abrupt changes in velocity."""

from __future__ import annotations

import math
import pickle

import pytest

from stavellum.presentation.axis import TimeAxis


@pytest.mark.parametrize("beats,xs", [
    ([0], [10]), ([0, 1], [10]), ([0, 0], [10, 20]), ([0, 1], [20, 10]),
    ([0, math.inf], [10, 20]), ([0, 1], [10, math.nan]),
])
def test_axis_rejects_invalid_anchors(beats, xs):
    with pytest.raises(ValueError):
        TimeAxis(beats, xs)


def test_two_anchors_and_extrapolation_follow_a_straight_line():
    axis = TimeAxis([2, 6], [100, 500])
    for beat in (-5, 0, 2, 3.5, 6, 12):
        assert axis.x_at(beat) == pytest.approx(100 * beat - 100)
        assert axis.beat_at(axis.x_at(beat)) == pytest.approx(beat)


@pytest.mark.parametrize("beats,xs", [
    ([0, 0.25, 0.5, 4, 8], [200, 700, 900, 2100, 2600]),
    ([0, 2, 4, 8, 12], [100, 1700, 2400, 5200, 6700]),
    ([0, 0.01, 100, 100.01], [0, 1000, 1001, 2001]),
])
def test_uneven_anchors_remain_monotone_exact_and_invertible(beats, xs):
    axis = TimeAxis(beats, xs)
    for beat, x in zip(beats, xs):
        assert axis.x_at(beat) == x
        assert axis.beat_at(x) == beat
    for index in range(len(beats) - 1):
        samples = [beats[index] + (beats[index + 1] - beats[index]) * step / 100
                   for step in range(101)]
        positions = [axis.x_at(beat) for beat in samples]
        assert all(right > left for left, right in zip(positions, positions[1:]))
        assert min(positions) >= xs[index]
        assert max(positions) <= xs[index + 1]
        for beat, x in zip(samples, positions):
            assert axis.beat_at(x) == pytest.approx(beat, abs=1e-7)


def test_velocity_is_continuous_where_linear_engraving_clock_would_jump():
    # Natural engraving changes spacing at rhythm and measure boundaries.
    axis = TimeAxis([0, 0.25, 0.5, 1, 2], [100, 600, 900, 1100, 2500])
    linear_speeds = [(right_x - left_x) / (right_b - left_b)
                     for left_b, right_b, left_x, right_x
                     in zip(axis.beats, axis.beats[1:], axis.xs, axis.xs[1:])]
    assert max(abs(after - before)
               for before, after in zip(linear_speeds, linear_speeds[1:])) >= 1000
    epsilon = 1e-7
    for beat, derivative in zip(axis.beats, axis.derivatives):
        left = (axis.x_at(beat) - axis.x_at(beat - epsilon)) / epsilon
        right = (axis.x_at(beat + epsilon) - axis.x_at(beat)) / epsilon
        assert left == pytest.approx(derivative, rel=1e-5)
        assert right == pytest.approx(derivative, rel=1e-5)


def test_long_interval_keeps_moving_instead_of_nearly_stopping():
    axis = TimeAxis([0, 0.01, 100, 100.01], [0, 1000, 1001, 2001])
    middle = 50
    secant = 1 / 99.99
    speed = (axis.x_at(middle + 0.001) - axis.x_at(middle - 0.001)) / 0.002
    assert speed >= secant * 0.49


def test_axis_is_picklable_and_random_access_has_no_history():
    axis = TimeAxis([0, 0.25, 2, 4], [200, 500, 1800, 2300])
    restored = pickle.loads(pickle.dumps(axis))
    expected = {beat: axis.x_at(beat) for beat in (-1, 0, 0.12, 0.25, 1, 4, 6)}
    for beat in (6, 0.12, -1, 4, 0.25, 1, 0, 0.12):
        assert restored.x_at(beat) == expected[beat]
        assert restored.beat_at(expected[beat]) == pytest.approx(beat, abs=1e-7)


def test_local_integrals_and_differences_include_linear_extrapolation():
    axis = TimeAxis([2, 6], [100, 500])
    for start, end in [(-5, 12), (2, 6), (3.5, 3.500001), (8, 10), (6, 2)]:
        assert axis.integral(start, end) == pytest.approx((end - start) * (50 * (start + end) - 100))
        assert axis.difference(start, end) == pytest.approx(100 * (end - start))
    assert axis.max_abs_acceleration == 0


def test_cubic_integral_matches_exact_gaussian_quadrature_and_derivative():
    axis = TimeAxis([0, .25, 1, 4], [100, 600, 900, 3000])
    root = math.sqrt(3 / 5)
    for start, end in zip(axis.beats, axis.beats[1:]):
        middle, radius = (start + end) / 2, (end - start) / 2
        expected = radius * (5 / 9 * axis.x_at(middle - root * radius)
                             + 8 / 9 * axis.x_at(middle)
                             + 5 / 9 * axis.x_at(middle + root * radius))
        assert axis.integral(start, end) == pytest.approx(expected)
        assert axis.difference(start, end) == pytest.approx(axis.x_at(end) - axis.x_at(start))
        epsilon = 1e-7
        numerical = (axis.x_at(middle + epsilon) - axis.x_at(middle - epsilon)) / (2 * epsilon)
        assert axis.speed_at(middle) == pytest.approx(numerical, rel=1e-6)


def test_narrow_integral_late_in_long_score_does_not_subtract_large_primitives():
    axis = TimeAxis([0, 1e6], [1e9, 2e9])
    start, end = 999999.0, 999999.0000001
    assert axis.integral(start, end) / (end - start) == pytest.approx(axis.x_at((start + end) / 2))
    assert axis.difference(start, end) == pytest.approx((end - start) * 1000)
