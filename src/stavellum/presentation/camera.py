"""Absolute-time lookahead camera, separate from the exact musical time axis."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .axis import TimeAxis


@dataclass(slots=True)
class CameraTimeline:
    axis: TimeAxis
    beats_per_second: float
    score_offset: float
    half_window_seconds: float
    certified_drift_source: float = 0.0

    def _beat(self, audio_time: float) -> float:
        return (audio_time - self.score_offset) * self.beats_per_second

    @property
    def radius(self) -> float:
        return self.half_window_seconds * self.beats_per_second

    def x_at(self, audio_time: float) -> float:
        beat = self._beat(audio_time)
        if self.radius == 0:
            return self.axis.x_at(beat)
        left, right = beat - self.radius, beat + self.radius
        # Normalize the represented interval; 2*radius can differ by ulps and
        # would amplify a large absolute score coordinate in very narrow windows.
        if right == left:
            return self.axis.x_at(beat)
        return self.axis.integral(left, right) / (right - left)

    def speed_at(self, audio_time: float) -> float:
        beat = self._beat(audio_time)
        if self.radius == 0:
            return self.axis.speed_at(beat) * self.beats_per_second
        left, right = beat - self.radius, beat + self.radius
        if right == left:
            return self.axis.speed_at(beat) * self.beats_per_second
        return self.axis.difference(left, right) / (right - left) * self.beats_per_second

    def acceleration_at(self, audio_time: float) -> float:
        beat = self._beat(audio_time)
        if self.radius == 0:
            return 2 * self.axis.polynomial_at(beat)[2] * self.beats_per_second**2
        left, right = beat - self.radius, beat + self.radius
        if right == left:
            return 2 * self.axis.polynomial_at(beat)[2] * self.beats_per_second**2
        return (self.axis.speed_at(right) - self.axis.speed_at(left)) / (right - left) * self.beats_per_second**2

    def min_speed(self, start: float, end: float) -> float:
        """Exact interval minimum: the box camera's speed is piecewise cubic."""
        if end < start:
            start, end = end, start
        lower, upper = self._beat(start), self._beat(end)
        splits = {lower, upper}
        for beat in self.axis.beats:
            for boundary in (beat - self.radius, beat + self.radius):
                if lower < boundary < upper:
                    splits.add(boundary)
        points = sorted(splits)
        minimum = min(self.speed_at(start), self.speed_at(end))
        for left, right in zip(points, points[1:]):
            middle = (left + right) / 2
            if self.radius:
                before, after = self.axis.polynomial_at(middle - self.radius), self.axis.polynomial_at(middle + self.radius)
                coefficients = [(after[index] - before[index]) / (2 * self.radius) for index in range(1, 4)]
                linear, square, cube = coefficients
            else:
                _, _, square, cube = self.axis.polynomial_at(middle)
                linear, square, cube = 2 * square, 3 * cube, 0.0
            candidates = [left, right]
            # Critical points of speed: linear + 2*square*u + 3*cube*u^2.
            quadratic, first = 3 * cube, 2 * square
            if quadratic == 0:
                roots = [-linear / first] if first else []
            else:
                discriminant = first * first - 4 * quadratic * linear
                roots = []
                if discriminant >= 0:
                    root = math.sqrt(discriminant)
                    stable = -0.5 * (first + math.copysign(root, first))
                    roots = [stable / quadratic]
                    if stable:
                        roots.append(linear / stable)
            candidates.extend(middle + root for root in roots if left < middle + root < right)
            minimum = min(minimum, *(self.speed_at(beat / self.beats_per_second + self.score_offset)
                                     for beat in candidates))
        return max(0.0, minimum)


def compile_camera(axis: TimeAxis, bpm: float, score_offset: float, drift_source: float) -> CameraTimeline:
    """Choose the widest globally certified positive box kernel without an optimizer.

    At 240 Hz, the between-sample interpolation error is at most A*dt^2/4:
    the exact axis and its box average each have acceleration bounded by A.
    A zero-width camera is only used when the viewport provides no drift room.
    """
    rate = bpm / 60
    if drift_source <= 0:
        return CameraTimeline(axis, rate, score_offset, 0.0)
    acceleration = axis.max_abs_acceleration * rate**2
    step = 1 / 240
    certificate = acceleration * step**2 / 4
    # Pathological tiny intervals need a finer certificate, not an invalid bound.
    if certificate > drift_source / 4:
        step = min(step, math.sqrt(drift_source / acceleration))
        certificate = acceleration * step**2 / 4
    for index in range(32):
        for base in (0.5, 0.35):
            window = math.ldexp(base, -index)
            camera = CameraTimeline(axis, rate, score_offset, window)
            start = axis.beats[0] / rate + score_offset - window
            end = axis.beats[-1] / rate + score_offset + window
            samples = math.ceil((end - start) / step)
            maximum = 0.0
            for sample in range(samples + 1):
                time = min(end, start + sample * step)
                error = abs(camera.x_at(time) - axis.x_at(camera._beat(time)))
                maximum = max(maximum, error)
                if maximum + certificate > drift_source:
                    break
            else:
                camera.certified_drift_source = maximum + certificate
                return camera
    raise ValueError("谱面时间轴的数值范围过大，无法可靠计算平滑相机。")
