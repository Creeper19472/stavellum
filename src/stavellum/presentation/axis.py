"""Monotone score position interpolation with continuous playback velocity."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field


@dataclass(slots=True)
class TimeAxis:
    beats: list[float]
    xs: list[float]
    derivatives: list[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.beats) != len(self.xs) or len(self.beats) < 2:
            raise ValueError("时间轴需要至少两个对应的拍点和谱面位置。")
        for values in (self.beats, self.xs):
            if not all(math.isfinite(value) for value in values):
                raise ValueError("时间轴锚点必须为有限数值。")
            if any(right <= left for left, right in zip(values, values[1:])):
                raise ValueError("时间轴锚点必须严格递增。")
        intervals = [right - left for left, right in zip(self.beats, self.beats[1:])]
        slopes = [(right - left) / interval
                  for left, right, interval in zip(self.xs, self.xs[1:], intervals)]
        if not all(math.isfinite(value) and value > 0 for value in intervals + slopes):
            raise ValueError("时间轴锚点间距和速度必须为有限正数。")
        self.derivatives = [slopes[0]]
        for index in range(1, len(self.beats) - 1):
            before, after = intervals[index - 1:index + 1]
            previous, following = slopes[index - 1:index + 1]
            largest_interval = max(before, after)
            before, after = before / largest_interval, after / largest_interval
            first_weight, second_weight = 2 * after + before, after + 2 * before
            smallest_slope = min(previous, following)
            derivative = smallest_slope * ((first_weight + second_weight)
                / (first_weight * (smallest_slope / previous)
                   + second_weight * (smallest_slope / following)))
            # A stricter PCHIP limit prevents near-stops within unusually long beats.
            self.derivatives.append(min(derivative, 2 * min(previous, following)))
        self.derivatives.append(slopes[-1])

    def _segment_x(self, index: int, fraction: float) -> float:
        interval = self.beats[index + 1] - self.beats[index]
        start, end = self.xs[index:index + 2]
        first, last = self.derivatives[index:index + 2]
        square, cube = fraction * fraction, fraction * fraction * fraction
        return (start + (-2 * cube + 3 * square) * (end - start)
                + (cube - 2 * square + fraction) * interval * first
                + (cube - square) * interval * last)

    def _coefficients(self, index: int) -> tuple[float, float, float, float]:
        """Cubic coefficients in the segment's normalized 0..1 coordinate."""
        interval = self.beats[index + 1] - self.beats[index]
        start, end = self.xs[index:index + 2]
        first, last = (value * interval for value in self.derivatives[index:index + 2])
        distance = end - start
        return start, first, 3 * distance - 2 * first - last, -2 * distance + first + last

    def polynomial_at(self, beat: float) -> tuple[float, float, float, float]:
        """Local Taylor polynomial in beat units, including linear extrapolation."""
        if beat <= self.beats[0]:
            return self.x_at(beat), self.derivatives[0], 0.0, 0.0
        if beat >= self.beats[-1]:
            return self.x_at(beat), self.derivatives[-1], 0.0, 0.0
        index = bisect_right(self.beats, beat) - 1
        interval = self.beats[index + 1] - self.beats[index]
        fraction = (beat - self.beats[index]) / interval
        _, first, second, third = self._coefficients(index)
        return (self.x_at(beat), (first + fraction * (2 * second + 3 * third * fraction)) / interval,
                (second + 3 * third * fraction) / interval**2, third / interval**3)

    def speed_at(self, beat: float) -> float:
        return self.polynomial_at(beat)[1]

    @property
    def max_abs_acceleration(self) -> float:
        """Exact bound in source coordinates per beat squared."""
        maximum = 0.0
        for index in range(len(self.beats) - 1):
            interval = self.beats[index + 1] - self.beats[index]
            _, _, second, third = self._coefficients(index)
            maximum = max(maximum, abs(2 * second / interval**2),
                          abs((2 * second + 6 * third) / interval**2))
        return maximum

    def integral(self, start: float, end: float) -> float:
        """Integrate locally rather than subtracting large cumulative primitives."""
        return self._integrate(start, end, difference=False)

    def difference(self, start: float, end: float) -> float:
        """Stable x(end)-x(start), even for narrow windows late in a long score."""
        return self._integrate(start, end, difference=True)

    def _integrate(self, start: float, end: float, *, difference: bool) -> float:
        if end < start:
            return -self._integrate(end, start, difference=difference)
        pieces = []
        while start < end:
            if start < self.beats[0]:
                right = min(end, self.beats[0])
                value = self.derivatives[0] if difference else self.x_at((start + right) / 2)
            elif start >= self.beats[-1]:
                right = end
                value = self.derivatives[-1] if difference else self.x_at((start + right) / 2)
            else:
                index = bisect_right(self.beats, start) - 1
                interval = self.beats[index + 1] - self.beats[index]
                right = min(end, self.beats[index + 1])
                midpoint = ((start - self.beats[index]) + (right - self.beats[index])) / (2 * interval)
                radius = (right - start) / (2 * interval)
                first, second, third, fourth = self._coefficients(index)
                if difference:
                    value = (second + 2 * third * midpoint
                             + 3 * fourth * (midpoint**2 + radius**2 / 3)) / interval
                else:
                    value = (first + second * midpoint + third * (midpoint**2 + radius**2 / 3)
                             + fourth * (midpoint**3 + midpoint * radius**2))
            pieces.append((right - start) * value)
            start = right
        return math.fsum(pieces)

    def x_at(self, beat: float) -> float:
        if beat <= self.beats[0]:
            return self.xs[0] + (beat - self.beats[0]) * self.derivatives[0]
        if beat >= self.beats[-1]:
            return self.xs[-1] + (beat - self.beats[-1]) * self.derivatives[-1]
        index = bisect_right(self.beats, beat) - 1
        fraction = (beat - self.beats[index]) / (self.beats[index + 1] - self.beats[index])
        return self._segment_x(index, fraction)

    def beat_at(self, x: float) -> float:
        if x <= self.xs[0]:
            return self.beats[0] + (x - self.xs[0]) / self.derivatives[0]
        if x >= self.xs[-1]:
            return self.beats[-1] + (x - self.xs[-1]) / self.derivatives[-1]
        index = bisect_right(self.xs, x) - 1
        if x == self.xs[index]:
            return self.beats[index]
        low, high = 0.0, 1.0
        for _ in range(48):
            middle = (low + high) / 2
            if self._segment_x(index, middle) < x:
                low = middle
            else:
                high = middle
        return self.beats[index] + (low + high) / 2 * (self.beats[index + 1] - self.beats[index])
