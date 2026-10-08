"""Seekable polynomial curves shared by compiled presentation timelines."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field


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
