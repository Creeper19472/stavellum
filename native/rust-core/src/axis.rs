use crate::{math::accurate_sum, require};

/// Monotone PCHIP-style score axis (exact port of `axis.py::TimeAxis`).
pub struct TimeAxis {
    beats: Vec<f64>,
    xs: Vec<f64>,
    derivatives: Vec<f64>,
}

impl TimeAxis {
    pub fn new(beats: Vec<f64>, xs: Vec<f64>) -> Result<Self, String> {
        require(
            beats.len() == xs.len() && beats.len() >= 2,
            "时间轴需要至少两个对应的拍点和谱面位置。",
        )?;
        for values in [&beats, &xs] {
            if !values.iter().all(|value| value.is_finite()) {
                return Err("时间轴锚点必须为有限数值。".to_owned());
            }
            for pair in values.windows(2) {
                if pair[1] <= pair[0] {
                    return Err("时间轴锚点必须严格递增。".to_owned());
                }
            }
        }
        let mut intervals = Vec::with_capacity(beats.len() - 1);
        let mut slopes = Vec::with_capacity(beats.len() - 1);
        for index in 0..beats.len() - 1 {
            let interval = beats[index + 1] - beats[index];
            let slope = (xs[index + 1] - xs[index]) / interval;
            if !interval.is_finite() || !slope.is_finite() || interval <= 0.0 || slope <= 0.0 {
                return Err("时间轴锚点间距和速度必须为有限正数。".to_owned());
            }
            intervals.push(interval);
            slopes.push(slope);
        }
        let mut derivatives = vec![slopes[0]];
        for index in 1..beats.len() - 1 {
            let (before, after) = (intervals[index - 1], intervals[index]);
            let (previous, following) = (slopes[index - 1], slopes[index]);
            let largest_interval = before.max(after);
            let before = before / largest_interval;
            let after = after / largest_interval;
            let first_weight = 2.0 * after + before;
            let second_weight = after + 2.0 * before;
            let smallest_slope = previous.min(following);
            let value = smallest_slope
                * ((first_weight + second_weight)
                    / (first_weight * (smallest_slope / previous)
                        + second_weight * (smallest_slope / following)));
            // A stricter PCHIP limit prevents near-stops within long beats.
            derivatives.push(value.min(2.0 * previous.min(following)));
        }
        derivatives.push(*slopes.last().expect("non-empty"));
        Ok(Self {
            beats,
            xs,
            derivatives,
        })
    }

    fn segment(&self, index: usize) -> (f64, f64, f64, f64, f64) {
        let interval = self.beats[index + 1] - self.beats[index];
        (
            interval,
            self.xs[index],
            self.xs[index + 1],
            self.derivatives[index],
            self.derivatives[index + 1],
        )
    }

    fn segment_x(&self, index: usize, fraction: f64) -> f64 {
        let (interval, start, end, first, last) = self.segment(index);
        let square = fraction * fraction;
        let cube = square * fraction;
        start
            + (-2.0 * cube + 3.0 * square) * (end - start)
            + (cube - 2.0 * square + fraction) * interval * first
            + (cube - square) * interval * last
    }

    /// Cubic coefficients in the segment's normalized coordinate.
    fn coefficients(&self, index: usize) -> (f64, f64, f64, f64) {
        let (interval, start, end, first, last) = self.segment(index);
        let first = first * interval;
        let last = last * interval;
        let distance = end - start;
        (
            start,
            first,
            3.0 * distance - 2.0 * first - last,
            -2.0 * distance + first + last,
        )
    }

    fn search(&self, beat: f64) -> usize {
        match self.beats.binary_search_by(|value| {
            value
                .partial_cmp(&beat)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(index) => index.min(self.beats.len() - 2),
            Err(index) => index.saturating_sub(1).min(self.beats.len() - 2),
        }
    }

    pub fn x_at(&self, beat: f64) -> f64 {
        if beat <= self.beats[0] {
            return self.xs[0] + (beat - self.beats[0]) * self.derivatives[0];
        }
        if beat >= *self.beats.last().expect("non-empty") {
            let last = self.beats.len() - 1;
            return self.xs[last] + (beat - self.beats[last]) * self.derivatives[last];
        }
        let index = self.search(beat);
        let fraction = (beat - self.beats[index]) / (self.beats[index + 1] - self.beats[index]);
        self.segment_x(index, fraction)
    }

    /// Local Taylor polynomial in beat units, including linear extrapolation.
    pub fn polynomial_at(&self, beat: f64) -> (f64, f64, f64, f64) {
        if beat <= self.beats[0] {
            return (self.x_at(beat), self.derivatives[0], 0.0, 0.0);
        }
        if beat >= *self.beats.last().expect("non-empty") {
            let last = self.beats.len() - 1;
            return (self.x_at(beat), self.derivatives[last], 0.0, 0.0);
        }
        let index = self.search(beat);
        let interval = self.beats[index + 1] - self.beats[index];
        let fraction = (beat - self.beats[index]) / interval;
        let (_, first, second, third) = self.coefficients(index);
        (
            self.x_at(beat),
            (first + fraction * (2.0 * second + 3.0 * third * fraction)) / interval,
            (second + 3.0 * third * fraction) / (interval * interval),
            third / (interval * interval * interval),
        )
    }

    pub fn speed_at(&self, beat: f64) -> f64 {
        self.polynomial_at(beat).1
    }

    pub fn max_abs_acceleration(&self) -> f64 {
        let mut maximum = 0.0f64;
        for index in 0..self.beats.len() - 1 {
            let interval = self.beats[index + 1] - self.beats[index];
            let interval_squared = interval * interval;
            let (_, _, second, third) = self.coefficients(index);
            maximum = maximum
                .max((2.0 * second / interval_squared).abs())
                .max(((2.0 * second + 6.0 * third) / interval_squared).abs());
        }
        maximum
    }

    /// Integrate locally rather than subtracting large cumulative primitives.
    fn integrate(&self, mut start: f64, end: f64, difference: bool) -> f64 {
        if end < start {
            return -self.integrate(end, start, difference);
        }
        let mut pieces: Vec<f64> = Vec::new();
        while start < end {
            let (right, value) = if start < self.beats[0] {
                let right = end.min(self.beats[0]);
                let middle = (start + right) / 2.0;
                let value = if difference {
                    self.derivatives[0]
                } else {
                    self.x_at(middle)
                };
                (right, value)
            } else if start >= *self.beats.last().expect("non-empty") {
                let middle = (start + end) / 2.0;
                let value = if difference {
                    *self.derivatives.last().expect("non-empty")
                } else {
                    self.x_at(middle)
                };
                (end, value)
            } else {
                let index = self.search(start);
                let interval = self.beats[index + 1] - self.beats[index];
                let right = end.min(self.beats[index + 1]);
                let midpoint =
                    ((start - self.beats[index]) + (right - self.beats[index])) / (2.0 * interval);
                let radius = (right - start) / (2.0 * interval);
                let (first, second, third, fourth) = self.coefficients(index);
                let value = if difference {
                    (second
                        + 2.0 * third * midpoint
                        + 3.0 * fourth * (midpoint * midpoint + radius * radius / 3.0))
                        / interval
                } else {
                    first
                        + second * midpoint
                        + third * (midpoint * midpoint + radius * radius / 3.0)
                        + fourth * (midpoint * midpoint * midpoint + midpoint * radius * radius)
                };
                (right, value)
            };
            pieces.push((right - start) * value);
            start = right;
        }
        accurate_sum(pieces)
    }

    pub fn integral(&self, start: f64, end: f64) -> f64 {
        self.integrate(start, end, false)
    }

    pub fn difference(&self, start: f64, end: f64) -> f64 {
        self.integrate(start, end, true)
    }

    pub fn beat_at(&self, x: f64) -> f64 {
        if x <= self.xs[0] {
            return self.beats[0] + (x - self.xs[0]) / self.derivatives[0];
        }
        let last = self.beats.len() - 1;
        if x >= self.xs[last] {
            return self.beats[last] + (x - self.xs[last]) / self.derivatives[last];
        }
        let index = match self
            .xs
            .binary_search_by(|value| value.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(index) => return self.beats[index],
            Err(index) => index.saturating_sub(1),
        };
        let (mut low, mut high) = (0.0f64, 1.0f64);
        for _ in 0..48 {
            let middle = (low + high) / 2.0;
            if self.segment_x(index, middle) < x {
                low = middle;
            } else {
                high = middle;
            }
        }
        self.beats[index] + (low + high) / 2.0 * (self.beats[index + 1] - self.beats[index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_matches_linear_and_cubic_expectations() {
        let axis = TimeAxis::new(vec![0.0, 1.0, 2.0], vec![0.0, 100.0, 260.0]).unwrap();
        assert!((axis.x_at(0.0) - 0.0).abs() < 1e-12);
        assert!((axis.x_at(1.0) - 100.0).abs() < 1e-12);
        assert!((axis.x_at(2.0) - 260.0).abs() < 1e-12);
        assert!((axis.x_at(-1.0) + axis.derivatives[0]).abs() < 1e-12);
        assert!(axis.x_at(3.0) > 260.0);
        let beat = axis.beat_at(axis.x_at(0.5));
        assert!((beat - 0.5).abs() < 1e-9, "{beat}");
    }
}
