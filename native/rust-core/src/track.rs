use crate::{abi::CurveKey, require};

/// Quintic Hermite coefficients in the segment's normalized 0..1 coordinate.
fn hermite(first: CurveKey, last: CurveKey) -> [f64; 6] {
    let duration = last.time - first.time;
    let first_velocity = first.velocity * duration;
    let first_acceleration = first.acceleration * duration * duration / 2.0;
    let displacement = last.value - first.value - first_velocity - first_acceleration;
    let velocity = last.velocity * duration - first_velocity - 2.0 * first_acceleration;
    let acceleration = last.acceleration * duration * duration - 2.0 * first_acceleration;
    [
        first.value,
        first_velocity,
        first_acceleration,
        10.0 * displacement - 4.0 * velocity + acceleration / 2.0,
        -15.0 * displacement + 7.0 * velocity - acceleration,
        6.0 * displacement - 3.0 * velocity + acceleration / 2.0,
    ]
}

fn polynomial(coefficients: &[f64; 6], value: f64) -> f64 {
    // Horner over indices 0..count, matching Python's tuple evaluation.
    let mut result = 0.0;
    for coefficient in coefficients.iter().rev() {
        result = result * value + coefficient;
    }
    result
}

fn derivative(coefficients: &[f64; 6]) -> [f64; 6] {
    let mut result = [0.0; 6];
    for index in 1..coefficients.len() {
        result[index - 1] = index as f64 * coefficients[index];
    }
    result
}

/// Compiled curve track: quintic segments keyed by time.
#[derive(Clone)]
pub struct Track {
    keys: Vec<CurveKey>,
}

impl Track {
    pub(crate) fn new(keys: Vec<CurveKey>) -> Result<Self, String> {
        require(!keys.is_empty(), "Layout track requires at least one key")?;
        for pair in keys.windows(2) {
            require(
                pair[1].time > pair[0].time,
                "Layout track keys must be sorted by time",
            )?;
        }
        if !keys.iter().all(|key| {
            key.time.is_finite()
                && key.value.is_finite()
                && key.velocity.is_finite()
                && key.acceleration.is_finite()
        }) {
            return Err("Layout track keys must be finite".to_owned());
        }
        Ok(Self { keys })
    }

    pub(crate) fn sample(&self, time: f64) -> CurveKey {
        let index = match self.keys.binary_search_by(|key| {
            key.time
                .partial_cmp(&time)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(index) => index,
            Err(index) => index.saturating_sub(1),
        };
        let first = self.keys[index];
        let Some(last) = self.keys.get(index + 1) else {
            return CurveKey {
                time,
                value: first.value,
                velocity: 0.0,
                acceleration: 0.0,
            };
        };
        if time < first.time || last.time <= first.time {
            return CurveKey {
                time,
                value: first.value,
                velocity: 0.0,
                acceleration: 0.0,
            };
        }
        let duration = last.time - first.time;
        let fraction = ((time - first.time) / duration).clamp(0.0, 1.0);
        let coefficients = hermite(first, *last);
        let first_derivative = derivative(&coefficients);
        let second_derivative = derivative(&first_derivative);
        CurveKey {
            time,
            value: polynomial(&coefficients, fraction),
            velocity: polynomial(&first_derivative, fraction) / duration,
            acceleration: polynomial(&second_derivative, fraction) / (duration * duration),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(time: f64, value: f64) -> CurveKey {
        CurveKey {
            time,
            value,
            velocity: 0.0,
            acceleration: 0.0,
        }
    }

    #[test]
    fn track_interpolates_with_zero_endpoint_derivatives() {
        let track = Track::new(vec![key(0.0, 0.0), key(1.0, 1.0)]).unwrap();
        assert!((track.sample(0.0).value - 0.0).abs() < 1e-12);
        assert!((track.sample(1.0).value - 1.0).abs() < 1e-12);
        let middle = track.sample(0.5).value;
        assert!((middle - 0.5).abs() < 1e-3, "smoothstep midpoint {middle}");
        assert!(track.sample(2.0).value > 0.999);
    }
}
