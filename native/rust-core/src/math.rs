/// Sum finite integration pieces using a non-overlapping floating expansion.
/// Keep discarded low bits until the final round-to-even, matching math.fsum.
pub(crate) fn accurate_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut expansion: Vec<f64> = Vec::new();
    for mut value in values {
        let mut used = 0;
        for index in 0..expansion.len() {
            let mut term = expansion[index];
            if value.abs() < term.abs() {
                std::mem::swap(&mut value, &mut term);
            }
            let rounded = value + term;
            let error = term - (rounded - value);
            if error != 0.0 {
                expansion[used] = error;
                used += 1;
            }
            value = rounded;
        }
        expansion.truncate(used);
        if value != 0.0 {
            expansion.push(value);
        }
    }
    let mut result = expansion.pop().unwrap_or(0.0);
    while let Some(term) = expansion.pop() {
        let rounded = result + term;
        let error = term - (rounded - result);
        result = rounded;
        if error != 0.0 {
            // A remaining term with the same sign makes an apparent midpoint
            // lie strictly beyond the halfway point in that direction.
            if expansion
                .last()
                .is_some_and(|next| (*next > 0.0) == (error > 0.0))
            {
                let correction = 2.0 * error;
                let adjusted = result + correction;
                if adjusted - result == correction {
                    result = adjusted;
                }
            }
            break;
        }
    }
    result
}

pub(crate) fn ease(value: f64) -> f64 {
    // Clamped quintic with zero endpoint velocity and acceleration.
    let value = value.clamp(0.0, 1.0);
    value * value * value * (10.0 + value * (-15.0 + 6.0 * value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integration_sum_retains_low_bits_and_rounds_half_even() {
        assert_eq!(accurate_sum([1e16, 1.0, -1e16]), 1.0);
        let half_ulp = 2.0f64.powi(-53);
        assert_eq!(accurate_sum([1.0, half_ulp]), 1.0);
        assert_eq!(accurate_sum([1.0, half_ulp, 1e-30]), 1.0 + 2.0 * half_ulp);
        assert_eq!(accurate_sum([1e-30, half_ulp, 1.0]), 1.0 + 2.0 * half_ulp);
        assert_eq!(
            accurate_sum([-1.0, -half_ulp, -1e-30]),
            -1.0 - 2.0 * half_ulp
        );
    }
}
