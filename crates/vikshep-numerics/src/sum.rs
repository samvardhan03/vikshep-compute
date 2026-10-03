//! Fixed-order summation for host (Tier-2) reductions (VDS-1 section 2.5).
//!
//! The order of a floating-point sum changes its value, so every Tier-2
//! reduction uses one of the fixed orders defined here, independent of
//! thread count or hardware.

/// Pairwise sum in binary64 (VDS-1 section 14.9):
///
/// ```text
/// psum([])          = +0
/// psum([a])         = a
/// psum(a[0..n])     = psum(a[0..h]) + psum(a[h..n]),  h = floor(n / 2)
/// ```
#[must_use]
pub fn pairwise_sum(values: &[f64]) -> f64 {
    match values.len() {
        0 => 0.0,
        1 => values[0],
        n => {
            let h = n / 2;
            pairwise_sum(&values[..h]) + pairwise_sum(&values[h..])
        }
    }
}

/// Sequential sum in binary64, left to right, starting from +0.
#[must_use]
pub fn sequential_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().fold(0.0, |acc, v| acc + v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_order_is_as_specified() {
        assert_eq!(pairwise_sum(&[]).to_bits(), 0.0f64.to_bits());
        assert_eq!(pairwise_sum(&[3.0]), 3.0);
        // psum([a,b,c,d]) = (a + b) + (c + d).
        let v = [1e16, 1.0, -1e16, 1.0];
        assert_eq!(pairwise_sum(&v), (1e16 + 1.0) + (-1e16 + 1.0));
        // psum([a,b,c]) = a + (b + c).
        let w = [1.0, 1e16, -1e16];
        assert_eq!(pairwise_sum(&w), 1.0 + (1e16 + -1e16));
        assert_eq!(sequential_sum(w), (1.0 + 1e16) + -1e16);
    }
}
