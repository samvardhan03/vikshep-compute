//! Total-order sorting for Tier-2 code (VDS-1 section 15.5).
//!
//! Every sort of floating-point keys uses `f64::total_cmp` (IEEE-754
//! totalOrder: -NaN < -inf < ... < -0 < +0 < ... < +inf < +NaN) and breaks
//! ties by the element index, so the result never depends on the sorting
//! algorithm or platform.

use core::cmp::Ordering;

/// Compare `(key, index)` pairs in total order.
#[must_use]
pub fn cmp_key_index(a: (f64, usize), b: (f64, usize)) -> Ordering {
    a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
}

/// Indices of `values` in ascending total order, ties by index.
#[must_use]
pub fn argsort(values: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&i, &j| cmp_key_index((values[i], i), (values[j], j)));
    idx
}

/// Indices of `values` in descending total order, ties by ascending index.
#[must_use]
pub fn argsort_desc(values: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&i, &j| values[j].total_cmp(&values[i]).then(i.cmp(&j)));
    idx
}

/// `values` sorted ascending in total order.
#[must_use]
pub fn sorted(values: &[f64]) -> Vec<f64> {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_break_by_index() {
        let v = [2.0, 1.0, 2.0, -0.0, 0.0];
        assert_eq!(argsort(&v), vec![3, 4, 1, 0, 2]);
        assert_eq!(argsort_desc(&v), vec![0, 2, 1, 4, 3]);
        assert_eq!(sorted(&v)[0].to_bits(), (-0.0f64).to_bits());
    }
}
