//! Jensen-Shannon divergence of binned distributions (`spec/VDS-1.md`
//! section 16.3).

use vikshep_numerics::sum::pairwise_sum;

use crate::StatsError;

/// Equal-width binning of `[lo, hi]` into `n_bins` bins.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Binning {
    /// Lower edge.
    pub lo: f64,
    /// Upper edge (inclusive: values equal to `hi` fall in the last bin).
    pub hi: f64,
    /// Number of bins (>= 1).
    pub n_bins: usize,
}

impl Binning {
    /// Edges from the minimum and maximum of `reference` (VDS-1 section
    /// 16.3: fixed-range binning over the pre-cut sample).
    pub fn from_range(reference: &[f64], n_bins: usize) -> Result<Self, StatsError> {
        if n_bins == 0 {
            return Err(StatsError::new("n_bins must be at least 1"));
        }
        if reference.is_empty() || reference.iter().any(|v| !v.is_finite()) {
            return Err(StatsError::new(
                "binning needs a non-empty finite reference sample",
            ));
        }
        let lo = reference.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = reference.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Ok(Self { lo, hi, n_bins })
    }

    /// Bin of `v`: `floor(((v - lo) / (hi - lo)) * n_bins)` clamped to
    /// `0..n_bins` (bin 0 when `hi == lo`).
    #[must_use]
    pub fn bin(&self, v: f64) -> usize {
        if self.hi <= self.lo {
            return 0;
        }
        let t = ((v - self.lo) / (self.hi - self.lo)) * self.n_bins as f64;
        if t <= 0.0 {
            0
        } else {
            (t.floor() as usize).min(self.n_bins - 1)
        }
    }

    /// Weighted histogram: bin `b` holds the pairwise sum, in event order,
    /// of the weights of the events in that bin.
    pub fn histogram(&self, values: &[f64], weights: &[f64]) -> Result<Vec<f64>, StatsError> {
        if values.len() != weights.len() {
            return Err(StatsError::new("values and weights differ in length"));
        }
        let mut per_bin: Vec<Vec<f64>> = vec![Vec::new(); self.n_bins];
        for (&v, &w) in values.iter().zip(weights) {
            per_bin[self.bin(v)].push(w);
        }
        Ok(per_bin.iter().map(|b| pairwise_sum(b)).collect())
    }
}

/// `sum_i p_i ln(p_i / q_i)` with `0 ln 0 = 0` (pairwise sum over bins).
fn kl(p: &[f64], q: &[f64]) -> f64 {
    let terms: Vec<f64> = p
        .iter()
        .zip(q)
        .map(|(&pi, &qi)| {
            if pi == 0.0 {
                0.0
            } else {
                pi * vikshep_detmath::ln(pi / qi)
            }
        })
        .collect();
    pairwise_sum(&terms)
}

/// Jensen-Shannon divergence (natural log, in `[0, ln 2]`) of two
/// histograms, each normalized by its own pairwise total:
/// `JSD = 0.5 KL(P||M) + 0.5 KL(Q||M)`, `M = 0.5 (P + Q)`.
pub fn jsd(p_counts: &[f64], q_counts: &[f64]) -> Result<f64, StatsError> {
    if p_counts.len() != q_counts.len() || p_counts.is_empty() {
        return Err(StatsError::new(
            "histograms must be non-empty and equally binned",
        ));
    }
    let tp = pairwise_sum(p_counts);
    let tq = pairwise_sum(q_counts);
    if tp.is_nan() || tq.is_nan() || tp <= 0.0 || tq <= 0.0 {
        return Err(StatsError::new("JSD of an empty histogram is undefined"));
    }
    let p: Vec<f64> = p_counts.iter().map(|v| v / tp).collect();
    let q: Vec<f64> = q_counts.iter().map(|v| v / tq).collect();
    let m: Vec<f64> = p.iter().zip(&q).map(|(a, b)| 0.5 * (a + b)).collect();
    Ok(0.5 * kl(&p, &m) + 0.5 * kl(&q, &m))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsd_bounds_and_symmetry() {
        let p = [1.0, 2.0, 3.0, 0.0];
        assert_eq!(jsd(&p, &p).unwrap(), 0.0);
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        let d = jsd(&a, &b).unwrap();
        assert!((d - core::f64::consts::LN_2).abs() < 1e-15, "{d}");
        let q = [0.5, 1.0, 0.0, 4.0];
        assert_eq!(
            jsd(&p, &q).unwrap().to_bits(),
            jsd(&q, &p).unwrap().to_bits()
        );
        assert!(jsd(&[0.0], &[1.0]).is_err());
    }

    #[test]
    fn binning_edges() {
        let b = Binning::from_range(&[0.0, 10.0, 5.0], 10).unwrap();
        assert_eq!(b.bin(0.0), 0);
        assert_eq!(b.bin(10.0), 9);
        assert_eq!(b.bin(9.999), 9);
        assert_eq!(b.bin(5.0), 5);
        assert_eq!(b.bin(-1.0), 0);
        let flat = Binning::from_range(&[2.0, 2.0], 4).unwrap();
        assert_eq!(flat.bin(2.0), 0);
        let h = b.histogram(&[0.0, 10.0, 10.0], &[1.0, 2.0, 0.5]).unwrap();
        assert_eq!(h[9], 2.5);
    }
}
