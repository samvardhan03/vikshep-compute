//! Datasets, standardization, features from scattering ratios, and the
//! synthetic benchmark generator (`spec/VDS-1.md` sections 17.1 and 17.8).

use vikshep_numerics::oid::oid;
use vikshep_numerics::rng::Stream;
use vikshep_numerics::sum::pairwise_sum;
use vikshep_scatter::reduce::R2Output;

use crate::TrainError;

/// Frozen features with labels, Monte Carlo weights and the protected
/// variable.
#[derive(Clone, Debug, PartialEq)]
pub struct Dataset {
    /// Events.
    pub n: usize,
    /// Features per event.
    pub d: usize,
    /// Row-major `n x d` features.
    pub x: Vec<f64>,
    /// Labels: 1 signal, 0 background.
    pub y: Vec<u8>,
    /// Non-negative event weights.
    pub w: Vec<f64>,
    /// Protected variable (for example an invariant mass).
    pub m: Vec<f64>,
}

impl Dataset {
    /// Validate and build.
    pub fn new(
        d: usize,
        x: Vec<f64>,
        y: Vec<u8>,
        w: Vec<f64>,
        m: Vec<f64>,
    ) -> Result<Self, TrainError> {
        let n = y.len();
        if d == 0 || x.len() != n * d || w.len() != n || m.len() != n {
            return Err(TrainError::new("dataset arrays have inconsistent lengths"));
        }
        if y.iter().any(|&v| v > 1) {
            return Err(TrainError::new("labels must be 0 or 1"));
        }
        if x.iter().chain(&m).any(|v| !v.is_finite())
            || w.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(TrainError::new(
                "features, weights and protected values must be finite; weights >= 0",
            ));
        }
        Ok(Self { n, d, x, y, w, m })
    }

    /// Row `i`.
    #[must_use]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.x[i * self.d..(i + 1) * self.d]
    }

    /// Canonical bytes: `x` (binary64 LE, row-major), `y` (one byte each),
    /// `w`, `m` (binary64 LE).
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(self.n * (8 * self.d + 17));
        b.extend(self.x.iter().flat_map(|v| v.to_le_bytes()));
        b.extend(&self.y);
        b.extend(self.w.iter().flat_map(|v| v.to_le_bytes()));
        b.extend(self.m.iter().flat_map(|v| v.to_le_bytes()));
        b
    }

    /// OID of the canonical bytes.
    #[must_use]
    pub fn oid(&self) -> String {
        oid(&self.canonical_bytes())
    }
}

/// Per-feature standardization, as in the public Vikshep CLI:
/// `mu_j = psum_i(x_ij) / n`, `std_j = sqrt(psum_i((x_ij - mu_j)^2) / n) + 1e-8`.
#[derive(Clone, Debug, PartialEq)]
pub struct Standardizer {
    /// Means.
    pub mu: Vec<f64>,
    /// Standard deviations (with the `1e-8` floor added).
    pub std: Vec<f64>,
}

impl Standardizer {
    /// Fit on `x` (`n x d`, row-major).
    #[must_use]
    pub fn fit(x: &[f64], n: usize, d: usize) -> Self {
        let mut mu = Vec::with_capacity(d);
        let mut std = Vec::with_capacity(d);
        let mut col = vec![0.0f64; n];
        for j in 0..d {
            for (i, c) in col.iter_mut().enumerate() {
                *c = x[i * d + j];
            }
            let m = pairwise_sum(&col) / n as f64;
            let sq: Vec<f64> = col.iter().map(|v| (v - m) * (v - m)).collect();
            mu.push(m);
            std.push((pairwise_sum(&sq) / n as f64).sqrt() + 1e-8);
        }
        Self { mu, std }
    }

    /// `(x_j - mu_j) / std_j` for one row.
    #[must_use]
    pub fn apply_row(&self, row: &[f64]) -> Vec<f64> {
        row.iter()
            .zip(self.mu.iter().zip(&self.std))
            .map(|(v, (m, s))| (v - m) / s)
            .collect()
    }

    /// Standardize every row.
    #[must_use]
    pub fn apply(&self, x: &[f64], d: usize) -> Vec<f64> {
        x.chunks_exact(d).flat_map(|r| self.apply_row(r)).collect()
    }
}

/// Features from C1 scale-free ratios: per event and r2 path, the pairwise
/// mean of the ratio over output positions. Returns `(batch x n_paths)`
/// row-major features.
#[must_use]
pub fn features_from_r2(r2: &R2Output) -> Vec<f64> {
    let p = r2.pairs.len();
    let mut out = Vec::with_capacity(r2.batch * p);
    for b in 0..r2.batch {
        for k in 0..p {
            let start = (b * p + k) * r2.out_len;
            let vals: Vec<f64> = r2.values[start..start + r2.out_len]
                .iter()
                .map(|&v| f64::from(v))
                .collect();
            out.push(pairwise_sum(&vals) / r2.out_len as f64);
        }
    }
    out
}

/// Philox stream base of the synthetic generator (VDS-1 section 15.4).
pub const STREAM_SYNTHETIC: u64 = 0x5_0000;

/// Synthetic benchmark sample (VDS-1 section 17.8), drawn sequentially from
/// `Stream(seed, STREAM_SYNTHETIC + split)` (wrapping) with `d >= 2` features. Per
/// event: `u = next_f64_unit()`; signal if `u < 0.3`. Protected variable:
/// signal `120 + 8 z` (z standard normal), background `50 + 150 u'`.
/// Features: `x0 = (signal ? 1 : 0) + z`, `x1 = 0.04 (m - 100) + z`,
/// `x2.. = z`. Weight: signal 1, background `0.5 + u''`.
pub fn synthetic(n: usize, d: usize, seed: u64, split: u64) -> Result<Dataset, TrainError> {
    if d < 2 {
        return Err(TrainError::new("the synthetic sample needs d >= 2"));
    }
    let mut s = Stream::new(seed, STREAM_SYNTHETIC.wrapping_add(split));
    let mut x = Vec::with_capacity(n * d);
    let mut y = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    let mut m = Vec::with_capacity(n);
    for _ in 0..n {
        let signal = s.next_f64_unit() < 0.3;
        let mass = if signal {
            120.0 + 8.0 * s.next_normal_f64()
        } else {
            50.0 + 150.0 * s.next_f64_unit()
        };
        x.push(if signal { 1.0 } else { 0.0 } + s.next_normal_f64());
        x.push(0.04 * (mass - 100.0) + s.next_normal_f64());
        for _ in 2..d {
            x.push(s.next_normal_f64());
        }
        y.push(u8::from(signal));
        w.push(if signal { 1.0 } else { 0.5 + s.next_f64_unit() });
        m.push(mass);
    }
    Dataset::new(d, x, y, w, m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_is_reproducible_and_mixed() {
        let a = synthetic(500, 4, 7, 0).unwrap();
        assert_eq!(a, synthetic(500, 4, 7, 0).unwrap());
        assert_ne!(a, synthetic(500, 4, 7, 1).unwrap());
        let sig = a.y.iter().filter(|&&v| v == 1).count();
        assert!((100..200).contains(&sig), "{sig}");
        assert_eq!(a.oid().len(), 28);
    }

    #[test]
    fn standardizer() {
        let x = [1.0, 10.0, 3.0, 30.0];
        let st = Standardizer::fit(&x, 2, 2);
        assert_eq!(st.mu, vec![2.0, 20.0]);
        assert!((st.std[0] - 1.0).abs() < 1e-7);
        assert_eq!(st.apply_row(&[2.0, 20.0]), vec![0.0, 0.0]);
    }
}
