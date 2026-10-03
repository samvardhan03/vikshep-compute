//! Fingerprint distributions and the Sliced Wasserstein-1 distance
//! (`spec/VDS-1.md` sections 19.1 and 19.2).
//!
//! An event's fingerprint distribution is the empirical distribution, with
//! equal weights, of its per-position log-coefficient vectors
//! `f_s = (ln(2^-20 + |c_{p,s}|))_p` (one point per output position `s`, one
//! coordinate per path `p`); its mean is exactly the C1 log-mean
//! fingerprint. A single log-mean vector is the one-point special case.

use vikshep_numerics::rng::Stream;
use vikshep_numerics::sum::pairwise_sum;
use vikshep_scatter::ScatterOutput;
use vikshep_scatter::reduce::LOG_MEAN_EPS;

use crate::AnomalyError;

/// Philox stream of the projection directions (VDS-1 section 15.4).
pub const STREAM_SW1_DIRECTIONS: u64 = 0x4_0001;

/// A point cloud: `n_points` points of dimension `dim`, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct Cloud {
    /// Dimension of each point.
    pub dim: usize,
    /// Row-major points.
    pub points: Vec<f64>,
}

impl Cloud {
    /// Build from row-major points.
    pub fn new(dim: usize, points: Vec<f64>) -> Result<Self, AnomalyError> {
        if dim == 0 || points.is_empty() || !points.len().is_multiple_of(dim) {
            return Err(AnomalyError::new(
                "a cloud needs at least one point of positive dimension",
            ));
        }
        if points.iter().any(|v| !v.is_finite()) {
            return Err(AnomalyError::new("cloud points must be finite"));
        }
        Ok(Self { dim, points })
    }

    /// Number of points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len() / self.dim
    }

    /// True if the cloud has no points (never for a constructed cloud).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// Fingerprint distributions of every signal of a scattering output.
pub fn fingerprint_clouds(out: &ScatterOutput) -> Result<Vec<Cloud>, AnomalyError> {
    let n_paths = out.paths.len();
    let n_pos = out.out_len();
    (0..out.batch)
        .map(|b| {
            let mut pts = vec![0.0f64; n_pos * n_paths];
            for p in 0..n_paths {
                for (s, &c) in out.path(b, p).iter().enumerate() {
                    pts[s * n_paths + p] = vikshep_detmath::ln(LOG_MEAN_EPS + f64::from(c).abs());
                }
            }
            Cloud::new(n_paths, pts)
        })
        .collect()
}

/// Unit projection directions: direction `k` has components
/// `g_kc = next_normal_f64()` (drawn in order `k`, then `c`) divided by
/// `sqrt(psum_c(g_kc^2))`.
#[derive(Clone, Debug, PartialEq)]
pub struct Directions {
    /// Dimension.
    pub dim: usize,
    /// `count x dim`, row-major.
    pub theta: Vec<f64>,
}

impl Directions {
    /// `count` directions in `dim` dimensions from `Stream(seed,
    /// STREAM_SW1_DIRECTIONS)`.
    pub fn new(dim: usize, count: usize, seed: u64) -> Result<Self, AnomalyError> {
        if dim == 0 || count == 0 {
            return Err(AnomalyError::new(
                "need at least one direction of positive dimension",
            ));
        }
        let mut s = Stream::new(seed, STREAM_SW1_DIRECTIONS);
        let mut theta = Vec::with_capacity(dim * count);
        for _ in 0..count {
            let g: Vec<f64> = (0..dim).map(|_| s.next_normal_f64()).collect();
            let sq: Vec<f64> = g.iter().map(|v| v * v).collect();
            let norm = pairwise_sum(&sq).sqrt();
            if norm > 0.0 {
                theta.extend(g.iter().map(|v| v / norm));
            } else {
                theta.push(1.0);
                theta.extend(std::iter::repeat_n(0.0, dim - 1));
            }
        }
        Ok(Self { dim, theta })
    }

    /// Number of directions.
    #[must_use]
    pub fn count(&self) -> usize {
        self.theta.len() / self.dim
    }

    /// Sorted projections of a cloud onto every direction (sequential dot
    /// products, ascending total order).
    pub fn project(&self, cloud: &Cloud) -> Result<Projected, AnomalyError> {
        if cloud.dim != self.dim {
            return Err(AnomalyError::new(
                "cloud dimension differs from the directions",
            ));
        }
        let sorted = self
            .theta
            .chunks_exact(self.dim)
            .map(|t| {
                let mut v: Vec<f64> = cloud
                    .points
                    .chunks_exact(self.dim)
                    .map(|p| p.iter().zip(t).fold(0.0, |acc, (a, b)| acc + a * b))
                    .collect();
                v.sort_by(f64::total_cmp);
                v
            })
            .collect();
        Ok(Projected { sorted })
    }
}

/// A cloud's sorted projections, one list per direction.
#[derive(Clone, Debug, PartialEq)]
pub struct Projected {
    /// `sorted[k]` are the ascending projections on direction `k`.
    pub sorted: Vec<Vec<f64>>,
}

/// Exact 1-D Wasserstein-1 between two equal-weight empirical distributions
/// given as ascending samples: `integral |F_a(z) - F_b(z)| dz`, by merging
/// the samples (on equal values, `a` first) and summing, pairwise, the terms
/// `(|i m - j n| / (n m)) (z_next - z)` over consecutive merged values.
#[must_use]
pub fn w1_sorted(a: &[f64], b: &[f64]) -> f64 {
    let (n, m) = (a.len(), b.len());
    let (mut i, mut j) = (0usize, 0usize);
    let mut prev: Option<f64> = None;
    let nm = (n * m) as f64;
    let mut terms = Vec::with_capacity(n + m);
    while i < n || j < m {
        let take_a = j == m || (i < n && a[i].total_cmp(&b[j]).is_le());
        let z = if take_a { a[i] } else { b[j] };
        if let Some(p) = prev {
            let gap = ((i * m).abs_diff(j * n)) as f64 / nm;
            terms.push(gap * (z - p));
        }
        if take_a {
            i += 1;
        } else {
            j += 1;
        }
        prev = Some(z);
    }
    pairwise_sum(&terms)
}

/// Sliced Wasserstein-1: the pairwise mean over directions of the 1-D W1
/// between the projected distributions.
#[must_use]
pub fn sw1(a: &Projected, b: &Projected) -> f64 {
    let per: Vec<f64> = a
        .sorted
        .iter()
        .zip(&b.sorted)
        .map(|(x, y)| w1_sorted(x, y))
        .collect();
    pairwise_sum(&per) / per.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w1_known_values() {
        assert_eq!(w1_sorted(&[0.0], &[3.0]), 3.0);
        assert_eq!(w1_sorted(&[0.0, 1.0], &[0.0, 1.0]), 0.0);
        // shift by 2
        assert_eq!(w1_sorted(&[0.0, 1.0, 5.0], &[2.0, 3.0, 7.0]), 2.0);
        // unequal sizes: {0} vs {0, 2}: integral of |1 - 1/2| over [0, 2] = 1
        assert_eq!(w1_sorted(&[0.0], &[0.0, 2.0]), 1.0);
        assert_eq!(w1_sorted(&[0.0, 2.0], &[0.0]), 1.0);
    }

    #[test]
    fn sw1_is_a_metric_on_samples() {
        let d = Directions::new(3, 16, 5).unwrap();
        for t in d.theta.chunks(3) {
            let n: f64 = t.iter().map(|v| v * v).sum();
            assert!((n - 1.0).abs() < 1e-15);
        }
        let a = Cloud::new(3, vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0]).unwrap();
        let b = Cloud::new(3, vec![0.5, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0]).unwrap();
        let (pa, pb) = (d.project(&a).unwrap(), d.project(&b).unwrap());
        assert_eq!(sw1(&pa, &pa), 0.0);
        assert_eq!(sw1(&pa, &pb).to_bits(), sw1(&pb, &pa).to_bits());
        assert!(sw1(&pa, &pb) > 0.0);
    }
}
