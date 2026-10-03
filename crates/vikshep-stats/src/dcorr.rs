//! Weighted distance correlation (Szekely-Rizzo, weighted form) and its
//! exact gradient (`spec/VDS-1.md` section 16.1, `docs/disco_gradient.md`).
//!
//! With normalized weights `v_i = w_i / sum(w)`, `a_ij = |x_i - x_j|`,
//! `r_i = sum_j v_j a_ij`, `mu = sum_i v_i r_i` and
//! `A_ij = a_ij - r_i - r_j + mu` (and `B` likewise for `y`):
//!
//! ```text
//! dCov2(x, y)  = sum_ij v_i v_j A_ij B_ij
//! dCorr2(x, y) = dCov2(x, y) / sqrt(dCov2(x, x) dCov2(y, y))
//! ```
//!
//! Sums over events are pairwise trees (`vikshep_numerics::sum::pairwise_sum`):
//! every inner sum over `j` per row `i`, then the outer sum over `i`. Rows are
//! computed in parallel and combined in index order.

use rayon::prelude::*;
use vikshep_numerics::rng::Stream;
use vikshep_numerics::sum::pairwise_sum;

use crate::StatsError;

/// Above this many events, [`weighted_dcorr2`] uses the chunked estimator.
pub const CHUNK_THRESHOLD: usize = 10_000;
/// Largest chunk of the chunked estimator.
pub const CHUNK_MAX: usize = 4096;
/// Denominators below this make dCorr2 (and its gradient) exactly 0.
pub const DENOM_EPS: f64 = 1e-12;
/// Philox stream of the chunked estimator's permutation (VDS-1 section 15.4).
pub const STREAM_CHUNK_PERMUTATION: u64 = 0x2_0001;

/// Validate inputs and return normalized weights.
fn normalized_weights(n: usize, y_len: usize, w: Option<&[f64]>) -> Result<Vec<f64>, StatsError> {
    if y_len != n {
        return Err(StatsError::new(format!(
            "x and y must have the same length ({n} vs {y_len})"
        )));
    }
    let w: Vec<f64> = match w {
        None => vec![1.0; n],
        Some(w) => {
            if w.len() != n {
                return Err(StatsError::new(format!(
                    "w must have the same length as x ({} vs {n})",
                    w.len()
                )));
            }
            w.to_vec()
        }
    };
    if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(StatsError::new("weights must be finite and non-negative"));
    }
    let total = pairwise_sum(&w);
    if total.is_nan() || total <= 0.0 {
        return Err(StatsError::new("weights must not all be zero"));
    }
    Ok(w.iter().map(|v| v / total).collect())
}

fn check_finite(name: &str, v: &[f64]) -> Result<(), StatsError> {
    if v.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(StatsError::new(format!(
            "{name} contains a non-finite value"
        )))
    }
}

/// Weighted row means `r_i = sum_j v_j |x_i - x_j|` and grand mean
/// `mu = sum_i v_i r_i`.
fn centering(x: &[f64], v: &[f64]) -> (Vec<f64>, f64) {
    let n = x.len();
    let r: Vec<f64> = (0..n)
        .into_par_iter()
        .map_init(
            || vec![0.0f64; n],
            |buf, i| {
                for (j, b) in buf.iter_mut().enumerate() {
                    *b = v[j] * (x[i] - x[j]).abs();
                }
                pairwise_sum(buf)
            },
        )
        .collect();
    let terms: Vec<f64> = r.iter().zip(v).map(|(ri, vi)| vi * ri).collect();
    (r, pairwise_sum(&terms))
}

/// `A_ij = ((a_ij - r_i) - r_j) + mu`.
#[inline]
fn centered(x: &[f64], r: &[f64], mu: f64, i: usize, j: usize) -> f64 {
    (((x[i] - x[j]).abs() - r[i]) - r[j]) + mu
}

/// The three weighted inner products `(dCov2(x,y), dCov2(x,x), dCov2(y,y))`.
struct Moments {
    rx: Vec<f64>,
    mux: f64,
    ry: Vec<f64>,
    muy: f64,
    xy: f64,
    xx: f64,
    yy: f64,
}

fn moments(x: &[f64], y: &[f64], v: &[f64]) -> Moments {
    let n = x.len();
    let (rx, mux) = centering(x, v);
    let (ry, muy) = centering(y, v);
    let rows: Vec<[f64; 3]> = (0..n)
        .into_par_iter()
        .map_init(
            || (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]),
            |(bxy, bxx, byy), i| {
                for j in 0..n {
                    let a = centered(x, &rx, mux, i, j);
                    let b = centered(y, &ry, muy, i, j);
                    bxy[j] = v[j] * a * b;
                    bxx[j] = v[j] * a * a;
                    byy[j] = v[j] * b * b;
                }
                [pairwise_sum(bxy), pairwise_sum(bxx), pairwise_sum(byy)]
            },
        )
        .collect();
    let outer = |k: usize| {
        let t: Vec<f64> = rows.iter().zip(v).map(|(row, vi)| vi * row[k]).collect();
        pairwise_sum(&t)
    };
    Moments {
        xy: outer(0),
        xx: outer(1),
        yy: outer(2),
        rx,
        mux,
        ry,
        muy,
    }
}

/// `dCov2(x,y) / sqrt(dCov2(x,x) dCov2(y,y))`, or 0 when the denominator is
/// below [`DENOM_EPS`].
fn ratio(m: &Moments) -> (f64, f64) {
    let prod = m.xx * m.yy;
    let den = if prod > 0.0 { prod.sqrt() } else { 0.0 };
    if den < DENOM_EPS {
        (0.0, den)
    } else {
        (m.xy / den, den)
    }
}

/// Exact O(n^2) weighted dCorr2 (any `n >= 2`; 0 for `n < 2`).
pub fn dcorr2_exact(x: &[f64], y: &[f64], w: Option<&[f64]>) -> Result<f64, StatsError> {
    let v = normalized_weights(x.len(), y.len(), w)?;
    check_finite("x", x)?;
    check_finite("y", y)?;
    if x.len() < 2 {
        return Ok(0.0);
    }
    Ok(ratio(&moments(x, y, &v)).0)
}

/// Chunked estimator (an estimator, not the exact statistic; VDS-1 section
/// 16.1.2): permute the events with Philox stream
/// `(seed, STREAM_CHUNK_PERMUTATION)`, split the permutation into
/// consecutive chunks of `c = min(CHUNK_MAX, floor(n / 2))` events (a final
/// partial chunk is dropped), compute the exact dCorr2 of each chunk with its
/// own weights, and return the pairwise mean of the chunk values.
pub fn dcorr2_chunked(
    x: &[f64],
    y: &[f64],
    w: Option<&[f64]>,
    seed: u64,
) -> Result<f64, StatsError> {
    normalized_weights(x.len(), y.len(), w)?;
    check_finite("x", x)?;
    check_finite("y", y)?;
    let n = x.len();
    let chunk = CHUNK_MAX.min(n / 2);
    if chunk < 4 {
        return dcorr2_exact(x, y, w);
    }
    let perm = Stream::new(seed, STREAM_CHUNK_PERMUTATION).permutation(n);
    let ones = vec![1.0; n];
    let w = w.unwrap_or(&ones);
    let mut estimates = Vec::new();
    let mut start = 0;
    while start + chunk <= n {
        let idx = &perm[start..start + chunk];
        let cx: Vec<f64> = idx.iter().map(|&i| x[i]).collect();
        let cy: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
        let cw: Vec<f64> = idx.iter().map(|&i| w[i]).collect();
        estimates.push(dcorr2_exact(&cx, &cy, Some(&cw))?);
        start += chunk;
    }
    Ok(pairwise_sum(&estimates) / estimates.len() as f64)
}

/// Weighted dCorr2: exact for `n <= CHUNK_THRESHOLD`, chunked above
/// (seed 0, matching the threshold of the Python reference metric).
pub fn weighted_dcorr2(x: &[f64], y: &[f64], w: Option<&[f64]>) -> Result<f64, StatsError> {
    if x.len() > CHUNK_THRESHOLD {
        dcorr2_chunked(x, y, w, 0)
    } else {
        dcorr2_exact(x, y, w)
    }
}

/// dCorr2 and its exact gradient with respect to `s` (`y = m` held fixed):
///
/// ```text
/// g_k = (2 v_k / sqrt(Vs Vm)) sum_j v_j sign(s_k - s_j) (B_kj - (D / Vs) A_kj)
/// ```
///
/// with `D = dCov2(s, m)`, `Vs = dCov2(s, s)`, `Vm = dCov2(m, m)`,
/// `sign(0) = 0`; the gradient is 0 wherever dCorr2 is defined as 0
/// (denominator below [`DENOM_EPS`]). Derivation: `docs/disco_gradient.md`.
pub fn dcorr2_grad(s: &[f64], m: &[f64], w: Option<&[f64]>) -> Result<(f64, Vec<f64>), StatsError> {
    let v = normalized_weights(s.len(), m.len(), w)?;
    check_finite("scores", s)?;
    check_finite("protected variable", m)?;
    let n = s.len();
    if n < 2 {
        return Ok((0.0, vec![0.0; n]));
    }
    let mo = moments(s, m, &v);
    let (value, den) = ratio(&mo);
    if den < DENOM_EPS {
        return Ok((0.0, vec![0.0; n]));
    }
    let c = mo.xy / mo.xx;
    let grad: Vec<f64> = (0..n)
        .into_par_iter()
        .map_init(
            || vec![0.0f64; n],
            |buf, k| {
                for (j, b) in buf.iter_mut().enumerate() {
                    let d = s[k] - s[j];
                    let sign = if d > 0.0 {
                        1.0
                    } else if d < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    let a = centered(s, &mo.rx, mo.mux, k, j);
                    let bb = centered(m, &mo.ry, mo.muy, k, j);
                    *b = v[j] * sign * (bb - c * a);
                }
                ((2.0 * v[k]) / den) * pairwise_sum(buf)
            },
        )
        .collect();
    Ok((value, grad))
}

/// FAST MODE, NOT THE EXACT GRADIENT: the Pearson-correlation proxy of the
/// public Vikshep CLI (`backend/ingest/src/vikshep_ingest/cli/_train.py`,
/// `_pearson_dcorr2_grad`), ported for comparison. It is the gradient of
/// `r^2`, the squared weighted Pearson correlation of `s` and `m`, with the
/// same `1e-12` regularizers, and is O(n).
pub fn pearson_proxy_grad(s: &[f64], m: &[f64], w: &[f64]) -> Result<Vec<f64>, StatsError> {
    if s.len() != m.len() || s.len() != w.len() {
        return Err(StatsError::new(
            "scores, masses and weights must have the same length",
        ));
    }
    let total = pairwise_sum(w) + 1e-12;
    let v: Vec<f64> = w.iter().map(|x| x / total).collect();
    let dot = |a: &[f64], b: &[f64]| {
        let t: Vec<f64> = a.iter().zip(b).map(|(p, q)| p * q).collect();
        pairwise_sum(&t)
    };
    let mu_s = dot(s, &v);
    let mu_m = dot(m, &v);
    let ds: Vec<f64> = s.iter().map(|x| x - mu_s).collect();
    let dm: Vec<f64> = m.iter().map(|x| x - mu_m).collect();
    let dsdm: Vec<f64> = ds.iter().zip(&dm).map(|(a, b)| a * b).collect();
    let ds2: Vec<f64> = ds.iter().map(|a| a * a).collect();
    let dm2: Vec<f64> = dm.iter().map(|a| a * a).collect();
    let cov = dot(&dsdm, &v);
    let var_s = dot(&ds2, &v) + 1e-12;
    let var_m = dot(&dm2, &v) + 1e-12;
    let sd = (var_s * var_m).sqrt();
    let r = cov / sd;
    Ok((0..s.len())
        .map(|i| 2.0 * r * v[i] * (dm[i] / sd - r * ds[i] / var_s))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut s = Stream::new(seed, 1);
        let x: Vec<f64> = (0..n).map(|_| s.next_normal_f64()).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|v| v * v + 0.3 * s.next_normal_f64())
            .collect();
        let w: Vec<f64> = (0..n).map(|_| 0.1 + s.next_f64_unit()).collect();
        (x, y, w)
    }

    #[test]
    fn identities() {
        let (x, y, w) = data(150, 1);
        let d = dcorr2_exact(&x, &x, Some(&w)).unwrap();
        assert!((d - 1.0).abs() < 1e-12, "{d}");
        let r = dcorr2_exact(&x, &y, Some(&w)).unwrap();
        assert!(r > 0.1 && r < 1.0);
        // affine invariance in each variable
        let y2: Vec<f64> = y.iter().map(|v| 3.0 * v - 7.0).collect();
        let r2 = dcorr2_exact(&x, &y2, Some(&w)).unwrap();
        assert!((r - r2).abs() < 1e-12);
        // weight scale invariance
        let w7: Vec<f64> = w.iter().map(|v| 7.0 * v).collect();
        assert!((r - dcorr2_exact(&x, &y, Some(&w7)).unwrap()).abs() < 1e-12);
        assert_eq!(dcorr2_exact(&[1.0], &[2.0], None).unwrap(), 0.0);
        assert_eq!(
            dcorr2_exact(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0], None).unwrap(),
            0.0
        );
        assert!(dcorr2_exact(&[1.0, 2.0], &[1.0], None).is_err());
    }

    #[test]
    fn gradient_value_matches_statistic() {
        let (x, y, w) = data(80, 2);
        let (v, g) = dcorr2_grad(&x, &y, Some(&w)).unwrap();
        assert_eq!(
            v.to_bits(),
            dcorr2_exact(&x, &y, Some(&w)).unwrap().to_bits()
        );
        assert_eq!(g.len(), 80);
        // translation invariance: the gradient sums to zero
        let total: f64 = g.iter().sum();
        assert!(total.abs() < 1e-12, "{total}");
    }

    #[test]
    fn chunked_is_deterministic_and_close() {
        let (x, y, w) = data(2000, 3);
        let a = dcorr2_chunked(&x, &y, Some(&w), 11).unwrap();
        let b = dcorr2_chunked(&x, &y, Some(&w), 11).unwrap();
        assert_eq!(a.to_bits(), b.to_bits());
        let exact = dcorr2_exact(&x, &y, Some(&w)).unwrap();
        assert!((a - exact).abs() < 0.05, "{a} vs {exact}");
    }
}
