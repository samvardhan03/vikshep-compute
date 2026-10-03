//! Calibration: ridge regression in closed form with a deterministic
//! Cholesky factorization (`spec/VDS-1.md` section 18).

use vikshep_numerics::jcs::{Value, object};
use vikshep_numerics::oid::oid;
use vikshep_numerics::sum::pairwise_sum;

use crate::TrainError;
use crate::data::Standardizer;

/// Default ridge strength (as in the public Vikshep CLI).
pub const DEFAULT_RIDGE: f64 = 1e-4;

/// Result of a calibration fit.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    /// Feature standardization.
    pub standardizer: Standardizer,
    /// Coefficients in standardized feature space.
    pub coef: Vec<f64>,
    /// Intercept.
    pub bias: f64,
    /// `1 - ss_res / (ss_tot + 1e-12)`.
    pub r2: f64,
    /// Population standard deviation of the residuals.
    pub residual_std: f64,
    /// `t - prediction` per event.
    pub residuals: Vec<f64>,
}

/// Cholesky factor `L` (row-major, lower) of a symmetric positive-definite
/// `d x d` matrix: for `i` ascending, `j = 0..=i`,
/// `s = A_ij - sum_{k<j} L_ik L_jk` (sequential in `k`), `L_ii = sqrt(s)`
/// (`s > 0` required), `L_ij = s / L_jj`.
pub fn cholesky(a: &[f64], d: usize) -> Result<Vec<f64>, TrainError> {
    let mut l = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut s = a[i * d + j];
            for k in 0..j {
                s -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                if s.is_nan() || s <= 0.0 {
                    return Err(TrainError::new("matrix is not positive definite"));
                }
                l[i * d + i] = s.sqrt();
            } else {
                l[i * d + j] = s / l[j * d + j];
            }
        }
    }
    Ok(l)
}

/// Solve `L L^T x = b`: forward substitution for `L y = b`, then backward
/// for `L^T x = y`, inner sums sequential.
#[must_use]
pub fn cholesky_solve(l: &[f64], d: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0f64; d];
    for i in 0..d {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * d + k] * y[k];
        }
        y[i] = s / l[i * d + i];
    }
    let mut x = vec![0.0f64; d];
    for i in (0..d).rev() {
        let mut s = y[i];
        for k in i + 1..d {
            s -= l[k * d + i] * x[k];
        }
        x[i] = s / l[i * d + i];
    }
    x
}

/// Ridge regression of `t` on `x` (`n x d`, row-major): standardize the
/// features, form `G = X^T X + ridge I` and `c = X^T t` with pairwise sums
/// over events, solve `G w = c` by Cholesky, `bias = mean(t) - mean(X w)`.
pub fn ridge(
    x: &[f64],
    n: usize,
    d: usize,
    t: &[f64],
    ridge: f64,
) -> Result<Calibration, TrainError> {
    if d == 0 || n == 0 || x.len() != n * d || t.len() != n {
        return Err(TrainError::new(
            "calibration inputs have inconsistent lengths",
        ));
    }
    if x.iter().chain(t).any(|v| !v.is_finite()) || ridge.is_nan() || ridge < 0.0 {
        return Err(TrainError::new(
            "calibration inputs must be finite and ridge >= 0",
        ));
    }
    let standardizer = Standardizer::fit(x, n, d);
    let xs = standardizer.apply(x, d);
    let mut g = vec![0.0f64; d * d];
    let mut c = vec![0.0f64; d];
    let mut col = vec![0.0f64; n];
    for j in 0..d {
        for k in 0..=j {
            for (i, v) in col.iter_mut().enumerate() {
                *v = xs[i * d + j] * xs[i * d + k];
            }
            let s = pairwise_sum(&col);
            g[j * d + k] = s;
            g[k * d + j] = s;
        }
        g[j * d + j] += ridge;
        for (i, v) in col.iter_mut().enumerate() {
            *v = xs[i * d + j] * t[i];
        }
        c[j] = pairwise_sum(&col);
    }
    let l = cholesky(&g, d)?;
    let coef = cholesky_solve(&l, d, &c);
    let fit: Vec<f64> = xs
        .chunks_exact(d)
        .map(|r| crate::model::dot(r, &coef))
        .collect();
    let mean_t = pairwise_sum(t) / n as f64;
    let bias = mean_t - pairwise_sum(&fit) / n as f64;
    let residuals: Vec<f64> = t.iter().zip(&fit).map(|(ti, f)| ti - (f + bias)).collect();
    let sq: Vec<f64> = residuals.iter().map(|r| r * r).collect();
    let ss_res = pairwise_sum(&sq);
    let dev: Vec<f64> = t.iter().map(|ti| (ti - mean_t) * (ti - mean_t)).collect();
    let ss_tot = pairwise_sum(&dev) + 1e-12;
    let mean_r = pairwise_sum(&residuals) / n as f64;
    let rdev: Vec<f64> = residuals
        .iter()
        .map(|r| (r - mean_r) * (r - mean_r))
        .collect();
    Ok(Calibration {
        standardizer,
        coef,
        bias,
        r2: 1.0 - ss_res / ss_tot,
        residual_std: (pairwise_sum(&rdev) / n as f64).sqrt(),
        residuals,
    })
}

impl Calibration {
    /// Predictions for raw rows.
    #[must_use]
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        x.chunks_exact(self.coef.len())
            .map(|r| crate::model::dot(&self.standardizer.apply_row(r), &self.coef) + self.bias)
            .collect()
    }

    /// Residual tensor bytes (binary64 LE).
    #[must_use]
    pub fn residual_bytes(&self) -> Vec<u8> {
        self.residuals
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    }

    /// Calibration constants as canonical JSON: coefficients, bias,
    /// standardization, fit quality, and the OID of the residual tensor.
    #[must_use]
    pub fn constants(&self) -> Value {
        let nums = |v: &[f64]| Value::Array(v.iter().map(|&x| Value::Num(x)).collect());
        object([
            ("bias", Value::Num(self.bias)),
            ("coef", nums(&self.coef)),
            ("kind", Value::Str("vikshep.calibration".into())),
            ("mu", nums(&self.standardizer.mu)),
            ("r2", Value::Num(self.r2)),
            ("residual_std", Value::Num(self.residual_std)),
            ("residuals_oid", Value::Str(oid(&self.residual_bytes()))),
            ("std", nums(&self.standardizer.std)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_solves_spd_system() {
        let a = [4.0, 2.0, 0.6, 2.0, 5.0, 1.0, 0.6, 1.0, 3.0];
        let l = cholesky(&a, 3).unwrap();
        let x = cholesky_solve(&l, 3, &[1.0, 2.0, 3.0]);
        for i in 0..3 {
            let r: f64 = (0..3).map(|k| a[i * 3 + k] * x[k]).sum();
            assert!((r - [1.0, 2.0, 3.0][i]).abs() < 1e-14);
        }
        assert!(cholesky(&[1.0, 2.0, 2.0, 1.0], 2).is_err());
    }

    #[test]
    fn ridge_recovers_a_linear_map() {
        let n = 200;
        let mut s = vikshep_numerics::rng::Stream::new(4, 4);
        let x: Vec<f64> = (0..n * 3).map(|_| s.next_normal_f64()).collect();
        let t: Vec<f64> = x
            .chunks(3)
            .map(|r| 2.0 * r[0] - r[1] + 0.5 * r[2] + 1.0)
            .collect();
        let c = ridge(&x, n, 3, &t, DEFAULT_RIDGE).unwrap();
        assert!(c.r2 > 0.999_999);
        let p = c.predict(&x);
        assert!(p.iter().zip(&t).all(|(a, b)| (a - b).abs() < 1e-3));
        assert!(
            c.constants()
                .canonical()
                .unwrap()
                .contains("\"kind\":\"vikshep.calibration\"")
        );
    }
}
