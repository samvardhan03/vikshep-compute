//! Host-side filter banks (VDS-1 sections 14.2 and 14.3).
//!
//! Every filter is computed in binary64 with `vikshep-detmath`, stored as a
//! real-valued Fourier-domain filter, truncated (values below `2^-40` of the
//! filter's peak become exact zeros) and rounded once to binary32.
//!
//! The parameterization follows Kymatio 0.3.0 (BSD-3-Clause):
//! `kymatio/scattering1d/filter_bank.py` (1-D, built directly in the Fourier
//! domain) and `kymatio/scattering2d/filter_bank.py` (2-D, built in the
//! spatial domain and transformed with the binary64 Stockham FFT). The
//! deviations from Kymatio are listed in VDS-1 section 14.3.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;
use vikshep_backend_api::Canvas;
use vikshep_detmath::{cos, exp, ln, sin};
use vikshep_numerics::fft::{Complex64, Direction, fft_2d};
use vikshep_numerics::oid::{hex, sha3_256};

use crate::config::ScatterConfig;

/// Kymatio `sigma0`: low-pass width at scale 0 (1-D).
pub const SIGMA0_1D: f64 = 0.1;
/// Kymatio `alpha`: used to assign the dyadic scale `j` of a 1-D filter.
pub const ALPHA_1D: f64 = 5.0;
/// Kymatio `eps` of `adaptive_choice_P`.
pub const EPS_P_1D: f64 = 1e-7;
/// Kymatio `P_max`.
pub const P_MAX_1D: usize = 5;
/// Fourier-filter truncation: values with `|h| < 2^-40 * max|h|` become 0.
pub const TRUNCATION_LOG2: i32 = -40;

/// One real-valued Fourier-domain filter.
#[derive(Clone, Debug, PartialEq)]
pub struct Filter {
    /// Dyadic scale `j` (Kymatio's `j`).
    pub j: u32,
    /// Orientation index `l` (2-D), angle `l * pi / L`.
    pub theta: Option<u32>,
    /// Centre frequency: cycles per sample (1-D) or radians per sample (2-D).
    pub xi: f64,
    /// Width parameter: Fourier-domain sigma in cycles per sample (1-D) or
    /// spatial sigma in samples (2-D).
    pub sigma: f64,
    /// Filter values on the canvas frequency grid (row-major), binary32.
    pub fourier: Vec<f32>,
}

/// The filters of one configuration's canvas.
#[derive(Clone, Debug)]
pub struct FilterBank {
    /// Canvas the filters are sampled on.
    pub canvas: Canvas,
    /// Low-pass filter `phi_J`.
    pub phi: Filter,
    /// First-order wavelets in canonical order.
    pub psi1: Vec<Filter>,
    /// Second-order wavelets (1-D: the Q = 1 bank); `None` in 2-D, where the
    /// second order reuses `psi1`.
    pub psi2: Option<Vec<Filter>>,
    /// Largest discarded imaginary part relative to the largest real part
    /// over all filters (0 for 1-D filters, which are real by construction).
    pub max_imag_ratio: f64,
}

impl FilterBank {
    /// Second-order wavelets.
    #[must_use]
    pub fn second_order(&self) -> &[Filter] {
        self.psi2.as_deref().unwrap_or(&self.psi1)
    }

    /// All filter bytes in canonical order: `phi`, the first-order bank,
    /// then the second-order bank when it is distinct (1-D).
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut push = |f: &Filter| out.extend(f.fourier.iter().flat_map(|v| v.to_le_bytes()));
        push(&self.phi);
        self.psi1.iter().for_each(&mut push);
        if let Some(p2) = &self.psi2 {
            p2.iter().for_each(&mut push);
        }
        out
    }

    /// Filter-bank fingerprint: lowercase hex SHA3-256 of
    /// [`Self::canonical_bytes`] (VDS-1 section 14.3.4).
    #[must_use]
    pub fn fingerprint(&self) -> String {
        hex(&sha3_256(&self.canonical_bytes()))
    }

    /// Littlewood-Paley bounds `(min, max)` over all frequencies of
    /// `|phi(w)|^2 + 1/2 sum_psi (|psi(w)|^2 + |psi(-w)|^2)` for the given
    /// wavelet bank (the symmetrisation accounts for analytic wavelets
    /// applied to real signals).
    #[must_use]
    pub fn littlewood_paley(&self, bank: &[Filter]) -> (f64, f64) {
        let (rows, cols) = (self.canvas.rows, self.canvas.cols);
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for r in 0..rows {
            for c in 0..cols {
                let i = r * cols + c;
                let ni = ((rows - r) % rows) * cols + (cols - c) % cols;
                let p = f64::from(self.phi.fourier[i]);
                let mut a = p * p;
                for f in bank {
                    let x = f64::from(f.fourier[i]);
                    let y = f64::from(f.fourier[ni]);
                    a += 0.5 * (x * x + y * y);
                }
                lo = lo.min(a);
                hi = hi.max(a);
            }
        }
        (lo, hi)
    }
}

/// Exact `2^e` for integer `e`; `exp(e * ln 2)` otherwise (Kymatio's
/// `math.pow(2, e)`).
fn pow2_real(e: f64) -> f64 {
    if e == e.trunc() && e.abs() < 1000.0 {
        let mut v = 1.0;
        let k = e as i32;
        for _ in 0..k.abs() {
            v = if k > 0 { v * 2.0 } else { v * 0.5 };
        }
        v
    } else {
        exp(e * core::f64::consts::LN_2)
    }
}

/// Exact `2^-k`.
fn pow2_neg(k: u32) -> f64 {
    pow2_real(-f64::from(k))
}

/// Kymatio `compute_sigma_psi`.
fn sigma_psi(xi: f64, q: u32, r: f64) -> f64 {
    let factor = 1.0 / pow2_real(1.0 / f64::from(q));
    let term1 = (1.0 - factor) / (1.0 + factor);
    let term2 = 1.0 / (2.0 * ln(1.0 / r)).sqrt();
    xi * term1 * term2
}

/// Kymatio `compute_xi_max`.
fn xi_max(q: u32) -> f64 {
    (1.0 / (1.0 + pow2_real(3.0 / f64::from(q)))).max(0.35)
}

/// Kymatio `get_max_dyadic_subsampling`: `floor(-log2(ub)) - 1` with
/// `ub = min(xi + alpha * sigma, 0.5)`, evaluated with exact power-of-two
/// comparisons (`floor(-log2(ub))` is the largest `k` with `ub <= 2^-k`).
fn max_dyadic_subsampling(xi: f64, sigma: f64) -> u32 {
    let ub = (xi + ALPHA_1D * sigma).min(0.5);
    let mut k = 1;
    while ub <= pow2_neg(k + 1) {
        k += 1;
    }
    k - 1
}

/// Kymatio `compute_params_filterbank`: `(xi, sigma, j)` per filter.
#[must_use]
pub fn params_1d(j_scales: u32, q: u32) -> Vec<(f64, f64, u32)> {
    let r_psi = 0.5f64.sqrt();
    let sigma_min = SIGMA0_1D * pow2_neg(j_scales);
    let xmax = xi_max(q);
    let smax = sigma_psi(xmax, q, r_psi);
    let step = pow2_real(1.0 / f64::from(q));
    let mut xis = Vec::new();
    let mut sigmas = Vec::new();
    let elbow_xi = if smax <= sigma_min {
        smax
    } else {
        xis.push(xmax);
        sigmas.push(smax);
        while *sigmas.last().unwrap() > sigma_min * step {
            let x = *xis.last().unwrap() / step;
            let s = *sigmas.last().unwrap() / step;
            xis.push(x);
            sigmas.push(s);
        }
        *xis.last().unwrap()
    };
    for k in 1..q {
        xis.push(elbow_xi - f64::from(k) / f64::from(q) * elbow_xi);
        sigmas.push(sigma_min);
    }
    xis.into_iter()
        .zip(sigmas)
        .map(|(x, s)| (x, s, max_dyadic_subsampling(x, s)))
        .collect()
}

/// Kymatio `adaptive_choice_P`, capped at `P_max`.
fn periods_1d(sigma: f64) -> usize {
    let val = (-2.0 * (sigma * sigma) * ln(EPS_P_1D)).sqrt();
    let p = (val + 1.0).ceil() as usize;
    p.min(P_MAX_1D)
}

/// Host binary64 inverse DFT (spec FFT, 2^-m scaled) of a real sequence.
fn ifft_real_f64(h: &[f64], rows: usize, cols: usize) -> Vec<Complex64> {
    let mut z: Vec<Complex64> = h.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    fft_2d(&mut z, rows, cols, Direction::Inverse);
    z
}

/// Kymatio `morlet_1d` / `gauss_1d` (`xi = None`) on `n` frequency bins,
/// before truncation and rounding.
#[must_use]
pub fn morlet_1d_f64(n: usize, xi: Option<f64>, sigma: f64) -> Vec<f64> {
    let p = periods_1d(sigma);
    let periods = 2 * p - 1;
    let start = -((p as i64 - 1) * n as i64);
    let freq = |i: usize| (start + i as i64) as f64 / n as f64;
    let den = 2.0 * (sigma * sigma);
    let periodize = |center: f64| -> Vec<f64> {
        (0..n)
            .map(|k| {
                let mut acc = 0.0;
                for per in 0..periods {
                    let d = freq(per * n + k) - center;
                    acc += exp(-(d * d) / den);
                }
                acc / periods as f64
            })
            .collect()
    };
    let low = periodize(0.0);
    let mut filt = match xi {
        Some(x) => {
            let gabor = periodize(x);
            let kappa = gabor[0] / low[0];
            gabor.iter().zip(&low).map(|(g, l)| g - kappa * l).collect()
        }
        None => low,
    };
    let h = ifft_real_f64(&filt, 1, n);
    let l1 = h
        .iter()
        .fold(0.0, |acc, z| acc + (z.re * z.re + z.im * z.im).sqrt());
    for v in &mut filt {
        *v /= l1;
    }
    filt
}

/// Kymatio `gabor_2d` (`with_carrier = false` gives the `xi = 0` envelope
/// sum), returned together with its envelope sum so a Morlet can share the
/// envelope evaluations. Returns `(gabor, envelope)`, both already divided
/// by `2 * pi * sigma^2 / slant`.
fn gabor_2d_pair(
    rows: usize,
    cols: usize,
    sigma: f64,
    theta: f64,
    xi: f64,
    slant: f64,
) -> (Vec<Complex64>, Vec<f64>) {
    let (c, s) = (cos(theta), sin(theta));
    let sl2 = slant * slant;
    let den = 2.0 * sigma * sigma;
    let c00 = (c * c + sl2 * (s * s)) / den;
    let cross = (2.0 * (c * s) * (1.0 - sl2)) / den;
    let c11 = (s * s + sl2 * (c * c)) / den;
    let norm = 2.0 * core::f64::consts::PI * sigma * sigma / slant;
    let mut gab = vec![Complex64::new(0.0, 0.0); rows * cols];
    let mut env = vec![0.0f64; rows * cols];
    for n in 0..rows {
        for m in 0..cols {
            let mut acc = Complex64::new(0.0, 0.0);
            let mut acc_env = 0.0f64;
            for ex in -2i64..=2 {
                for ey in -2i64..=2 {
                    let x = (n as i64 + ex * rows as i64) as f64;
                    let y = (m as i64 + ey * cols as i64) as f64;
                    let re = -(c00 * (x * x) + cross * (x * y) + c11 * (y * y));
                    // exp(re) is exactly +0 below -745.14; adding (+-0) terms
                    // leaves the sums unchanged, so they are skipped.
                    if re < -746.0 {
                        continue;
                    }
                    let e = exp(re);
                    let im = x * xi * c + y * xi * s;
                    acc.re += e * cos(im);
                    acc.im += e * sin(im);
                    acc_env += e;
                }
            }
            gab[n * cols + m] = Complex64::new(acc.re / norm, acc.im / norm);
            env[n * cols + m] = acc_env / norm;
        }
    }
    (gab, env)
}

/// Kymatio `morlet_2d` in the spatial domain.
#[must_use]
pub fn morlet_2d_spatial(
    rows: usize,
    cols: usize,
    sigma: f64,
    theta: f64,
    xi: f64,
    slant: f64,
) -> Vec<Complex64> {
    let (wv, env) = gabor_2d_pair(rows, cols, sigma, theta, xi, slant);
    let mut sw = Complex64::new(0.0, 0.0);
    let mut se = 0.0;
    for (w, e) in wv.iter().zip(&env) {
        sw.re += w.re;
        sw.im += w.im;
        se += e;
    }
    let k = Complex64::new(sw.re / se, sw.im / se);
    wv.iter()
        .zip(&env)
        .map(|(w, e)| Complex64::new(w.re - k.re * e, w.im - k.im * e))
        .collect()
}

/// Spatial Gaussian low-pass (Kymatio `gabor_2d(M, N, sigma, 0, 0)`).
#[must_use]
pub fn gaussian_2d_spatial(rows: usize, cols: usize, sigma: f64) -> Vec<Complex64> {
    let (_, env) = gabor_2d_pair(rows, cols, sigma, 0.0, 0.0, 1.0);
    env.into_iter().map(|e| Complex64::new(e, 0.0)).collect()
}

/// Forward host FFT of a spatial filter; returns `(real part, max |imag| /
/// max |real|)`.
fn to_fourier_real(mut z: Vec<Complex64>, rows: usize, cols: usize) -> (Vec<f64>, f64) {
    fft_2d(&mut z, rows, cols, Direction::Forward);
    let max_re = z.iter().fold(0.0f64, |m, v| m.max(v.re.abs()));
    let max_im = z.iter().fold(0.0f64, |m, v| m.max(v.im.abs()));
    (z.into_iter().map(|v| v.re).collect(), max_im / max_re)
}

/// Truncate below `2^-40 * max|h|`, round once to binary32 and flush a
/// subnormal result to signed zero (VDS-1.1 section 8.1; no supported
/// filter has one, the truncation keeps values far above `2^-126`).
#[must_use]
pub fn truncate_and_round(h: &[f64]) -> Vec<f32> {
    let peak = h.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let threshold = peak * pow2_real(f64::from(TRUNCATION_LOG2));
    h.iter()
        .map(|&v| {
            if v.abs() < threshold {
                0.0
            } else {
                vikshep_numerics::flush::ftz(v as f32)
            }
        })
        .collect()
}

fn build_1d(cfg: &ScatterConfig, canvas: Canvas) -> FilterBank {
    let n = canvas.cols;
    let make = |params: Vec<(f64, f64, u32)>| -> Vec<Filter> {
        params
            .into_par_iter()
            .map(|(xi, sigma, j)| Filter {
                j,
                theta: None,
                xi,
                sigma,
                fourier: truncate_and_round(&morlet_1d_f64(n, Some(xi), sigma)),
            })
            .collect()
    };
    let sigma_low = SIGMA0_1D * pow2_neg(cfg.j);
    let phi = Filter {
        j: cfg.j,
        theta: None,
        xi: 0.0,
        sigma: sigma_low,
        fourier: truncate_and_round(&morlet_1d_f64(n, None, sigma_low)),
    };
    FilterBank {
        canvas,
        phi,
        psi1: make(params_1d(cfg.j, cfg.q)),
        psi2: Some(make(params_1d(cfg.j, 1))),
        max_imag_ratio: 0.0,
    }
}

/// 2-D wavelet parameters for `(j, l)`: `(sigma, theta, xi, slant)`.
#[must_use]
pub fn params_2d(j: u32, l: u32, big_l: u32) -> (f64, f64, f64, f64) {
    let scale = pow2_real(f64::from(j));
    let sigma = 0.8 * scale;
    let theta = f64::from(l) * core::f64::consts::PI / f64::from(big_l);
    let xi = 3.0 / 4.0 * core::f64::consts::PI / scale;
    let slant = 4.0 / f64::from(big_l);
    (sigma, theta, xi, slant)
}

fn build_2d(cfg: &ScatterConfig, canvas: Canvas) -> FilterBank {
    let (rows, cols) = (canvas.rows, canvas.cols);
    let specs: Vec<(u32, u32)> = (0..cfg.j)
        .flat_map(|j| (0..cfg.l).map(move |l| (j, l)))
        .collect();
    let built: Vec<(Filter, f64)> = specs
        .into_par_iter()
        .map(|(j, l)| {
            let (sigma, theta, xi, slant) = params_2d(j, l, cfg.l);
            let (re, ratio) = to_fourier_real(
                morlet_2d_spatial(rows, cols, sigma, theta, xi, slant),
                rows,
                cols,
            );
            let f = Filter {
                j,
                theta: Some(l),
                xi,
                sigma,
                fourier: truncate_and_round(&re),
            };
            (f, ratio)
        })
        .collect();
    let sigma_phi = 0.8 * pow2_real(f64::from(cfg.j) - 1.0);
    let (phi_re, phi_ratio) =
        to_fourier_real(gaussian_2d_spatial(rows, cols, sigma_phi), rows, cols);
    let max_imag_ratio = built.iter().fold(phi_ratio, |m, (_, r)| m.max(*r));
    FilterBank {
        canvas,
        phi: Filter {
            j: cfg.j,
            theta: None,
            xi: 0.0,
            sigma: sigma_phi,
            fourier: truncate_and_round(&phi_re),
        },
        psi1: built.into_iter().map(|(f, _)| f).collect(),
        psi2: None,
        max_imag_ratio,
    }
}

type BankKey = (usize, usize, usize, u32, u32, u32);

fn cache() -> &'static Mutex<HashMap<BankKey, Arc<FilterBank>>> {
    static CACHE: OnceLock<Mutex<HashMap<BankKey, Arc<FilterBank>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The filter bank of a (validated) configuration, cached per canvas and
/// parameters. Group, order, shape within the canvas and carrier cutoff do
/// not affect the filters.
#[must_use]
pub fn filter_bank(cfg: &ScatterConfig) -> Arc<FilterBank> {
    let canvas = cfg.canvas();
    let key = (cfg.dim, canvas.rows, canvas.cols, cfg.j, cfg.q, cfg.l);
    if let Some(b) = cache().lock().unwrap().get(&key) {
        return Arc::clone(b);
    }
    let bank = Arc::new(if cfg.dim == 1 {
        build_1d(cfg, canvas)
    } else {
        build_2d(cfg, canvas)
    });
    cache().lock().unwrap().entry(key).or_insert(bank).clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Group, PadPolicy};

    #[test]
    fn kymatio_1d_parameters_q1() {
        // Kymatio 0.3.0 compute_params_filterbank(0.1 / 2**4, 1, 5.0) starts
        // at xi_max = 0.35 and halves xi per filter.
        let p = params_1d(4, 1);
        assert_eq!(p[0].0, 0.35);
        for w in p.windows(2) {
            assert!((w[1].0 - w[0].0 / 2.0).abs() < 1e-15);
        }
        // j assignments are non-decreasing
        assert!(p.windows(2).all(|w| w[0].2 <= w[1].2));
    }

    #[test]
    fn dyadic_subsampling_matches_definition() {
        // ub = 0.5: floor(-log2 0.5) - 1 = 0
        assert_eq!(max_dyadic_subsampling(0.35, 0.03), 0);
        // ub = 0.15: floor(2.737) - 1 = 1
        assert_eq!(max_dyadic_subsampling(0.1, 0.01), 1);
        // ub = 0.125 exactly: floor(3) - 1 = 2
        assert_eq!(max_dyadic_subsampling(0.125, 0.0), 2);
    }

    #[test]
    fn filters_are_cached_and_finite() {
        let c = ScatterConfig::two_d(16, 16, 2, 4, [PadPolicy::Circular; 2], Group::Trivial);
        let a = filter_bank(&c);
        let b = filter_bank(&c);
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(a.psi1.len(), 8);
        assert!(
            a.psi1
                .iter()
                .all(|f| f.fourier.iter().all(|v| v.is_finite()))
        );
        // Zero mean: psi(0) is exactly representable as (near) zero.
        assert!(a.psi1.iter().all(|f| f.fourier[0].abs() < 1e-6));
    }
}
