//! The scattering cascade and its execution plan (VDS-1 sections 14.4 to
//! 14.6).
//!
//! The host prepares filters and twiddle tables and moves data; the backend
//! executes only the five kernels `fft`, `ifft`, `mul_real_filter`,
//! `modulus` and `subsample`. Pooling, r2 and reductions run on the host.

use std::sync::Arc;

use vikshep_backend_api::{BackendError, Canvas, Complex32, ScatterBackend, Twiddles};
use vikshep_numerics::fft::Real;
use vikshep_numerics::flush::ftz;

use crate::config::{ConfigError, Group, ScatterConfig};
use crate::filters::{FilterBank, filter_bank};

/// One output path (VDS-1 section 14.7).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PathInfo {
    /// Scattering order (0, 1 or 2).
    pub order: u32,
    /// Filter indices into the first-order and second-order banks
    /// (trivial group only; empty for pooled paths).
    pub n: Vec<usize>,
    /// Scales `j1` (and `j2`).
    pub j: Vec<u32>,
    /// Orientations: absolute `l1, l2` (2-D trivial) or the relative
    /// orientation `delta` of a pooled second-order path; empty otherwise.
    pub theta: Vec<u32>,
}

/// Scattering coefficients of a batch.
#[derive(Clone, Debug, PartialEq)]
pub struct ScatterOutput {
    /// Paths in canonical order.
    pub paths: Vec<PathInfo>,
    /// Spatial output shape (`dim` entries).
    pub out_shape: Vec<usize>,
    /// Number of signals.
    pub batch: usize,
    /// `[batch][path][spatial]`, binary32.
    pub coefficients: Vec<f32>,
}

impl ScatterOutput {
    /// Output samples per path.
    #[must_use]
    pub fn out_len(&self) -> usize {
        self.out_shape.iter().product()
    }

    /// Coefficients of signal `b`, path `p`.
    #[must_use]
    pub fn path(&self, b: usize, p: usize) -> &[f32] {
        let n = self.out_len();
        let start = (b * self.paths.len() + p) * n;
        &self.coefficients[start..start + n]
    }

    /// Canonical bytes (little-endian binary32, `[batch][path][spatial]`).
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        self.coefficients
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    }
}

/// A prepared scattering transform: configuration, filters, tables, paths.
#[derive(Clone, Debug)]
pub struct Scattering {
    cfg: ScatterConfig,
    bank: Arc<FilterBank>,
    /// `[phi, psi1..., psi2...]`, one canvas each (2-D: no separate psi2).
    flat: Vec<f32>,
    psi2_offset: u32,
    tw_fwd: (Vec<Complex32>, Vec<Complex32>),
    tw_inv: (Vec<Complex32>, Vec<Complex32>),
    /// Trivial-group paths (the computed set).
    raw_paths: Vec<PathInfo>,
    /// For each first-order filter, the second-order filters it pairs with.
    pairs: Vec<Vec<usize>>,
    /// Output paths (after pooling for `so2_relative`).
    out_paths: Vec<PathInfo>,
}

fn tables(canvas: Canvas, inverse: bool) -> (Vec<Complex32>, Vec<Complex32>) {
    let t = |n: usize| -> Vec<Complex32> {
        if n == 1 {
            Vec::new()
        } else if inverse {
            f32::twiddles_inverse(n.trailing_zeros()).to_vec()
        } else {
            f32::twiddles(n.trailing_zeros()).to_vec()
        }
    };
    (t(canvas.cols), t(canvas.rows))
}

impl Scattering {
    /// Validate the configuration and build (or fetch cached) filters.
    pub fn new(cfg: ScatterConfig) -> Result<Self, ConfigError> {
        cfg.validate()?;
        let bank = filter_bank(&cfg);
        let canvas = cfg.canvas();
        let mut flat = Vec::with_capacity(canvas.len() * (1 + bank.psi1.len() * 2));
        flat.extend_from_slice(&bank.phi.fourier);
        for f in &bank.psi1 {
            flat.extend_from_slice(&f.fourier);
        }
        let psi2_offset = if let Some(p2) = &bank.psi2 {
            let off = (1 + bank.psi1.len()) as u32;
            for f in p2 {
                flat.extend_from_slice(&f.fourier);
            }
            off
        } else {
            1
        };
        let second = bank.second_order();
        let mut raw_paths = vec![PathInfo {
            order: 0,
            n: vec![],
            j: vec![],
            theta: vec![],
        }];
        let theta_of = |t: Option<u32>| t.into_iter().collect::<Vec<_>>();
        for (n1, f1) in bank.psi1.iter().enumerate() {
            raw_paths.push(PathInfo {
                order: 1,
                n: vec![n1],
                j: vec![f1.j],
                theta: theta_of(f1.theta),
            });
        }
        let mut pairs = vec![Vec::new(); bank.psi1.len()];
        if cfg.max_order == 2 {
            for (n1, f1) in bank.psi1.iter().enumerate() {
                for (n2, f2) in second.iter().enumerate() {
                    if f2.j > f1.j {
                        pairs[n1].push(n2);
                        let mut theta = theta_of(f1.theta);
                        theta.extend(theta_of(f2.theta));
                        raw_paths.push(PathInfo {
                            order: 2,
                            n: vec![n1, n2],
                            j: vec![f1.j, f2.j],
                            theta,
                        });
                    }
                }
            }
        }
        let out_paths = match cfg.group {
            Group::Trivial => raw_paths.clone(),
            Group::So2Relative => pooled_paths(&cfg),
        };
        Ok(Self {
            tw_fwd: tables(canvas, false),
            tw_inv: tables(canvas, true),
            cfg,
            bank,
            flat,
            psi2_offset,
            raw_paths,
            pairs,
            out_paths,
        })
    }

    /// The configuration.
    #[must_use]
    pub fn config(&self) -> &ScatterConfig {
        &self.cfg
    }

    /// The filter bank.
    #[must_use]
    pub fn filters(&self) -> &FilterBank {
        &self.bank
    }

    /// Output paths in canonical order.
    #[must_use]
    pub fn paths(&self) -> &[PathInfo] {
        &self.out_paths
    }

    /// Paths computed by the backend (trivial group).
    #[must_use]
    pub fn raw_paths(&self) -> &[PathInfo] {
        &self.raw_paths
    }

    /// Run the cascade on `inputs` (`batch * signal_len` values, row-major
    /// signals back to back).
    pub fn run(
        &self,
        backend: &dyn ScatterBackend,
        inputs: &[f32],
    ) -> Result<ScatterOutput, BackendError> {
        let raw = self.run_raw(backend, inputs)?;
        Ok(match self.cfg.group {
            Group::Trivial => raw,
            Group::So2Relative => self.pool(&raw),
        })
    }

    /// Run the cascade without pooling (trivial-group coefficients).
    pub fn run_raw(
        &self,
        backend: &dyn ScatterBackend,
        inputs: &[f32],
    ) -> Result<ScatterOutput, BackendError> {
        let sig_len = self.cfg.signal_len();
        if !inputs.len().is_multiple_of(sig_len) {
            return Err(BackendError::InvalidArgument(format!(
                "input length {} is not a multiple of the signal length {sig_len}",
                inputs.len()
            )));
        }
        let batch = inputs.len() / sig_len;
        let canvas = self.cfg.canvas();
        let clen = canvas.len();
        let spec = self.cfg.subsample_spec();
        let out_len = spec.out_len();
        let n_paths = self.raw_paths.len();
        let tw = Twiddles {
            cols: &self.tw_fwd.0,
            rows: &self.tw_fwd.1,
        };
        let twi = Twiddles {
            cols: &self.tw_inv.0,
            rows: &self.tw_inv.1,
        };
        let mut coefficients = vec![0.0f32; batch * n_paths * out_len];

        // 1. Embed every signal in its canvas (flushing subnormal samples to
        //    signed zero on the host, VDS-1.1 section 8.1) and transform.
        let mut x = vec![Complex32::default(); batch * clen];
        let axes = self.cfg.axes();
        let (r0, c0, rows_in, cols_in) = if self.cfg.dim == 1 {
            (0, axes[0].offset, 1, axes[0].n)
        } else {
            (axes[0].offset, axes[1].offset, axes[0].n, axes[1].n)
        };
        for b in 0..batch {
            let sig = &inputs[b * sig_len..(b + 1) * sig_len];
            let cv = &mut x[b * clen..(b + 1) * clen];
            for r in 0..rows_in {
                for c in 0..cols_in {
                    cv[(r0 + r) * canvas.cols + c0 + c] =
                        Complex32::new(ftz(sig[r * cols_in + c]), 0.0);
                }
            }
        }
        backend.fft(&mut x, canvas, &tw)?;

        // 2. Order 0 for the whole batch.
        let mut s0 = vec![0.0f32; batch * out_len];
        let mut t = x.clone();
        backend.mul_real_filter(&mut t, canvas, &self.flat, &vec![0; batch])?;
        backend.ifft(&mut t, canvas, &twi)?;
        backend.subsample(&t, canvas, spec, &mut s0)?;
        for b in 0..batch {
            let dst = b * n_paths * out_len;
            coefficients[dst..dst + out_len].copy_from_slice(&s0[b * out_len..(b + 1) * out_len]);
        }

        let n1 = self.bank.psi1.len();
        let mut next_path = vec![0usize; n1];
        {
            // path index of each first-order filter's first order-2 path
            let mut p = 1 + n1;
            for (i, slot) in next_path.iter_mut().enumerate() {
                *slot = p;
                p += self.pairs[i].len();
            }
        }

        // 3. Orders 1 and 2, one signal at a time.
        for b in 0..batch {
            let xb = &x[b * clen..(b + 1) * clen];
            let base = b * n_paths * out_len;
            // U1 = |x * psi_l1| for every first-order filter, then its FFT.
            let mut u1 = Vec::with_capacity(n1 * clen);
            for _ in 0..n1 {
                u1.extend_from_slice(xb);
            }
            let idx1: Vec<u32> = (1..=n1 as u32).collect();
            backend.mul_real_filter(&mut u1, canvas, &self.flat, &idx1)?;
            backend.ifft(&mut u1, canvas, &twi)?;
            backend.modulus(&mut u1)?;
            backend.fft(&mut u1, canvas, &tw)?;
            // S1 = (U1 * phi) subsampled.
            let mut t1 = u1.clone();
            backend.mul_real_filter(&mut t1, canvas, &self.flat, &vec![0; n1])?;
            backend.ifft(&mut t1, canvas, &twi)?;
            let mut s1 = vec![0.0f32; n1 * out_len];
            backend.subsample(&t1, canvas, spec, &mut s1)?;
            coefficients[base + out_len..base + (1 + n1) * out_len].copy_from_slice(&s1);
            drop(t1);
            // S2 for each first-order filter.
            for l1 in 0..n1 {
                let partners = &self.pairs[l1];
                if partners.is_empty() {
                    continue;
                }
                let k = partners.len();
                let src = &u1[l1 * clen..(l1 + 1) * clen];
                let mut z = Vec::with_capacity(k * clen);
                for _ in 0..k {
                    z.extend_from_slice(src);
                }
                let idx2: Vec<u32> = partners
                    .iter()
                    .map(|&n2| self.psi2_offset + n2 as u32)
                    .collect();
                backend.mul_real_filter(&mut z, canvas, &self.flat, &idx2)?;
                backend.ifft(&mut z, canvas, &twi)?;
                backend.modulus(&mut z)?;
                backend.fft(&mut z, canvas, &tw)?;
                backend.mul_real_filter(&mut z, canvas, &self.flat, &vec![0; k])?;
                backend.ifft(&mut z, canvas, &twi)?;
                let mut s2 = vec![0.0f32; k * out_len];
                backend.subsample(&z, canvas, spec, &mut s2)?;
                let p0 = next_path[l1];
                coefficients[base + p0 * out_len..base + (p0 + k) * out_len].copy_from_slice(&s2);
            }
        }
        Ok(ScatterOutput {
            paths: self.raw_paths.clone(),
            out_shape: self.cfg.out_shape(),
            batch,
            coefficients,
        })
    }

    /// `so2_relative` pooling (VDS-1 section 14.5): the mean over absolute
    /// orientation `l1`, per spatial position, computed as a binary64 sum in
    /// increasing `l1`, divided by `L` in binary64, rounded once to binary32.
    #[must_use]
    pub fn pool(&self, raw: &ScatterOutput) -> ScatterOutput {
        let big_l = self.cfg.l as usize;
        let out_len = raw.out_len();
        let n_out = self.out_paths.len();
        // raw path index of (order 1, j, l) and (order 2, j1, l1, j2, l2)
        let p1 = |j: usize, l: usize| 1 + j * big_l + l;
        let mut p2 = std::collections::HashMap::new();
        for (i, p) in raw.paths.iter().enumerate() {
            if p.order == 2 {
                p2.insert((p.j[0], p.theta[0], p.j[1], p.theta[1]), i);
            }
        }
        let mut coefficients = vec![0.0f32; raw.batch * n_out * out_len];
        let mean = |sources: &[usize], b: usize, dst: &mut [f32]| {
            for (s, d) in dst.iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for &p in sources {
                    acc += f64::from(raw.path(b, p)[s]);
                }
                *d = (acc / big_l as f64) as f32;
            }
        };
        for b in 0..raw.batch {
            let base = b * n_out * out_len;
            coefficients[base..base + out_len].copy_from_slice(raw.path(b, 0));
            for (o, path) in self.out_paths.iter().enumerate().skip(1) {
                let sources: Vec<usize> = if path.order == 1 {
                    (0..big_l).map(|l| p1(path.j[0] as usize, l)).collect()
                } else {
                    let delta = path.theta[0] as usize;
                    (0..big_l)
                        .map(|l1| {
                            let l2 = (l1 + delta) % big_l;
                            p2[&(path.j[0], l1 as u32, path.j[1], l2 as u32)]
                        })
                        .collect()
                };
                let dst = &mut coefficients[base + o * out_len..base + (o + 1) * out_len];
                mean(&sources, b, dst);
            }
        }
        ScatterOutput {
            paths: self.out_paths.clone(),
            out_shape: raw.out_shape.clone(),
            batch: raw.batch,
            coefficients,
        }
    }
}

fn pooled_paths(cfg: &ScatterConfig) -> Vec<PathInfo> {
    let mut v = vec![PathInfo {
        order: 0,
        n: vec![],
        j: vec![],
        theta: vec![],
    }];
    for j1 in 0..cfg.j {
        v.push(PathInfo {
            order: 1,
            n: vec![],
            j: vec![j1],
            theta: vec![],
        });
    }
    if cfg.max_order == 2 {
        for j1 in 0..cfg.j {
            for j2 in (j1 + 1)..cfg.j {
                for delta in 0..cfg.l {
                    v.push(PathInfo {
                        order: 2,
                        n: vec![],
                        j: vec![j1, j2],
                        theta: vec![delta],
                    });
                }
            }
        }
    }
    v
}
