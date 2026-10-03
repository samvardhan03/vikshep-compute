//! Tagger heads, activations and the Adam optimizer (`spec/VDS-1.md`
//! sections 17.2, 17.3 and 17.5).

use vikshep_numerics::rng::Stream;

/// Head architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadKind {
    /// `z = b + sum_j theta_j x_j`.
    Logistic,
    /// One hidden layer of `hidden` tanh units:
    /// `z = b2 + sum_h W2_h tanh(b1_h + sum_j W1_hj x_j)`.
    Mlp {
        /// Hidden width.
        hidden: usize,
    },
}

impl HeadKind {
    /// Canonical name.
    #[must_use]
    pub fn name(self) -> String {
        match self {
            Self::Logistic => "logistic".into(),
            Self::Mlp { hidden } => format!("mlp{hidden}"),
        }
    }
}

/// Philox stream of parameter initialization (VDS-1 section 15.4).
pub const STREAM_INIT: u64 = 0x3_0000;

/// `1 / (1 + exp(-z))`, evaluated as `e / (1 + e)` with `e = exp(z)` for
/// `z < 0` (no overflow).
#[must_use]
pub fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + vikshep_detmath::exp(-z))
    } else {
        let e = vikshep_detmath::exp(z);
        e / (1.0 + e)
    }
}

/// `tanh(a) = sign(a) (1 - e) / (1 + e)`, `e = exp(-2 |a|)`.
#[must_use]
pub fn tanh(a: f64) -> f64 {
    let e = vikshep_detmath::exp(-2.0 * a.abs());
    let t = (1.0 - e) / (1.0 + e);
    if a < 0.0 { -t } else { t }
}

/// `ln(1 + exp(z))` as `max(z, 0) + ln(1 + exp(-|z|))`.
#[must_use]
pub fn softplus(z: f64) -> f64 {
    z.max(0.0) + vikshep_detmath::ln(1.0 + vikshep_detmath::exp(-z.abs()))
}

/// Sequential dot product (increasing index).
#[must_use]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0, |acc, (x, y)| acc + x * y)
}

/// Parameters of a head, flattened in a fixed order.
///
/// * Logistic: `[b, theta_0 .. theta_{d-1}]`.
/// * MLP: `[W1 (hidden x d, row-major), b1 (hidden), W2 (hidden), b2]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Head {
    /// Architecture.
    pub kind: HeadKind,
    /// Input features.
    pub d: usize,
    /// Flattened parameters.
    pub params: Vec<f64>,
}

impl Head {
    /// Initialize from `Stream(seed, STREAM_INIT)`: logistic weights
    /// `0.01 z`, bias 0; MLP `W1 = sqrt(1/d) z` (row-major), `b1 = 0`,
    /// `W2 = sqrt(1/hidden) z`, `b2 = 0` (`z` standard normal, drawn in
    /// parameter order).
    #[must_use]
    pub fn init(kind: HeadKind, d: usize, seed: u64) -> Self {
        let mut s = Stream::new(seed, STREAM_INIT);
        let params = match kind {
            HeadKind::Logistic => {
                let mut p = vec![0.0];
                p.extend((0..d).map(|_| 0.01 * s.next_normal_f64()));
                p
            }
            HeadKind::Mlp { hidden } => {
                let s1 = (1.0 / d as f64).sqrt();
                let s2 = (1.0 / hidden as f64).sqrt();
                let mut p: Vec<f64> = (0..hidden * d).map(|_| s1 * s.next_normal_f64()).collect();
                p.extend(std::iter::repeat_n(0.0, hidden));
                p.extend((0..hidden).map(|_| s2 * s.next_normal_f64()));
                p.push(0.0);
                p
            }
        };
        Self { kind, d, params }
    }

    /// Number of parameters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// True if there are no parameters (never for a valid head).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Logit for one standardized row.
    #[must_use]
    pub fn logit(&self, x: &[f64]) -> f64 {
        match self.kind {
            HeadKind::Logistic => self.params[0] + dot(&self.params[1..], x),
            HeadKind::Mlp { hidden } => {
                let d = self.d;
                let (w1, rest) = self.params.split_at(hidden * d);
                let (b1, rest) = rest.split_at(hidden);
                let (w2, b2) = rest.split_at(hidden);
                let mut z = b2[0];
                for h in 0..hidden {
                    let a = b1[h] + dot(&w1[h * d..(h + 1) * d], x);
                    z += w2[h] * tanh(a);
                }
                z
            }
        }
    }

    /// `dz/dparams` for one standardized row, written into `out`.
    pub fn logit_grad(&self, x: &[f64], out: &mut [f64]) {
        match self.kind {
            HeadKind::Logistic => {
                out[0] = 1.0;
                out[1..].copy_from_slice(x);
            }
            HeadKind::Mlp { hidden } => {
                let d = self.d;
                let (w1, rest) = self.params.split_at(hidden * d);
                let (b1, rest) = rest.split_at(hidden);
                let w2 = &rest[..hidden];
                for h in 0..hidden {
                    let t = tanh(b1[h] + dot(&w1[h * d..(h + 1) * d], x));
                    let back = w2[h] * (1.0 - t * t);
                    for j in 0..d {
                        out[h * d + j] = back * x[j];
                    }
                    out[hidden * d + h] = back;
                    out[hidden * d + hidden + h] = t;
                }
                out[hidden * d + 2 * hidden] = 1.0;
            }
        }
    }

    /// Canonical bytes (binary64 LE in parameter order).
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        self.params.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

/// Adam constants (VDS-1 section 17.5).
pub const ADAM_BETA1: f64 = 0.9;
/// Adam constants.
pub const ADAM_BETA2: f64 = 0.999;
/// Adam constants.
pub const ADAM_EPS: f64 = 1e-8;

/// Adam optimizer state. Bias corrections use running products of the betas
/// (no `powf`).
#[derive(Clone, Debug)]
pub struct Adam {
    lr: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    beta1_t: f64,
    beta2_t: f64,
}

impl Adam {
    /// Zero state for `n` parameters.
    #[must_use]
    pub fn new(n: usize, lr: f64) -> Self {
        Self {
            lr,
            m: vec![0.0; n],
            v: vec![0.0; n],
            beta1_t: 1.0,
            beta2_t: 1.0,
        }
    }

    /// One step, per parameter `k`:
    /// `m = b1 m + (1 - b1) g`; `v = b2 v + (1 - b2) g g`;
    /// `p -= lr (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + eps)`.
    pub fn step(&mut self, params: &mut [f64], grad: &[f64]) {
        self.beta1_t *= ADAM_BETA1;
        self.beta2_t *= ADAM_BETA2;
        let c1 = 1.0 - self.beta1_t;
        let c2 = 1.0 - self.beta2_t;
        for k in 0..params.len() {
            let g = grad[k];
            self.m[k] = ADAM_BETA1 * self.m[k] + (1.0 - ADAM_BETA1) * g;
            self.v[k] = ADAM_BETA2 * self.v[k] + (1.0 - ADAM_BETA2) * (g * g);
            let mhat = self.m[k] / c1;
            let vhat = self.v[k] / c2;
            params[k] -= self.lr * mhat / (vhat.sqrt() + ADAM_EPS);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activations() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert!((sigmoid(800.0) - 1.0).abs() < 1e-300 + f64::EPSILON);
        assert!(sigmoid(-800.0) >= 0.0);
        assert!((sigmoid(2.0) + sigmoid(-2.0) - 1.0).abs() < 4e-16);
        assert_eq!(tanh(0.0), 0.0);
        assert_eq!(tanh(-1.5), -tanh(1.5));
        assert!((tanh(1.0) - 0.761_594_155_955_764_9).abs() < 1e-15);
        assert!((softplus(0.0) - core::f64::consts::LN_2).abs() < 1e-15);
        assert!((softplus(50.0) - 50.0).abs() < 1e-15);
    }

    #[test]
    fn mlp_gradient_matches_finite_differences() {
        let head = Head::init(HeadKind::Mlp { hidden: 5 }, 3, 9);
        let x = [0.3, -1.2, 0.7];
        let mut g = vec![0.0; head.len()];
        head.logit_grad(&x, &mut g);
        for (k, gk) in g.iter().enumerate() {
            let mut hp = head.clone();
            let mut hm = head.clone();
            hp.params[k] += 1e-6;
            hm.params[k] -= 1e-6;
            let fd = (hp.logit(&x) - hm.logit(&x)) / 2e-6;
            assert!((fd - gk).abs() < 1e-8, "param {k}: {fd} vs {gk}");
        }
    }
}
