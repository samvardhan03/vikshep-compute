//! DisCo training of a tagger head on frozen features (`spec/VDS-1.md`
//! section 17.4):
//!
//! ```text
//! L = wBCE(yhat, y) + lambda * dCorr2_w(yhat, m | background)
//! ```
//!
//! per minibatch, with the exact analytic gradient of the dependence term
//! (or, explicitly requested, the Pearson-proxy fast mode).

use vikshep_numerics::oid::oid;
use vikshep_numerics::rng::Stream;
use vikshep_numerics::sum::pairwise_sum;
use vikshep_stats::dcorr::{dcorr2_grad, pearson_proxy_grad};

use crate::TrainError;
use crate::data::{Dataset, Standardizer};
use crate::model::{Adam, Head, HeadKind, sigmoid, softplus};

/// Gradient of the DisCo term.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradientMode {
    /// Exact analytic gradient of dCorr2 (`docs/disco_gradient.md`).
    Exact,
    /// FAST MODE: gradient of the squared weighted Pearson correlation, as
    /// in the public Vikshep CLI. Not the gradient of dCorr2.
    PearsonProxy,
}

impl GradientMode {
    /// Canonical name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::PearsonProxy => "pearson_proxy",
        }
    }
}

/// Philox stream base of the per-epoch shuffles (`base + epoch`).
pub const STREAM_SHUFFLE: u64 = 0x3_1000_0000;

/// Training configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainConfig {
    /// Head architecture.
    pub head: HeadKind,
    /// Passes over the data.
    pub epochs: usize,
    /// Minibatch size (the last batch of an epoch may be smaller).
    pub batch_size: usize,
    /// Adam learning rate.
    pub lr: f64,
    /// DisCo strength.
    pub lambda: f64,
    /// Gradient of the DisCo term.
    pub gradient: GradientMode,
    /// Master seed (initialization and shuffles).
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            head: HeadKind::Logistic,
            epochs: 20,
            batch_size: 256,
            lr: 0.01,
            lambda: 0.0,
            gradient: GradientMode::Exact,
            seed: 0,
        }
    }
}

/// A trained head with its standardization and loss history.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainedModel {
    /// Configuration used.
    pub config: TrainConfig,
    /// Feature standardization fitted on the training set.
    pub standardizer: Standardizer,
    /// Final parameters.
    pub head: Head,
    /// Per epoch: pairwise mean of the minibatch losses.
    pub epoch_loss: Vec<f64>,
}

impl TrainedModel {
    /// Scores `sigmoid(z)` for every row of `x` (`n x d`, raw features).
    #[must_use]
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        x.chunks_exact(self.head.d)
            .map(|r| sigmoid(self.head.logit(&self.standardizer.apply_row(r))))
            .collect()
    }

    /// Canonical bytes: standardizer `mu`, `std`, then parameters, then the
    /// epoch losses (all binary64 LE).
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        for v in self
            .standardizer
            .mu
            .iter()
            .chain(&self.standardizer.std)
            .chain(&self.head.params)
            .chain(&self.epoch_loss)
        {
            b.extend(v.to_le_bytes());
        }
        b
    }

    /// OID of [`Self::canonical_bytes`].
    #[must_use]
    pub fn oid(&self) -> String {
        oid(&self.canonical_bytes())
    }
}

/// Loss and parameter gradient of one minibatch.
fn batch_step(
    head: &Head,
    cfg: &TrainConfig,
    xs: &[f64],
    batch: &[usize],
    data: &Dataset,
    grad: &mut [f64],
) -> Result<f64, TrainError> {
    let d = data.d;
    let k = batch.len();
    let z: Vec<f64> = batch
        .iter()
        .map(|&i| head.logit(&xs[i * d..(i + 1) * d]))
        .collect();
    let yhat: Vec<f64> = z.iter().map(|&v| sigmoid(v)).collect();
    let wb: Vec<f64> = batch.iter().map(|&i| data.w[i]).collect();
    let wsum = pairwise_sum(&wb);
    if wsum.is_nan() || wsum <= 0.0 {
        grad.iter_mut().for_each(|g| *g = 0.0);
        return Ok(0.0);
    }
    // wBCE with weights normalized within the batch, on logits.
    let mut dz: Vec<f64> = Vec::with_capacity(k);
    let mut bce: Vec<f64> = Vec::with_capacity(k);
    for (t, &i) in batch.iter().enumerate() {
        let v = wb[t] / wsum;
        let yi = f64::from(data.y[i]);
        bce.push(v * (softplus(z[t]) - yi * z[t]));
        dz.push(v * (yhat[t] - yi));
    }
    let mut loss = pairwise_sum(&bce);
    // DisCo term on the background events of the batch.
    let bkg: Vec<usize> = (0..k).filter(|&t| data.y[batch[t]] == 0).collect();
    if cfg.lambda != 0.0 && bkg.len() >= 2 {
        let s: Vec<f64> = bkg.iter().map(|&t| yhat[t]).collect();
        let m: Vec<f64> = bkg.iter().map(|&t| data.m[batch[t]]).collect();
        let w: Vec<f64> = bkg.iter().map(|&t| wb[t]).collect();
        let (value, g) = match cfg.gradient {
            GradientMode::Exact => {
                dcorr2_grad(&s, &m, Some(&w)).map_err(|e| TrainError::new(e.0))?
            }
            GradientMode::PearsonProxy => {
                let g = pearson_proxy_grad(&s, &m, &w).map_err(|e| TrainError::new(e.0))?;
                (0.0, g)
            }
        };
        loss += cfg.lambda * value;
        for (q, &t) in bkg.iter().enumerate() {
            dz[t] += cfg.lambda * g[q] * yhat[t] * (1.0 - yhat[t]);
        }
    }
    // dL/dparams = psum over batch events of dz_t * dz_t/dparams.
    let np = head.len();
    let mut per_event = vec![0.0f64; k * np];
    let mut buf = vec![0.0f64; np];
    for (t, &i) in batch.iter().enumerate() {
        head.logit_grad(&xs[i * d..(i + 1) * d], &mut buf);
        for p in 0..np {
            per_event[t * np + p] = dz[t] * buf[p];
        }
    }
    let mut col = vec![0.0f64; k];
    for (p, g) in grad.iter_mut().enumerate() {
        for t in 0..k {
            col[t] = per_event[t * np + p];
        }
        *g = pairwise_sum(&col);
    }
    Ok(loss)
}

/// Train on `data` (VDS-1 section 17.6): standardize, initialize, then for
/// each epoch `e` permute the events with `Stream(seed, STREAM_SHUFFLE + e)`
/// and take consecutive minibatches, one Adam step each.
pub fn train(data: &Dataset, cfg: TrainConfig) -> Result<TrainedModel, TrainError> {
    if cfg.batch_size == 0 || cfg.epochs == 0 {
        return Err(TrainError::new("epochs and batch_size must be positive"));
    }
    if cfg.lr.is_nan() || cfg.lr <= 0.0 || !cfg.lambda.is_finite() || cfg.lambda < 0.0 {
        return Err(TrainError::new(
            "lr must be positive and lambda finite and non-negative",
        ));
    }
    if let HeadKind::Mlp { hidden } = cfg.head
        && hidden == 0
    {
        return Err(TrainError::new("hidden width must be positive"));
    }
    let standardizer = Standardizer::fit(&data.x, data.n, data.d);
    let xs = standardizer.apply(&data.x, data.d);
    let mut head = Head::init(cfg.head, data.d, cfg.seed);
    let mut adam = Adam::new(head.len(), cfg.lr);
    let mut grad = vec![0.0f64; head.len()];
    let mut epoch_loss = Vec::with_capacity(cfg.epochs);
    for e in 0..cfg.epochs {
        let perm = Stream::new(cfg.seed, STREAM_SHUFFLE + e as u64).permutation(data.n);
        let mut losses = Vec::new();
        for batch in perm.chunks(cfg.batch_size) {
            let loss = batch_step(&head, &cfg, &xs, batch, data, &mut grad)?;
            adam.step(&mut head.params, &grad);
            losses.push(loss);
        }
        epoch_loss.push(pairwise_sum(&losses) / losses.len() as f64);
    }
    Ok(TrainedModel {
        config: cfg,
        standardizer,
        head,
        epoch_loss,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::synthetic;

    #[test]
    fn training_is_reproducible_and_learns() {
        let data = synthetic(800, 4, 3, 0).unwrap();
        let cfg = TrainConfig {
            epochs: 6,
            batch_size: 128,
            ..TrainConfig::default()
        };
        let a = train(&data, cfg).unwrap();
        let b = train(&data, cfg).unwrap();
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
        assert!(a.epoch_loss.last().unwrap() < &a.epoch_loss[0]);
        let mlp = train(
            &data,
            TrainConfig {
                head: HeadKind::Mlp { hidden: 8 },
                lambda: 2.0,
                ..cfg
            },
        )
        .unwrap();
        assert_eq!(mlp.head.len(), 8 * 4 + 8 + 8 + 1);
        assert!(mlp.predict(&data.x).iter().all(|p| (0.0..=1.0).contains(p)));
    }
}
