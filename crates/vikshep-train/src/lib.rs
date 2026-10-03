//! Deterministic Tier-2 training on frozen features (`spec/VDS-1.md`
//! sections 17 and 18): logistic and one-hidden-layer MLP tagger heads
//! trained with the DisCo objective and Adam, the lambda sweep and benchmark
//! report, ridge calibration with a deterministic Cholesky factorization,
//! and the synthetic benchmark generator.

use core::fmt;

pub mod calibrate;
pub mod data;
pub mod model;
pub mod sweep;
pub mod train;

/// A training error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainError(pub String);

impl TrainError {
    pub(crate) fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl fmt::Display for TrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TrainError {}
