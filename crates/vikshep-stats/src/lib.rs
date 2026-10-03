//! Deterministic Tier-2 statistics (`spec/VDS-1.md` sections 15 and 16):
//! weighted distance correlation (exact, chunked, exact gradient, and the
//! labelled Pearson-proxy fast mode), Jensen-Shannon divergence, cut
//! selection, the Asimov significance proxy, lambda-frontier rows and the
//! benchmark report.
//!
//! Every sum over events is a pairwise tree, randomness comes only from
//! documented Philox streams, sorting uses a total order with index
//! tie-break, and transcendental functions come from `vikshep-detmath`.

use core::fmt;

pub mod dcorr;
pub mod divergence;
pub mod report;
pub mod significance;

/// A statistics error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatsError(pub String);

impl StatsError {
    pub(crate) fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl fmt::Display for StatsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for StatsError {}
