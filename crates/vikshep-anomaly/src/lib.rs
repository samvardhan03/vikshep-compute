//! Deterministic Tier-2 anomaly search (`spec/VDS-1.md` section 19):
//! fingerprint distributions from C1 scattering outputs, the Sliced
//! Wasserstein-1 distance, a deterministic HNSW index, k-th-neighbour
//! flagging, a fixed-iteration spring layout and the graph output contract.

use core::fmt;

pub mod graph;
pub mod hnsw;
pub mod sw1;

/// An anomaly-search error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnomalyError(pub String);

impl AnomalyError {
    pub(crate) fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl fmt::Display for AnomalyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AnomalyError {}
