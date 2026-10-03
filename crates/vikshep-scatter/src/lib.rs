//! Wavelet scattering transform: Morlet filter banks, the order-0/1/2
//! cascade, orientation pooling and host reductions (`spec/VDS-1.md`
//! section 14).
//!
//! ```text
//! S0 x         = (x * phi_J) subsampled by 2^J
//! S1[l1] x     = (|x * psi_l1| * phi_J) subsampled
//! S2[l1, l2] x = (||x * psi_l1| * psi_l2| * phi_J) subsampled, j2 > j1
//! ```
//!
//! The arithmetic of S0, S1 and S2 is Tier 1 (executed by a
//! [`vikshep_backend_api::ScatterBackend`]); filters, pooling, r2 and the
//! log-mean fingerprint are Tier 2 (this crate, on the host).

pub mod cascade;
pub mod config;
pub mod filters;
pub mod manifest;
pub mod reduce;

pub use cascade::{PathInfo, ScatterOutput, Scattering};
pub use config::{Group, PadPolicy, ScatterConfig};
