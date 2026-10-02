//! Deterministic numerical building blocks shared by every Vikshep crate.
//!
//! * [`rng`]: SplitMix64 and Philox4x32-10 streams (VDS-1 section 5).
//! * [`oid`]: canonical tensor hashing (VDS-1 section 9).
//!
//! The FFT and twiddle-table construction (VDS-1 sections 6 and 7) are added
//! by the next milestone (C1).

pub mod oid;
pub mod rng;

/// Current numerics version (VDS-1 section 10).
pub const NUMERICS_VERSION: u32 = 1;
