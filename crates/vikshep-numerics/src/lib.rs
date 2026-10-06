//! Deterministic numerical building blocks shared by every Vikshep crate.
//!
//! * [`fft`]: Stockham FFT and twiddle tables (VDS-1 sections 6 and 7).
//! * [`flush`]: VDS-1.1 software flush-to-zero of binary32 (section 8.1).
//! * [`rng`]: SplitMix64 and Philox4x32-10 streams (VDS-1 section 5).
//! * [`oid`]: canonical tensor hashing (VDS-1 section 9).
//! * [`sum`]: fixed-order (pairwise) summation for host reductions.
//! * [`order`]: total-order sorting with index tie-break.
//! * [`jcs`]: RFC 8785 canonical JSON, including number formatting.
//! * [`provenance`]: provenance manifests and their hash.

pub mod fft;
pub mod flush;
pub mod jcs;
pub mod oid;
pub mod order;
pub mod provenance;
pub mod rng;
pub mod sum;

/// Current numerics version (VDS-1 section 10): 2 since VDS-1.1
/// (flush-to-zero of binary32 Tier-1 operands and results, section 8.1).
pub const NUMERICS_VERSION: u32 = 2;

/// Version of the Tier-2 algorithms of VDS-1 sections 15 to 19 (statistics,
/// training, calibration, anomaly search), versioned independently of the
/// Tier-1 arithmetic (VDS-1 section 10.1).
pub const TIER2_VERSION: u32 = 1;
