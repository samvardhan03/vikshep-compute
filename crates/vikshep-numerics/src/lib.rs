//! Deterministic numerical building blocks shared by every Vikshep crate.
//!
//! * [`fft`]: Stockham FFT and twiddle tables (VDS-1 sections 6 and 7).
//! * [`rng`]: SplitMix64 and Philox4x32-10 streams (VDS-1 section 5).
//! * [`oid`]: canonical tensor hashing (VDS-1 section 9).
//! * [`sum`]: fixed-order (pairwise) summation for host reductions.

pub mod fft;
pub mod oid;
pub mod rng;
pub mod sum;

/// Current numerics version (VDS-1 section 10).
pub const NUMERICS_VERSION: u32 = 1;
