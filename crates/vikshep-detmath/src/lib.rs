//! Portable, platform-independent transcendental functions for Vikshep.
//!
//! Every transcendental function used anywhere in Tier-2 code (see
//! `spec/VDS-1.md`, section 4) MUST come from this crate and never from the
//! `std` float methods, which lower to the platform C library and differ
//! between operating systems.
//!
//! # Implementation (decision D-MATH, Option A)
//!
//! The functions wrap the pure-Rust [`libm`] crate pinned to exactly
//! `=0.2.16` with `default-features = false`. With the `arch` feature off,
//! `libm` uses no architecture-specific code paths on any supported target:
//! every function is a sequence of IEEE-754 binary64/binary32 `+ - * /`,
//! comparisons and integer bit manipulation, all of which Rust evaluates
//! with round-to-nearest-even and without contraction. The results are
//! therefore identical on every conforming platform. This is checked by the
//! cross-platform determinism sweep (`cargo run -p vikshep-conformance --
//! hashes`) in CI, not merely asserted.
//!
//! The only target where `libm` would select a hardware path regardless of
//! features is 32-bit x86 without SSE2 (x87); that target is not supported.
//!
//! # Single-precision variants
//!
//! The `*_f32` functions evaluate the binary64 function on the exactly
//! widened argument and round the result once to binary32
//! (round-to-nearest-even). They are not guaranteed to be correctly
//! rounded; the measured error is recorded in `spec/VDS-1.md`.
//!
//! # Accuracy
//!
//! Maximum observed error against correctly rounded `mpmath` references
//! (fixture: `tests/fixtures/mpmath_reference.txt`, generator:
//! `scripts/gen_detmath_fixtures.py`) is asserted exactly by
//! `tests/accuracy.rs` and recorded in `spec/VDS-1.md` section 4.

#![no_std]

/// The `libm` version this crate is pinned to. Part of the numerics contract.
pub const LIBM_VERSION: &str = "0.2.16";

/// e^x in binary64.
#[inline]
#[must_use]
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

/// Natural logarithm in binary64.
#[inline]
#[must_use]
pub fn ln(x: f64) -> f64 {
    libm::log(x)
}

/// Sine (argument in radians) in binary64.
#[inline]
#[must_use]
pub fn sin(x: f64) -> f64 {
    libm::sin(x)
}

/// Cosine (argument in radians) in binary64.
#[inline]
#[must_use]
pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}

/// e^x in binary32: evaluated in binary64, rounded once to binary32.
#[inline]
#[must_use]
pub fn exp_f32(x: f32) -> f32 {
    exp(f64::from(x)) as f32
}

/// Natural logarithm in binary32: evaluated in binary64, rounded once.
#[inline]
#[must_use]
pub fn ln_f32(x: f32) -> f32 {
    ln(f64::from(x)) as f32
}

/// Sine in binary32: evaluated in binary64, rounded once.
#[inline]
#[must_use]
pub fn sin_f32(x: f32) -> f32 {
    sin(f64::from(x)) as f32
}

/// Cosine in binary32: evaluated in binary64, rounded once.
#[inline]
#[must_use]
pub fn cos_f32(x: f32) -> f32 {
    cos(f64::from(x)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_special_values() {
        assert_eq!(exp(0.0).to_bits(), 1.0f64.to_bits());
        assert_eq!(ln(1.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(sin(0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(sin(-0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(cos(0.0).to_bits(), 1.0f64.to_bits());
        assert_eq!(exp(f64::NEG_INFINITY).to_bits(), 0.0f64.to_bits());
        assert_eq!(exp(f64::INFINITY), f64::INFINITY);
        assert_eq!(ln(0.0), f64::NEG_INFINITY);
        assert!(ln(-1.0).is_nan());
        assert_eq!(exp_f32(0.0).to_bits(), 1.0f32.to_bits());
        assert_eq!(ln_f32(1.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(cos_f32(0.0).to_bits(), 1.0f32.to_bits());
        assert_eq!(sin_f32(-0.0).to_bits(), (-0.0f32).to_bits());
    }
}
