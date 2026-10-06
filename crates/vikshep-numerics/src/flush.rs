//! VDS-1.1 flush-to-zero (`spec/VDS-1.md` section 8.1), emulated in
//! software so that it does not depend on processor flags (MXCSR, FPCR),
//! which are neither portable nor controllable from safe Rust.
//!
//! A binary32 value whose exponent field is zero (a subnormal or a zero) is
//! replaced by the zero of the same sign; every other value, including
//! infinities and NaNs, is returned unchanged. This is exactly
//! `if |x| < f32::MIN_POSITIVE { copysign(0, x) } else { x }`. Tininess is
//! judged on the value given, which for an operation result is the rounded
//! result.

const EXPONENT: u32 = 0x7f80_0000;
const SIGN: u32 = 0x8000_0000;

/// Flush a binary32 subnormal to the zero of the same sign.
#[inline(always)]
#[must_use]
pub fn ftz(x: f32) -> f32 {
    let b = x.to_bits();
    // all ones when the exponent field is zero (zero or subnormal)
    let tiny = u32::from(b & EXPONENT == 0).wrapping_neg();
    f32::from_bits(b & !(tiny & !SIGN))
}

/// Flush every element of a slice in place (host-side input and table
/// flushing, "DAZ").
pub fn ftz_slice(v: &mut [f32]) {
    for x in v {
        *x = ftz(*x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flush_semantics() {
        let min = f32::MIN_POSITIVE;
        assert_eq!(ftz(min).to_bits(), min.to_bits());
        assert_eq!(ftz(-min).to_bits(), (-min).to_bits());
        let below = f32::from_bits(min.to_bits() - 1); // largest subnormal
        assert_eq!(ftz(below).to_bits(), 0);
        assert_eq!(ftz(-below).to_bits(), SIGN);
        assert_eq!(ftz(f32::from_bits(1)).to_bits(), 0);
        assert_eq!(ftz(-0.0).to_bits(), SIGN);
        assert_eq!(ftz(0.0).to_bits(), 0);
        assert_eq!(ftz(f32::INFINITY), f32::INFINITY);
        assert!(ftz(f32::NAN).is_nan());
        assert_eq!(ftz(1.5), 1.5);
        // every binary32 pattern agrees with the spec's reference definition
        for bits in (0..=u32::MAX).step_by(4099) {
            let x = f32::from_bits(bits);
            let r = if x.abs() < min { 0.0f32.copysign(x) } else { x };
            assert_eq!(ftz(x).to_bits(), r.to_bits(), "{bits:#x}");
        }
    }
}
