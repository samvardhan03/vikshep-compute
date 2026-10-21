//! VDS-1.1/1.2 flush-to-zero and per-multiplication tininess
//! (`spec/VDS-1.md` sections 8.1 and 8.3), emulated in software so that it
//! does not depend on processor flags (MXCSR, FPCR), which are neither
//! portable nor controllable from safe Rust.
//!
//! A binary32 value whose exponent field is zero (a subnormal or a zero) is
//! replaced by the zero of the same sign; every other value, including
//! infinities and NaNs, is returned unchanged. This is exactly
//! `if |x| < f32::MIN_POSITIVE { copysign(0, x) } else { x }`. Under
//! VDS-1.2 every binary32 Tier-1 multiplication uses [`mul_tier1`], which
//! judges tininess on the exact product before rounding.

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

/// VDS-1.2 multiply: tininess judged on the exact product before rounding
/// (`spec/VDS-1.md` section 8.3).  For two flushed (post-DAZ) binary32
/// inputs `a` and `b`, the exact product `(a as f64) * (b as f64)` is exact
/// (two 24-bit significands fit in binary64's 53-bit mantissa).  If the
/// exact product is strictly below FLT_MIN (`f32::MIN_POSITIVE`, `2^-126`),
/// the result is the signed zero with the same sign as the binary32 product.
/// Otherwise the correctly rounded binary32 product is returned unchanged,
/// applying the VDS-1.1 flush to any remaining subnormal (for normal
/// products, `ftz` is the identity).
#[inline(always)]
#[must_use]
pub fn mul_tier1(a: f32, b: f32) -> f32 {
    let r = a * b;
    if r.abs() <= f32::MIN_POSITIVE && r != 0.0 {
        let exact = (a as f64) * (b as f64);
        if exact.abs() < f32::MIN_POSITIVE as f64 {
            0.0f32.copysign(r)
        } else {
            r
        }
    } else {
        ftz(r)
    }
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

    #[test]
    fn mul_tier1_boundary() {
        let min = f32::MIN_POSITIVE;
        let min64 = min as f64;
        // exact product >= FLT_MIN: fl32 result kept
        let at = f32::from_bits(0x2000_0000); // 2^-63; 2^-63 * 2^-63 = FLT_MIN exactly
        let r = mul_tier1(at, at);
        assert_eq!(r.to_bits(), min.to_bits(), "exact == FLT_MIN");
        // exact product just above FLT_MIN: fl32 result kept
        let above = f32::from_bits(at.to_bits() + 1);
        let r2 = mul_tier1(above, above);
        assert!(r2 >= min, "exact > FLT_MIN");
        // round_to_flt_min: exact product just below FLT_MIN, fl32 rounds to FLT_MIN
        // data = FLT_MIN, filter = 1 - 2^-24 = 0x3f7fffff
        let data = min;
        let filt = f32::from_bits(0x3f7f_ffff); // 1 - 2^-24
        let exact = (data as f64) * (filt as f64);
        assert!(exact < min64, "exact product below FLT_MIN");
        assert_eq!(
            (data * filt).to_bits(),
            min.to_bits(),
            "fl32 rounds up to FLT_MIN"
        );
        let result = mul_tier1(data, filt);
        assert_eq!(result.to_bits(), 0, "before-rounding: flush to +0");
        // negative sign preserved
        let result_neg = mul_tier1(-data, filt);
        assert_eq!(
            result_neg.to_bits(),
            0x8000_0000u32,
            "before-rounding: flush to -0"
        );
        // subnormal product: both VDS-1.1 and VDS-1.2 flush to zero
        let tiny = f32::from_bits(1); // smallest positive subnormal
        assert_eq!(mul_tier1(min, tiny).to_bits(), 0, "product is subnormal");
        // zero inputs
        assert_eq!(mul_tier1(0.0, min).to_bits(), 0);
        assert_eq!(mul_tier1(min, 0.0).to_bits(), 0);
        // large product: normal result unchanged
        assert_eq!(mul_tier1(2.0, 3.0), 6.0);
    }
}
