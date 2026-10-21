//! VDS-1.2 property tests for `mul_tier1` (spec/VDS-1.md section 8.3).
//!
//! The function must agree with the direct rule: if the exact f64 product is
//! strictly below FLT_MIN return signed zero; otherwise return the binary32
//! product unchanged.

use vikshep_numerics::flush::mul_tier1;
use vikshep_numerics::rng::Stream;

const FLT_MIN: f64 = f32::MIN_POSITIVE as f64;

fn reference(a: f32, b: f32) -> u32 {
    let r = a * b;
    let exact = (a as f64) * (b as f64);
    if exact.abs() < FLT_MIN {
        0.0f32.copysign(r).to_bits()
    } else {
        r.to_bits()
    }
}

/// 10^7 Philox-generated pairs whose products land in [2^-127, 2^-125].
/// a ∈ [2^-63, 2^-62), b ∈ [2^-64, 2^-63): exact product ∈ [2^-127, 2^-125).
#[test]
fn property_near_flt_min() {
    let mut s = Stream::new(0x5644_5332, 0); // "VDS2"
    let scale_a = f32::from_bits(0x2000_0000u32); // 2^-63
    let scale_b = f32::from_bits(0x1f80_0000u32); // 2^-64
    for _ in 0..10_000_000u64 {
        let a = (1.0 + s.next_f32_unit()) * scale_a;
        let b = (1.0 + s.next_f32_unit()) * scale_b;
        assert_eq!(mul_tier1(a, b).to_bits(), reference(a, b), "a={a} b={b}");
        // negative signs
        assert_eq!(
            mul_tier1(-a, b).to_bits(),
            reference(-a, b),
            "a={a} b={b} negative a"
        );
    }
}

/// Explicit boundary cases: exactly at, just below and just above FLT_MIN.
#[test]
fn explicit_boundary() {
    let min = f32::MIN_POSITIVE;
    let at = f32::from_bits(0x2000_0000u32); // 2^-63; square = FLT_MIN exactly

    // exact == FLT_MIN: result is FLT_MIN (not flushed)
    assert_eq!(
        mul_tier1(at, at).to_bits(),
        min.to_bits(),
        "exact == FLT_MIN"
    );

    // exact just above FLT_MIN: fl32 > FLT_MIN, result normal
    let above = f32::from_bits(at.to_bits() + 1);
    assert!(mul_tier1(above, above) > min, "exact > FLT_MIN");

    // round_to_flt_min: exact just below FLT_MIN, fl32 rounds to FLT_MIN
    // data = FLT_MIN, filter = 1 - 2^-24
    let data = min;
    let filt = f32::from_bits(0x3f7f_ffff); // 1 - 2^-24
    let exact_val = (data as f64) * (filt as f64);
    assert!(exact_val < FLT_MIN, "exact is below FLT_MIN");
    assert_eq!((data * filt).to_bits(), min.to_bits(), "fl32 rounds up");
    assert_eq!(mul_tier1(data, filt).to_bits(), 0u32, "+0 for positive");
    assert_eq!(
        mul_tier1(-data, filt).to_bits(),
        0x8000_0000u32,
        "-0 for negative"
    );
    assert_eq!(
        mul_tier1(data, -filt).to_bits(),
        0x8000_0000u32,
        "-0 for negated filter"
    );

    // subnormal fl32 result (k=1,2,3 in round_to_flt_min still below FLT_MIN)
    let d1 = f32::from_bits(min.to_bits() + 1); // FLT_MIN + 2^-149
    let exact_d1 = (d1 as f64) * (filt as f64);
    if exact_d1 < FLT_MIN {
        assert_eq!(mul_tier1(d1, filt).to_bits(), 0, "still below FLT_MIN");
    } else {
        assert!(mul_tier1(d1, filt) >= min, "above FLT_MIN");
    }

    // data = FLT_MIN + 2^-149 (k=1): exact > FLT_MIN, fl32 = FLT_MIN, kept
    let d_k1 = f32::from_bits(min.to_bits() + 1); // first subnormal above FLT_MIN? No — FLT_MIN is normal
    // Actually FLT_MIN = 2^-126 is normal; FLT_MIN + 2^-149 = normal with next mantissa bit
    let exact_k1 = (d_k1 as f64) * (filt as f64);
    let result_k1 = mul_tier1(d_k1, filt);
    assert_eq!(
        result_k1.to_bits(),
        reference(d_k1, filt),
        "k=1 matches reference"
    );
    let _ = exact_k1; // used above for analysis, keep for documentation
}
