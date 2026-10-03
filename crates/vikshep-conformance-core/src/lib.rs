//! VDS-1 conformance cases (library behind the `vikshep-conformance`
//! command, the C ABI self-test and the Python self-test).
//!
//! * The cross-platform determinism sweeps of `spec/VDS-1.md` sections 4.4
//!   and 5.6 (this module): hashes of one million outputs of every
//!   `vikshep-detmath` function and of the random streams.
//! * Conformance suite v1 ([`suite`]): FFT, kernel and scattering cases
//!   expanded from `conformance/cases.toml`, with expected vectors in
//!   `conformance/vectors/v1/`. The sweeps are part of it as `sweep/*` cases.
//!
//! The sweep definitions below are normative: they are reproduced in the
//! specification, and any change to them is a change of conformance vectors.

pub mod suite;
pub mod tier2;

use vikshep_numerics::NUMERICS_VERSION;
use vikshep_numerics::oid::{hex, sha3_256};
use vikshep_numerics::rng::{SplitMix64, Stream};

/// Number of evaluations per sweep.
pub const SWEEP_LEN: usize = 1_000_000;

/// Master seed of every sweep stream: ASCII "VDS1".
pub const SWEEP_SEED: u64 = 0x5644_5331;

/// Golden hashes for the current numerics version.
pub const GOLDEN_HASHES_JSON: &str = include_str!("../../../conformance/vectors/v1/hashes.json");

/// Exact 2^e for binary64, -1022 <= e <= 1023.
fn pow2_f64(e: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&e));
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// Exact 2^e for binary32, -126 <= e <= 127.
fn pow2_f32(e: i32) -> f32 {
    debug_assert!((-126..=127).contains(&e));
    f32::from_bits(((e + 127) as u32) << 23)
}

const SIGN64: u64 = 1 << 63;
const INF64_BITS: u64 = 0x7FF0_0000_0000_0000;
const SIGN32: u32 = 1 << 31;
const INF32_BITS: u32 = 0x7F80_0000;

/// Sweep inputs for one function (VDS-1 section 4.3).
#[derive(Clone, Copy, Debug)]
pub enum Domain {
    /// exp f64: x = 1456 * u - 746, u = next_f64_unit (one rounding each).
    ExpF64,
    /// ln f64: x = from_bits(next_u64 mod 0x7FF0000000000000): every
    /// non-negative finite binary64 including +0 and subnormals.
    LnF64,
    /// sin/cos f64, input i consumes a = next_u64 then b = next_u64:
    /// if i mod 16 == 15, x = any finite binary64 built from a
    /// (sign bit of a, magnitude bits of a mod 0x7FF0000000000000);
    /// otherwise x = (2u - 1) * 2^e with u = (a >> 11) * 2^-53 and
    /// e = (b mod 65) - 32.
    TrigF64,
    /// exp f32: x = 194 * u - 105, u = next_f32_unit (binary32 arithmetic).
    ExpF32,
    /// ln f32: x = from_bits(next_u32 mod 0x7F800000).
    LnF32,
    /// sin/cos f32, as TrigF64 with a = next_u32, b = next_u32,
    /// u = (a >> 8) * 2^-24, e = (b mod 33) - 16 and the magnitude modulus
    /// 0x7F800000.
    TrigF32,
}

fn input_f64(domain: Domain, i: usize, s: &mut Stream) -> f64 {
    match domain {
        Domain::ExpF64 => 1456.0 * s.next_f64_unit() - 746.0,
        Domain::LnF64 => f64::from_bits(s.next_u64() % INF64_BITS),
        Domain::TrigF64 => {
            let a = s.next_u64();
            let b = s.next_u64();
            if i % 16 == 15 {
                f64::from_bits((a & SIGN64) | ((a & !SIGN64) % INF64_BITS))
            } else {
                let u = (a >> 11) as f64 * pow2_f64(-53);
                let e = (b % 65) as i32 - 32;
                (2.0 * u - 1.0) * pow2_f64(e)
            }
        }
        _ => unreachable!("not a binary64 domain"),
    }
}

fn input_f32(domain: Domain, i: usize, s: &mut Stream) -> f32 {
    match domain {
        Domain::ExpF32 => 194.0 * s.next_f32_unit() - 105.0,
        Domain::LnF32 => f32::from_bits(s.next_u32() % INF32_BITS),
        Domain::TrigF32 => {
            let a = s.next_u32();
            let b = s.next_u32();
            if i % 16 == 15 {
                f32::from_bits((a & SIGN32) | ((a & !SIGN32) % INF32_BITS))
            } else {
                let u = (a >> 8) as f32 * pow2_f32(-24);
                let e = (b % 33) as i32 - 16;
                (2.0 * u - 1.0) * pow2_f32(e)
            }
        }
        _ => unreachable!("not a binary32 domain"),
    }
}

/// One named sweep case.
pub struct Case {
    /// Key in the hashes JSON.
    pub name: &'static str,
    /// Stream id under [`SWEEP_SEED`].
    pub stream_id: u64,
    run: fn(&mut Stream) -> Vec<u8>,
}

fn sweep_f64(s: &mut Stream, domain: Domain, f: fn(f64) -> f64) -> Vec<u8> {
    let mut out = Vec::with_capacity(SWEEP_LEN * 8);
    for i in 0..SWEEP_LEN {
        let y = f(input_f64(domain, i, s));
        assert!(!y.is_nan(), "sweep produced NaN");
        out.extend_from_slice(&y.to_le_bytes());
    }
    out
}

fn sweep_f32(s: &mut Stream, domain: Domain, f: fn(f32) -> f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(SWEEP_LEN * 4);
    for i in 0..SWEEP_LEN {
        let y = f(input_f32(domain, i, s));
        assert!(!y.is_nan(), "sweep produced NaN");
        out.extend_from_slice(&y.to_le_bytes());
    }
    out
}

/// All C0 cases, in output order (sorted by name).
pub const CASES: &[Case] = &[
    Case {
        name: "detmath.cos.f32",
        stream_id: 8,
        run: |s| sweep_f32(s, Domain::TrigF32, vikshep_detmath::cos_f32),
    },
    Case {
        name: "detmath.cos.f64",
        stream_id: 4,
        run: |s| sweep_f64(s, Domain::TrigF64, vikshep_detmath::cos),
    },
    Case {
        name: "detmath.exp.f32",
        stream_id: 5,
        run: |s| sweep_f32(s, Domain::ExpF32, vikshep_detmath::exp_f32),
    },
    Case {
        name: "detmath.exp.f64",
        stream_id: 1,
        run: |s| sweep_f64(s, Domain::ExpF64, vikshep_detmath::exp),
    },
    Case {
        name: "detmath.ln.f32",
        stream_id: 6,
        run: |s| sweep_f32(s, Domain::LnF32, vikshep_detmath::ln_f32),
    },
    Case {
        name: "detmath.ln.f64",
        stream_id: 2,
        run: |s| sweep_f64(s, Domain::LnF64, vikshep_detmath::ln),
    },
    Case {
        name: "detmath.sin.f32",
        stream_id: 7,
        run: |s| sweep_f32(s, Domain::TrigF32, vikshep_detmath::sin_f32),
    },
    Case {
        name: "detmath.sin.f64",
        stream_id: 3,
        run: |s| sweep_f64(s, Domain::TrigF64, vikshep_detmath::sin),
    },
    Case {
        name: "rng.philox.f32_unit",
        stream_id: 101,
        run: |s| {
            (0..SWEEP_LEN)
                .flat_map(|_| s.next_f32_unit().to_le_bytes())
                .collect()
        },
    },
    Case {
        name: "rng.philox.f64_unit",
        stream_id: 102,
        run: |s| {
            (0..SWEEP_LEN)
                .flat_map(|_| s.next_f64_unit().to_le_bytes())
                .collect()
        },
    },
    Case {
        name: "rng.philox.normal_f64",
        stream_id: 103,
        run: |s| {
            (0..SWEEP_LEN)
                .flat_map(|_| s.next_normal_f64().to_le_bytes())
                .collect()
        },
    },
    Case {
        name: "rng.philox.u32",
        stream_id: 100,
        run: |s| {
            (0..SWEEP_LEN)
                .flat_map(|_| s.next_u32().to_le_bytes())
                .collect()
        },
    },
    Case {
        name: "rng.splitmix64.u64",
        stream_id: 0,
        run: |_| {
            let mut g = SplitMix64::new(SWEEP_SEED);
            (0..SWEEP_LEN)
                .flat_map(|_| g.next_u64().to_le_bytes())
                .collect()
        },
    },
];

/// Run one sweep and return its output bytes.
#[must_use]
pub fn case_bytes(case: &Case) -> Vec<u8> {
    let mut stream = Stream::new(SWEEP_SEED, case.stream_id);
    (case.run)(&mut stream)
}

/// Run one sweep and return the lowercase hex SHA3-256 of its output bytes.
#[must_use]
pub fn case_hash(case: &Case) -> String {
    hex(&sha3_256(&case_bytes(case)))
}

/// The `hashes` report: canonical JSON (sorted keys, two-space indent, LF
/// line endings, trailing newline). Contains no platform information so that
/// reports from different machines can be compared byte for byte.
#[must_use]
pub fn hashes_json() -> String {
    let mut s = String::from("{\n  \"hashes\": {\n");
    for (i, case) in CASES.iter().enumerate() {
        let sep = if i + 1 == CASES.len() { "" } else { "," };
        s.push_str(&format!(
            "    \"{}\": \"{}\"{sep}\n",
            case.name,
            case_hash(case)
        ));
    }
    s.push_str(&format!(
        "  }},\n  \"numerics_version\": {NUMERICS_VERSION},\n  \"sweep_length\": {SWEEP_LEN}\n}}\n"
    ));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cases_are_sorted_and_streams_unique() {
        let names: Vec<_> = CASES.iter().map(|c| c.name).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
        let mut ids: Vec<_> = CASES.iter().map(|c| c.stream_id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), CASES.len());
    }

    #[test]
    fn pow2_is_exact() {
        assert_eq!(pow2_f64(0), 1.0);
        assert_eq!(pow2_f64(-53), f64::EPSILON / 2.0);
        assert_eq!(pow2_f32(-24), f32::EPSILON / 2.0);
        assert_eq!(pow2_f32(16), 65536.0);
    }

    #[test]
    fn domains_cover_intended_ranges() {
        let mut s = Stream::new(SWEEP_SEED, 999);
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut subnormal = false;
        let mut huge = false;
        for i in 0..20_000 {
            let x = input_f64(Domain::ExpF64, i, &mut s);
            lo = lo.min(x);
            hi = hi.max(x);
            let y = input_f64(Domain::LnF64, i, &mut s);
            assert!(y.is_finite() && y >= 0.0);
            subnormal |= y.is_subnormal();
            let t = input_f64(Domain::TrigF64, i, &mut s);
            assert!(t.is_finite());
            huge |= t.abs() > 1e100;
        }
        assert!(lo < -745.0 && hi > 709.0, "exp domain [{lo}, {hi}]");
        assert!(huge);
        // Subnormals are 2^-11 of the ln domain; 20k samples see about 10.
        assert!(subnormal);
    }

    #[test]
    fn hashes_match_golden_vectors() {
        assert_eq!(hashes_json(), GOLDEN_HASHES_JSON);
    }
}
