//! Deterministic, counter-based random numbers (`spec/VDS-1.md` section 5).
//!
//! * [`SplitMix64`] expands a 64-bit master seed into key material.
//! * [`philox_u32x4`] is Philox4x32-10 (Salmon, Moraes, Dror, Shaw,
//!   "Parallel random numbers: as easy as 1, 2, 3", SC 2011), implemented
//!   here from the published algorithm and checked against the Random123
//!   known-answer vectors.
//! * [`Stream`] maps `(master_seed, stream_id, index)` to an output word
//!   without any hidden sequential state, so outputs can be generated in any
//!   order or in parallel and are still bit-identical.
//!
//! # Mapping (normative, VDS-1 section 5.3)
//!
//! ```text
//! k          = splitmix64_next(state = master_seed)   // first output
//! key        = [lo32(k), hi32(k)]
//! block      = index >> 2           (index: u32-word position in the stream)
//! lane       = index & 3
//! counter    = [lo32(block), hi32(block), lo32(stream_id), hi32(stream_id)]
//! word(index) = philox_u32x4(key, counter)[lane]
//! ```
//!
//! Each stream therefore owns 2^66 words; distinct `stream_id`s use disjoint
//! counters under the same key.

/// SplitMix64 increment: the odd integer closest to 2^64 / phi.
pub const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64 (Steele, Lea, Flood 2014; fixed-increment form by Vigna 2015).
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a generator whose state is `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance the state by [`SPLITMIX64_GAMMA`] and return the mixed value.
    pub const fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX64_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Philox4x32 round multipliers.
pub const PHILOX_M4X32_0: u32 = 0xD251_1F53;
/// Philox4x32 round multipliers.
pub const PHILOX_M4X32_1: u32 = 0xCD9E_8D57;
/// Philox4x32 Weyl key increments (golden ratio, sqrt(3) - 1).
pub const PHILOX_W32_0: u32 = 0x9E37_79B9;
/// Philox4x32 Weyl key increments (golden ratio, sqrt(3) - 1).
pub const PHILOX_W32_1: u32 = 0xBB67_AE85;
/// Number of rounds.
pub const PHILOX_ROUNDS: usize = 10;

#[inline]
const fn mulhilo32(a: u32, b: u32) -> (u32, u32) {
    let p = (a as u64) * (b as u64);
    ((p >> 32) as u32, p as u32)
}

/// Philox4x32-10 block function: one 128-bit counter to four 32-bit words.
#[must_use]
pub const fn philox_u32x4(key: [u32; 2], ctr: [u32; 4]) -> [u32; 4] {
    let mut k = key;
    let mut c = ctr;
    let mut r = 0;
    while r < PHILOX_ROUNDS {
        if r > 0 {
            k[0] = k[0].wrapping_add(PHILOX_W32_0);
            k[1] = k[1].wrapping_add(PHILOX_W32_1);
        }
        let (hi0, lo0) = mulhilo32(PHILOX_M4X32_0, c[0]);
        let (hi1, lo1) = mulhilo32(PHILOX_M4X32_1, c[2]);
        c = [hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0];
        r += 1;
    }
    c
}

/// Exact 2^-24.
const TWO_POW_M24_F32: f32 = 1.0 / 16_777_216.0;
/// Exact 2^-53.
const TWO_POW_M53_F64: f64 = 1.0 / 9_007_199_254_740_992.0;

/// One random stream: `(master_seed, stream_id)` with a word position.
#[derive(Clone, Debug)]
pub struct Stream {
    key: [u32; 2],
    stream_id: u64,
    /// Position (in u32 words) of the next word `next_u32` returns.
    index: u128,
    /// Cached block for `index >> 2`, valid when `cached_block == Some(..)`.
    cached_block: Option<(u128, [u32; 4])>,
}

impl Stream {
    /// Create stream `stream_id` of `master_seed`, positioned at word 0.
    #[must_use]
    pub fn new(master_seed: u64, stream_id: u64) -> Self {
        let k = SplitMix64::new(master_seed).next_u64();
        Self {
            key: [k as u32, (k >> 32) as u32],
            stream_id,
            index: 0,
            cached_block: None,
        }
    }

    /// The Philox key derived from the master seed.
    #[must_use]
    pub const fn key(&self) -> [u32; 2] {
        self.key
    }

    /// Word position of the next output.
    #[must_use]
    pub const fn position(&self) -> u128 {
        self.index
    }

    /// Move to word position `index` (must be below 2^66).
    pub fn seek(&mut self, index: u128) {
        assert!(index < (1u128 << 66), "stream position out of range");
        self.index = index;
    }

    /// The word at position `index`, independent of the current position.
    #[must_use]
    pub fn word_at(&self, index: u128) -> u32 {
        let block = index >> 2;
        philox_u32x4(self.key, self.counter(block))[(index & 3) as usize]
    }

    fn counter(&self, block: u128) -> [u32; 4] {
        assert!(block < (1u128 << 64), "stream position out of range");
        [
            block as u32,
            (block >> 32) as u32,
            self.stream_id as u32,
            (self.stream_id >> 32) as u32,
        ]
    }

    /// Next 32-bit word.
    pub fn next_u32(&mut self) -> u32 {
        let block = self.index >> 2;
        let words = match self.cached_block {
            Some((b, w)) if b == block => w,
            _ => {
                let w = philox_u32x4(self.key, self.counter(block));
                self.cached_block = Some((block, w));
                w
            }
        };
        let out = words[(self.index & 3) as usize];
        self.index += 1;
        out
    }

    /// Next 64-bit word: the first word drawn is the low half.
    pub fn next_u64(&mut self) -> u64 {
        let lo = u64::from(self.next_u32());
        let hi = u64::from(self.next_u32());
        lo | (hi << 32)
    }

    /// Uniform binary32 in [0, 1): the 24 high bits of one word times 2^-24.
    /// Exact (no rounding occurs).
    pub fn next_f32_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * TWO_POW_M24_F32
    }

    /// Uniform binary64 in [0, 1): the 53 high bits of [`Self::next_u64`]
    /// times 2^-53. Exact (no rounding occurs).
    pub fn next_f64_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * TWO_POW_M53_F64
    }

    /// Standard normal variate (binary64) by the Box-Muller transform, using
    /// only `vikshep-detmath` functions and the correctly rounded `sqrt`:
    ///
    /// ```text
    /// u1 = 1 - next_f64_unit()          // in (0, 1], so ln(u1) is finite
    /// u2 = next_f64_unit()              // in [0, 1)
    /// r  = sqrt(-2 * ln(u1))
    /// z  = r * cos(TAU * u2)
    /// ```
    ///
    /// Each call consumes exactly four words (two `next_f64_unit` calls) and
    /// returns one variate; the sine branch is discarded so that a variate
    /// depends only on its own position in the stream. Each operation is
    /// evaluated left to right with one rounding (no fusion).
    pub fn next_normal_f64(&mut self) -> f64 {
        let u1 = 1.0 - self.next_f64_unit();
        let u2 = self.next_f64_unit();
        let r = (-2.0 * vikshep_detmath::ln(u1)).sqrt();
        r * vikshep_detmath::cos(core::f64::consts::TAU * u2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Philox4x32-10 known-answer vectors, copied verbatim from the Random123
    // distribution, file `tests/kat_vectors` at release tag v1.14.0 of
    // https://github.com/DEShawResearch/random123 (the file is identical on
    // `main` at commit 9545ff64). Lines 27-29:
    //
    // philox4x32 10 00000000 00000000 00000000 00000000 00000000 00000000   6627e8d5 e169c58d bc57ac4c 9b00dbd8
    // philox4x32 10 ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff   408f276d 41c83b0e a20bc7c6 6d5451fd
    // philox4x32 10 243f6a88 85a308d3 13198a2e 03707344 a4093822 299f31d0   d16cfe09 94fdcceb 5001e420 24126ea1
    //
    // Column layout (from the file header): name, rounds, CTR (4 words),
    // KEY (2 words), EXPECTED (4 words).
    const PHILOX_KAT: [([u32; 4], [u32; 2], [u32; 4]); 3] = [
        (
            [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000],
            [0x0000_0000, 0x0000_0000],
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8],
        ),
        (
            [0xffff_ffff, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff],
            [0xffff_ffff, 0xffff_ffff],
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd],
        ),
        (
            [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
            [0xa409_3822, 0x299f_31d0],
            [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1],
        ),
    ];

    #[test]
    fn philox4x32_10_known_answers() {
        for (ctr, key, expected) in PHILOX_KAT {
            assert_eq!(
                philox_u32x4(key, ctr),
                expected,
                "ctr={ctr:08x?} key={key:08x?}"
            );
        }
    }

    // SplitMix64 reference outputs. Produced by two independent published
    // reference implementations, which agree on every value:
    //  1. Sebastiano Vigna's splitmix64.c (public domain, 2015), as
    //     reproduced in https://github.com/lemire/testingRNG,
    //     source/splitmix64.h, compiled with gcc;
    //  2. OpenJDK java.util.SplittableRandom (the original SplitMix of
    //     Steele, Lea and Flood, 2014): new SplittableRandom(seed).nextLong(),
    //     OpenJDK 21.0.11.
    const SPLITMIX64_SEED0: [u64; 5] = [
        0xe220_a839_7b1d_cdaf,
        0x6e78_9e6a_a1b9_65f4,
        0x06c4_5d18_8009_454f,
        0xf88b_b8a8_724c_81ec,
        0x1b39_896a_51a8_749b,
    ];
    const SPLITMIX64_SEED1234567: [u64; 5] = [
        0x599e_d017_fb08_fc85,
        0x2c73_f084_5854_0fa5,
        0x883e_bce5_a3f2_7c77,
        0x3fbe_f740_e917_7b3f,
        0xe3b8_3467_08cb_5ecd,
    ];

    #[test]
    fn splitmix64_known_answers() {
        for (seed, expected) in [(0, SPLITMIX64_SEED0), (1_234_567, SPLITMIX64_SEED1234567)] {
            let mut g = SplitMix64::new(seed);
            for want in expected {
                assert_eq!(g.next_u64(), want, "seed {seed}");
            }
        }
    }

    #[test]
    fn stream_mapping_matches_spec() {
        let s = Stream::new(42, 7);
        let k = SplitMix64::new(42).next_u64();
        assert_eq!(s.key(), [k as u32, (k >> 32) as u32]);
        let mut seq = s.clone();
        for index in 0..64u128 {
            let block = index >> 2;
            let ctr = [block as u32, (block >> 32) as u32, 7, 0];
            let want = philox_u32x4(s.key(), ctr)[(index & 3) as usize];
            assert_eq!(s.word_at(index), want);
            assert_eq!(seq.next_u32(), want);
        }
    }

    #[test]
    fn order_free_generation() {
        let s = Stream::new(1, 2);
        let forward: Vec<u32> = (0..1000u128).map(|i| s.word_at(i)).collect();
        let mut backward: Vec<u32> = (0..1000u128).rev().map(|i| s.word_at(i)).collect();
        backward.reverse();
        assert_eq!(forward, backward);
        let mut t = Stream::new(1, 2);
        t.seek(500);
        assert_eq!(t.next_u32(), forward[500]);
    }

    #[test]
    fn streams_are_distinct() {
        let a: Vec<u32> = (0..16).map(|i| Stream::new(9, 0).word_at(i)).collect();
        let b: Vec<u32> = (0..16).map(|i| Stream::new(9, 1).word_at(i)).collect();
        let c: Vec<u32> = (0..16).map(|i| Stream::new(10, 0).word_at(i)).collect();
        assert_ne!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn unit_conversions_are_exact_and_in_range() {
        let mut s = Stream::new(3, 0);
        for _ in 0..10_000 {
            let mut t = s.clone();
            let w = t.next_u32();
            let f = s.next_f32_unit();
            assert!((0.0..1.0).contains(&f));
            assert_eq!(f64::from(f), f64::from(w >> 8) / 16_777_216.0);
        }
        let mut s = Stream::new(3, 1);
        for _ in 0..10_000 {
            let mut t = s.clone();
            let w = t.next_u64();
            let f = s.next_f64_unit();
            assert!((0.0..1.0).contains(&f));
            assert_eq!(f * 9_007_199_254_740_992.0, (w >> 11) as f64);
        }
        // Extremes.
        assert_eq!(
            ((u32::MAX >> 8) as f32) * TWO_POW_M24_F32,
            1.0 - f32::EPSILON / 2.0
        );
        assert_eq!(
            ((u64::MAX >> 11) as f64) * TWO_POW_M53_F64,
            1.0 - f64::EPSILON / 2.0
        );
    }

    #[test]
    fn normal_variates_have_plausible_moments() {
        let mut s = Stream::new(2026, 0);
        let n = 200_000;
        let (mut sum, mut sum2) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let z = s.next_normal_f64();
            assert!(z.is_finite());
            sum += z;
            sum2 += z * z;
        }
        let mean = sum / f64::from(n);
        let var = sum2 / f64::from(n) - mean * mean;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "variance {var}");
        assert_eq!(s.position(), 4 * u128::from(n as u32));
    }
}
