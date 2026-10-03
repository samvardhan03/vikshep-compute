//! Deterministic radix-2 Stockham FFT (`spec/VDS-1.md` sections 6 and 7).
//!
//! The arithmetic of every output element is exactly that of the normative
//! recursion `fft0` in VDS-1 section 6.2: the recursion is unrolled into a
//! loop over levels that alternates the source and destination buffers, and
//! each butterfly evaluates
//!
//! ```text
//! y[q + s*(2p)]   = (a.re + b.re, a.im + b.im)
//! d               = (a.re - b.re, a.im - b.im)
//! y[q + s*(2p+1)] = (d.re*w.re - d.im*w.im, d.re*w.im + d.im*w.re)
//! ```
//!
//! with one rounding per operation and no fusion. The result always ends in
//! the input buffer (for odd `m` the final step copies it from scratch).
//!
//! Binary32 ([`Complex32`]) is the Tier-1 precision. The same code runs in
//! binary64 ([`Complex64`]) for host-side (Tier-2) filter construction.

use core::fmt::Debug;
use core::ops::{Add, Mul, Neg, Sub};
use std::sync::OnceLock;

/// Smallest supported log2 transform length.
pub const MIN_LOG2_LEN: u32 = 1;
/// Largest supported log2 transform length (N = 4096).
pub const MAX_LOG2_LEN: u32 = 12;

/// A complex number stored as `(re, im)`; `#[repr(C)]` so that buffers can be
/// handed to foreign backends as interleaved little-endian pairs.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex<T> {
    /// Real part.
    pub re: T,
    /// Imaginary part.
    pub im: T,
}

/// Binary32 complex value (Tier 1).
pub type Complex32 = Complex<f32>;
/// Binary64 complex value (host only).
pub type Complex64 = Complex<f64>;

impl<T> Complex<T> {
    /// Construct from parts.
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
}

/// Transform direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    /// `X[k] = sum_j x[j] exp(-2 pi i j k / N)`.
    Forward,
    /// `x[j] = 2^-m sum_k X[k] exp(+2 pi i j k / N)`.
    Inverse,
}

/// The floating-point types the FFT runs in.
pub trait Real:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + PartialEq
    + Debug
    + Send
    + Sync
    + 'static
{
    /// Round a binary64 value to this type (round-to-nearest-even).
    fn from_f64(x: f64) -> Self;
    /// The correctly rounded square root of 1/2 in this type.
    fn sqrt_half() -> Self;
    /// Exact `2^e` (normal range only).
    fn pow2(e: i32) -> Self;
    /// Cached forward twiddle table for `N = 2^m`.
    fn twiddles(m: u32) -> &'static [Complex<Self>];
    /// Cached conjugate (inverse) twiddle table for `N = 2^m`.
    fn twiddles_inverse(m: u32) -> &'static [Complex<Self>];
}

const TABLES: usize = MAX_LOG2_LEN as usize + 1;

static TW32: [OnceLock<Vec<Complex32>>; TABLES] = [const { OnceLock::new() }; TABLES];
static TW32_INV: [OnceLock<Vec<Complex32>>; TABLES] = [const { OnceLock::new() }; TABLES];
static TW64: [OnceLock<Vec<Complex64>>; TABLES] = [const { OnceLock::new() }; TABLES];
static TW64_INV: [OnceLock<Vec<Complex64>>; TABLES] = [const { OnceLock::new() }; TABLES];

fn check_log2(m: u32) {
    assert!(
        (MIN_LOG2_LEN..=MAX_LOG2_LEN).contains(&m),
        "FFT length 2^{m} outside 2..=4096"
    );
}

impl Real for f32 {
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    fn sqrt_half() -> Self {
        0.5f32.sqrt()
    }
    fn pow2(e: i32) -> Self {
        assert!((-126..=127).contains(&e));
        f32::from_bits(((e + 127) as u32) << 23)
    }
    fn twiddles(m: u32) -> &'static [Complex32] {
        check_log2(m);
        TW32[m as usize].get_or_init(|| build_twiddles(m))
    }
    fn twiddles_inverse(m: u32) -> &'static [Complex32] {
        check_log2(m);
        TW32_INV[m as usize].get_or_init(|| conjugate(Self::twiddles(m)))
    }
}

impl Real for f64 {
    fn from_f64(x: f64) -> Self {
        x
    }
    fn sqrt_half() -> Self {
        0.5f64.sqrt()
    }
    fn pow2(e: i32) -> Self {
        assert!((-1022..=1023).contains(&e));
        f64::from_bits(((e + 1023) as u64) << 52)
    }
    fn twiddles(m: u32) -> &'static [Complex64] {
        check_log2(m);
        TW64[m as usize].get_or_init(|| build_twiddles(m))
    }
    fn twiddles_inverse(m: u32) -> &'static [Complex64] {
        check_log2(m);
        TW64_INV[m as usize].get_or_init(|| conjugate(Self::twiddles(m)))
    }
}

fn conjugate<T: Real>(tw: &[Complex<T>]) -> Vec<Complex<T>> {
    tw.iter().map(|w| Complex::new(w.re, -w.im)).collect()
}

/// Build `TW_N[p]`, `p in 0..N/2`, by the first-octant construction of
/// VDS-1 section 7 (binary64 evaluation with `vikshep-detmath`, one rounding
/// to `T`, exact symmetry for the rest of the half circle).
#[must_use]
pub fn build_twiddles<T: Real>(m: u32) -> Vec<Complex<T>> {
    check_log2(m);
    let n = 1usize << m;
    let q4 = n / 4;
    let q8 = n / 8;
    // Exact: TAU times 2^-m.
    let step = core::f64::consts::TAU * <f64 as Real>::pow2(-(m as i32));
    let mut c = vec![T::from_f64(0.0); q4 + 1];
    let mut s = vec![T::from_f64(0.0); q4 + 1];
    for k in 0..=q8 {
        let theta = k as f64 * step;
        c[k] = T::from_f64(vikshep_detmath::cos(theta));
        s[k] = T::from_f64(vikshep_detmath::sin(theta));
    }
    if n >= 8 {
        c[q8] = T::sqrt_half();
        s[q8] = T::sqrt_half();
    }
    for k in (q8 + 1)..=q4 {
        c[k] = s[q4 - k];
        s[k] = c[q4 - k];
    }
    (0..n / 2)
        .map(|p| {
            if p <= q4 {
                Complex::new(c[p], -s[p])
            } else {
                Complex::new(-s[p - q4], -c[p - q4])
            }
        })
        .collect()
}

/// `log2(n)` for a supported transform length, panicking otherwise.
#[must_use]
pub fn log2_len(n: usize) -> u32 {
    assert!(n.is_power_of_two(), "FFT length {n} is not a power of two");
    let m = n.trailing_zeros();
    check_log2(m);
    m
}

/// One level of the recursion: `fft0(n, s, _, src, dst)` without the tail call.
///
/// Iterates over contiguous blocks of `s` elements (slices instead of index
/// arithmetic, so the compiler can drop bounds checks); the arithmetic of
/// each output element is exactly the butterfly of VDS-1 section 6.2.
#[inline]
fn stage<T: Real>(
    src: &[Complex<T>],
    dst: &mut [Complex<T>],
    n: usize,
    s: usize,
    tw: &[Complex<T>],
    tw_stride: usize,
) {
    let m = n / 2;
    let (src_a, src_b) = src[..n * s].split_at(m * s);
    let pairs = dst[..n * s].chunks_exact_mut(2 * s);
    let blocks = src_a.chunks_exact(s).zip(src_b.chunks_exact(s));
    for (p, (y, (a_blk, b_blk))) in pairs.zip(blocks).enumerate() {
        let w = tw[p * tw_stride];
        let (y0, y1) = y.split_at_mut(s);
        for (((y0, y1), a), b) in y0.iter_mut().zip(y1.iter_mut()).zip(a_blk).zip(b_blk) {
            *y0 = Complex::new(a.re + b.re, a.im + b.im);
            let d_re = a.re - b.re;
            let d_im = a.im - b.im;
            *y1 = Complex::new(d_re * w.re - d_im * w.im, d_re * w.im + d_im * w.re);
        }
    }
}

/// The normative recursion of VDS-1 section 6.2 with table `tw` (forward or
/// conjugate). `x.len()` must equal `2 * tw.len()`; `scratch` must be at
/// least as long as `x`. The result is left in `x`; no scaling is applied.
pub fn stockham<T: Real>(x: &mut [Complex<T>], scratch: &mut [Complex<T>], tw: &[Complex<T>]) {
    let big_n = x.len();
    assert_eq!(big_n, 2 * tw.len(), "twiddle table does not match length");
    let y = &mut scratch[..big_n];
    let mut n = big_n;
    let mut s = 1;
    let mut data_in_x = true;
    while n > 1 {
        if data_in_x {
            stage(x, y, n, s, tw, big_n / n);
        } else {
            stage(y, x, n, s, tw, big_n / n);
        }
        data_in_x = !data_in_x;
        n /= 2;
        s *= 2;
    }
    // Final n == 1 step with eo == true: copy the result into x.
    if !data_in_x {
        x.copy_from_slice(y);
    }
}

/// In-place 1-D transform of one signal (VDS-1 sections 6.2 and 6.5).
pub fn fft_1d<T: Real>(x: &mut [Complex<T>], scratch: &mut [Complex<T>], dir: Direction) {
    let m = log2_len(x.len());
    match dir {
        Direction::Forward => stockham(x, scratch, T::twiddles(m)),
        Direction::Inverse => {
            stockham(x, scratch, T::twiddles_inverse(m));
            let scale = T::pow2(-(m as i32));
            for v in x.iter_mut() {
                v.re = v.re * scale;
                v.im = v.im * scale;
            }
        }
    }
}

/// In-place 1-D transforms of every length-`n` chunk of `data`.
pub fn fft_batch<T: Real>(data: &mut [Complex<T>], n: usize, dir: Direction) {
    assert_eq!(data.len() % n, 0, "batch length is not a multiple of n");
    let mut scratch = vec![Complex::new(T::from_f64(0.0), T::from_f64(0.0)); n];
    for chunk in data.chunks_exact_mut(n) {
        fft_1d(chunk, &mut scratch, dir);
    }
}

/// Reusable buffers for [`fft_2d_with`].
#[derive(Debug, Default)]
pub struct Workspace<T> {
    scratch: Vec<Complex<T>>,
    transposed: Vec<Complex<T>>,
}

impl<T: Real> Workspace<T> {
    /// Empty workspace; grows on first use.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            scratch: Vec::new(),
            transposed: Vec::new(),
        }
    }

    fn ensure(&mut self, n: usize, total: usize) {
        let zero = Complex::new(T::from_f64(0.0), T::from_f64(0.0));
        if self.scratch.len() < n {
            self.scratch.resize(n, zero);
        }
        if self.transposed.len() < total {
            self.transposed.resize(total, zero);
        }
    }
}

/// In-place 2-D transform of a row-major `rows x cols` array (VDS-1 section
/// 6.6): every row first (length `cols`), then every column (length `rows`).
/// An axis of length 1 is left untouched, so `rows == 1` is a 1-D transform.
/// For the inverse, each 1-D transform applies its own `2^-m` scaling.
pub fn fft_2d_with<T: Real>(
    data: &mut [Complex<T>],
    rows: usize,
    cols: usize,
    dir: Direction,
    ws: &mut Workspace<T>,
) {
    let table = |n: usize| -> &'static [Complex<T>] {
        if n == 1 {
            &[]
        } else {
            match dir {
                Direction::Forward => T::twiddles(log2_len(n)),
                Direction::Inverse => T::twiddles_inverse(log2_len(n)),
            }
        }
    };
    fft_2d_tables(
        data,
        rows,
        cols,
        table(cols),
        table(rows),
        dir == Direction::Inverse,
        ws,
    );
}

/// 2-D transform with caller-supplied tables: `tw_cols` for the row
/// transforms (length `cols`), `tw_rows` for the column transforms (length
/// `rows`); a table for an axis of length 1 is ignored. Forward tables give
/// the forward transform; conjugate tables with `inverse_scaling` give the
/// inverse (each 1-D transform followed by multiplication with `2^-m`).
/// This is the entry point backends use with host-built tables (VDS-1
/// section 3.4).
pub fn fft_2d_tables<T: Real>(
    data: &mut [Complex<T>],
    rows: usize,
    cols: usize,
    tw_cols: &[Complex<T>],
    tw_rows: &[Complex<T>],
    inverse_scaling: bool,
    ws: &mut Workspace<T>,
) {
    assert_eq!(data.len(), rows * cols, "2-D buffer has the wrong length");
    ws.ensure(rows.max(cols), rows * cols);
    let one_d = |x: &mut [Complex<T>], scratch: &mut [Complex<T>], tw: &[Complex<T>]| {
        stockham(x, scratch, tw);
        if inverse_scaling {
            let scale = T::pow2(-(log2_len(x.len()) as i32));
            for v in x.iter_mut() {
                v.re = v.re * scale;
                v.im = v.im * scale;
            }
        }
    };
    if cols > 1 {
        let _ = log2_len(cols);
        for row in data.chunks_exact_mut(cols) {
            one_d(row, &mut ws.scratch[..cols], tw_cols);
        }
    }
    if rows > 1 {
        let _ = log2_len(rows);
        let t = &mut ws.transposed[..rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                t[c * rows + r] = data[r * cols + c];
            }
        }
        for col in t.chunks_exact_mut(rows) {
            one_d(col, &mut ws.scratch[..rows], tw_rows);
        }
        for r in 0..rows {
            for c in 0..cols {
                data[r * cols + c] = t[c * rows + r];
            }
        }
    }
}

/// [`fft_2d_with`] with a temporary workspace.
pub fn fft_2d<T: Real>(data: &mut [Complex<T>], rows: usize, cols: usize, dir: Direction) {
    fft_2d_with(data, rows, cols, dir, &mut Workspace::new());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_lengths_and_cache() {
        for m in MIN_LOG2_LEN..=MAX_LOG2_LEN {
            assert_eq!(f32::twiddles(m).len(), 1 << (m - 1));
            assert!(core::ptr::eq(f32::twiddles(m), f32::twiddles(m)));
            assert_eq!(f64::twiddles_inverse(m).len(), 1 << (m - 1));
        }
    }

    #[test]
    fn f32_table_is_f64_table_rounded() {
        // Every binary32 entry is the binary32 rounding of the binary64 entry
        // (both come from the same binary64 cos/sin values).
        for m in MIN_LOG2_LEN..=MAX_LOG2_LEN {
            for (a, b) in f32::twiddles(m).iter().zip(f64::twiddles(m)) {
                assert_eq!(a.re, b.re as f32);
                assert_eq!(a.im, b.im as f32);
            }
        }
    }

    #[test]
    #[should_panic(expected = "outside 2..=4096")]
    fn rejects_8192() {
        let _ = f32::twiddles(13);
    }
}
