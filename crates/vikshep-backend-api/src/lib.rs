//! Backend interface: the five Tier-1 kernels of `spec/VDS-1.md` section
//! 14.6 that every VDS-1 backend implements bit-exactly.
//!
//! The host (the `vikshep-scatter` crate) builds filters and twiddle tables,
//! moves data, and performs every reduction. A backend executes only:
//!
//! | Kernel | Effect on each element |
//! |---|---|
//! | [`ScatterBackend::fft`] | forward 2-D Stockham FFT (rows, then columns) |
//! | [`ScatterBackend::ifft`] | inverse FFT with conjugate tables and `2^-m` scaling per axis |
//! | [`ScatterBackend::mul_real_filter`] | `(re * h, im * h)` |
//! | [`ScatterBackend::modulus`] | `(sqrt(re * re + im * im), +0)` |
//! | [`ScatterBackend::subsample`] | `out = in[f * (o_r + a)][f * (o_c + b)].re` |
//!
//! All buffers are batches of contiguous row-major canvases of
//! [`Complex32`] (interleaved little-endian binary32 `re, im`). A 1-D canvas
//! has `rows == 1`. Kernels act on each canvas (and each element)
//! independently, so a backend may process canvases in any order or in
//! parallel; it must not reorder arithmetic within an element.

use core::fmt;

pub use vikshep_numerics::fft::Complex32;

/// Shape of one canvas. `rows == 1` for 1-D signals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Canvas {
    /// Number of rows (1 for 1-D).
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
}

impl Canvas {
    /// A 1-D canvas of length `n`.
    #[must_use]
    pub const fn one_d(n: usize) -> Self {
        Self { rows: 1, cols: n }
    }

    /// A 2-D canvas.
    #[must_use]
    pub const fn two_d(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    /// Number of elements in one canvas.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.rows * self.cols
    }

    /// True if the canvas has no elements (never valid for a kernel).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Host-built twiddle tables for one transform (VDS-1 section 7).
#[derive(Clone, Copy, Debug)]
pub struct Twiddles<'a> {
    /// Table for the row transforms (length `cols`), `cols / 2` entries.
    pub cols: &'a [Complex32],
    /// Table for the column transforms (length `rows`), `rows / 2` entries;
    /// empty when `rows == 1`.
    pub rows: &'a [Complex32],
}

/// Parameters of [`ScatterBackend::subsample`], in output units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubsampleSpec {
    /// Subsampling factor `f` (a power of two, applied to both axes of a 2-D
    /// canvas and to the columns of a 1-D canvas).
    pub factor: usize,
    /// Output row offset `o_r` (0 for 1-D).
    pub offset_rows: usize,
    /// Output column offset `o_c`.
    pub offset_cols: usize,
    /// Output rows (1 for 1-D).
    pub out_rows: usize,
    /// Output columns.
    pub out_cols: usize,
}

impl SubsampleSpec {
    /// Number of output values per canvas.
    #[must_use]
    pub const fn out_len(&self) -> usize {
        self.out_rows * self.out_cols
    }
}

/// What a backend supports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capabilities {
    /// Largest log2 transform length per axis (12 for VDS-1).
    pub max_log2_len: u32,
    /// Whether the backend processes canvases of one call in parallel.
    pub parallel: bool,
    /// Whether IEEE subnormals are preserved (VDS-1 section 8, D-FTZ).
    pub preserves_subnormals: bool,
}

/// Errors a kernel can report. A conforming backend never returns an error
/// for arguments that pass [`validate`] checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendError {
    /// Arguments violate the kernel contract.
    InvalidArgument(String),
    /// The device failed.
    Device(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
            Self::Device(m) => write!(f, "device error: {m}"),
        }
    }
}

impl std::error::Error for BackendError {}

/// The five Tier-1 kernels (VDS-1 section 14.6).
pub trait ScatterBackend: Send + Sync {
    /// Backend name, for example `"cpu"`.
    fn name(&self) -> &str;

    /// The `numerics_version` the backend implements.
    fn numerics_version(&self) -> u32;

    /// Supported features.
    fn capabilities(&self) -> Capabilities;

    /// In-place forward transform of every canvas in `data`
    /// (`data.len()` is a multiple of `canvas.len()`), using forward tables.
    fn fft(
        &self,
        data: &mut [Complex32],
        canvas: Canvas,
        tw: &Twiddles<'_>,
    ) -> Result<(), BackendError>;

    /// In-place inverse transform of every canvas, using conjugate tables;
    /// each 1-D transform is followed by multiplication with `2^-m`.
    fn ifft(
        &self,
        data: &mut [Complex32],
        canvas: Canvas,
        tw_conj: &Twiddles<'_>,
    ) -> Result<(), BackendError>;

    /// Multiply canvas `b` element-wise by filter `index[b]` of `bank`
    /// (`bank` holds filters of `canvas.len()` binary32 values each):
    /// `(re, im) -> (re * h, im * h)`.
    fn mul_real_filter(
        &self,
        data: &mut [Complex32],
        canvas: Canvas,
        bank: &[f32],
        index: &[u32],
    ) -> Result<(), BackendError>;

    /// Element-wise `(re, im) -> (sqrt(re * re + im * im), +0)`: two
    /// products, one addition, correctly rounded square root.
    fn modulus(&self, data: &mut [Complex32]) -> Result<(), BackendError>;

    /// For each canvas `b` of `input`, write the real parts
    /// `out[b][a][c] = input[b][f * (o_r + a)][f * (o_c + c)].re`.
    fn subsample(
        &self,
        input: &[Complex32],
        canvas: Canvas,
        spec: SubsampleSpec,
        out: &mut [f32],
    ) -> Result<(), BackendError>;
}

/// Argument checks shared by all backends.
pub mod validate {
    use super::{BackendError, Canvas, Complex32, SubsampleSpec, Twiddles};

    fn bad(msg: impl Into<String>) -> BackendError {
        BackendError::InvalidArgument(msg.into())
    }

    fn axis_ok(n: usize) -> bool {
        n == 1 || (n.is_power_of_two() && (2..=4096).contains(&n))
    }

    /// Number of canvases in `len` elements.
    pub fn batch(len: usize, canvas: Canvas) -> Result<usize, BackendError> {
        if canvas.is_empty() || !len.is_multiple_of(canvas.len()) {
            return Err(bad(format!(
                "buffer of {len} elements is not a whole number of {}x{} canvases",
                canvas.rows, canvas.cols
            )));
        }
        Ok(len / canvas.len())
    }

    /// Checks for `fft` / `ifft`.
    pub fn fft(
        data: &[Complex32],
        canvas: Canvas,
        tw: &Twiddles<'_>,
    ) -> Result<usize, BackendError> {
        if !axis_ok(canvas.rows) || !axis_ok(canvas.cols) || canvas.cols == 1 {
            return Err(bad(
                "canvas axes must be 1 or a power of two in 2..=4096 (cols >= 2)",
            ));
        }
        if tw.cols.len() != canvas.cols / 2 {
            return Err(bad("column twiddle table has the wrong length"));
        }
        if canvas.rows > 1 && tw.rows.len() != canvas.rows / 2 {
            return Err(bad("row twiddle table has the wrong length"));
        }
        batch(data.len(), canvas)
    }

    /// Checks for `mul_real_filter`.
    pub fn mul_real_filter(
        data: &[Complex32],
        canvas: Canvas,
        bank: &[f32],
        index: &[u32],
    ) -> Result<usize, BackendError> {
        let n = batch(data.len(), canvas)?;
        if index.len() != n {
            return Err(bad("index length differs from the number of canvases"));
        }
        if !bank.len().is_multiple_of(canvas.len()) {
            return Err(bad("filter bank is not a whole number of canvases"));
        }
        let filters = bank.len() / canvas.len();
        if index.iter().any(|&i| i as usize >= filters) {
            return Err(bad("filter index out of range"));
        }
        Ok(n)
    }

    /// Checks for `subsample`.
    pub fn subsample(
        input: &[Complex32],
        canvas: Canvas,
        spec: SubsampleSpec,
        out: &[f32],
    ) -> Result<usize, BackendError> {
        let n = batch(input.len(), canvas)?;
        if spec.factor == 0 || !spec.factor.is_power_of_two() {
            return Err(bad("subsampling factor must be a power of two"));
        }
        let rows_used = if canvas.rows == 1 {
            if spec.out_rows != 1 || spec.offset_rows != 0 {
                return Err(bad(
                    "1-D subsample must have out_rows == 1 and offset_rows == 0",
                ));
            }
            1
        } else {
            spec.factor * (spec.offset_rows + spec.out_rows - 1) + 1
        };
        let cols_used = spec.factor * (spec.offset_cols + spec.out_cols - 1) + 1;
        if spec.out_rows == 0
            || spec.out_cols == 0
            || rows_used > canvas.rows
            || cols_used > canvas.cols
        {
            return Err(bad("subsample window exceeds the canvas"));
        }
        if out.len() != n * spec.out_len() {
            return Err(bad("output buffer has the wrong length"));
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subsample_validation() {
        let canvas = Canvas::two_d(8, 8);
        let input = vec![Complex32::default(); 64];
        let spec = SubsampleSpec {
            factor: 4,
            offset_rows: 0,
            offset_cols: 1,
            out_rows: 2,
            out_cols: 1,
        };
        let out = vec![0.0; 2];
        assert_eq!(validate::subsample(&input, canvas, spec, &out), Ok(1));
        let too_far = SubsampleSpec {
            out_cols: 2,
            ..spec
        };
        assert!(validate::subsample(&input, canvas, too_far, &[0.0; 4]).is_err());
    }

    #[test]
    fn canvas_helpers() {
        assert_eq!(Canvas::one_d(16).len(), 16);
        assert_eq!(Canvas::two_d(4, 8).rows, 4);
    }
}
