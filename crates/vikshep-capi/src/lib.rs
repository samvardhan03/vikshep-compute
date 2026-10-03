//! C ABI mirror of the five Tier-1 kernels (`spec/VDS-1.md` section 14.6).
//!
//! The header `include/vikshep_backend.h` is generated from this file with
//! cbindgen (see `crates/vikshep-capi/cbindgen.toml`). A backend written in
//! C, C++/CUDA or Objective-C/Metal fills a [`VkspBackendV1`] table; Rust
//! code drives it through [`ForeignBackend`], which implements the
//! `ScatterBackend` trait. [`vksp_cpu_backend_v1`] exposes the CPU reference
//! through the same table, which the conformance runner uses to test the
//! ABI round trip (`vikshep-conformance run --backend capi-cpu`).
//!
//! # Memory ownership and alignment
//!
//! * The caller owns every buffer. A kernel reads and writes only the
//!   buffers passed to it and must not retain any pointer after it returns.
//! * Buffers are contiguous. `VkspComplex32` is two `float`s (`re`, `im`),
//!   8 bytes, alignment 4; `data` pointers must be aligned to at least 4
//!   bytes. Backends that need stronger alignment (for example for device
//!   transfers) must copy.
//! * Pointers are never null when the corresponding length is non-zero.
//!   `tw_rows` may be null when `canvas.rows == 1`.
//! * In-place kernels (`fft`, `ifft`, `mul_real_filter`, `modulus`) modify
//!   `data` only; `subsample` must not modify `input` and writes exactly
//!   `batch * out_rows * out_cols` floats to `out`.
//! * Calls are synchronous: when a kernel returns `VKSP_OK`, its results are
//!   in host memory.
//! * The table and `ctx` stay valid until `destroy` is called (if non-null);
//!   the table may be used from several threads at once only if the backend
//!   documents that.

use core::ffi::{c_char, c_void};

use vikshep_backend_api::{
    BackendError, Canvas, Capabilities, Complex32, ScatterBackend, SubsampleSpec, Twiddles,
};
use vikshep_cpu::CpuBackend;

/// Version of the VkspBackendV1 table layout.
pub const VKSP_ABI_VERSION: u32 = 1;
/// Success.
pub const VKSP_OK: i32 = 0;
/// Arguments violate the kernel contract.
pub const VKSP_INVALID_ARGUMENT: i32 = 1;
/// The device failed.
pub const VKSP_DEVICE_ERROR: i32 = 2;

/// Interleaved binary32 complex value.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VkspComplex32 {
    /// Real part.
    pub re: f32,
    /// Imaginary part.
    pub im: f32,
}

/// Canvas shape; `rows == 1` for 1-D.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VkspCanvas {
    /// Rows (1 for 1-D).
    pub rows: u32,
    /// Columns.
    pub cols: u32,
}

/// Subsampling window in output units:
/// `out[b][a][c] = input[b][factor * (offset_rows + a)][factor * (offset_cols + c)].re`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VkspSubsample {
    /// Power-of-two factor.
    pub factor: u32,
    /// Row offset (0 for 1-D).
    pub offset_rows: u32,
    /// Column offset.
    pub offset_cols: u32,
    /// Output rows (1 for 1-D).
    pub out_rows: u32,
    /// Output columns.
    pub out_cols: u32,
}

/// Backend capabilities.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VkspCapabilities {
    /// Largest log2 transform length per axis.
    pub max_log2_len: u32,
    /// Non-zero if canvases are processed in parallel.
    pub parallel: u8,
    /// Non-zero if IEEE subnormals are preserved.
    pub preserves_subnormals: u8,
}

/// Forward or inverse FFT of `batch` canvases in place.
type VkspFftFn = unsafe extern "C" fn(
    ctx: *mut c_void,
    data: *mut VkspComplex32,
    batch: usize,
    canvas: VkspCanvas,
    tw_cols: *const VkspComplex32,
    tw_rows: *const VkspComplex32,
) -> i32;

/// A backend: context pointer plus the five kernels and metadata.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VkspBackendV1 {
    /// Must equal `VKSP_ABI_VERSION`.
    pub abi_version: u32,
    /// Backend-owned context passed to every call.
    pub ctx: *mut c_void,
    /// NUL-terminated backend name, valid as long as the table.
    pub name: Option<unsafe extern "C" fn(ctx: *mut c_void) -> *const c_char>,
    /// The numerics version implemented.
    pub numerics_version: Option<unsafe extern "C" fn(ctx: *mut c_void) -> u32>,
    /// Capabilities.
    pub capabilities: Option<unsafe extern "C" fn(ctx: *mut c_void) -> VkspCapabilities>,
    /// Forward FFT of `batch` canvases in place, with forward tables
    /// (`tw_cols`: cols/2 entries; `tw_rows`: rows/2 entries or null).
    pub fft: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            data: *mut VkspComplex32,
            batch: usize,
            canvas: VkspCanvas,
            tw_cols: *const VkspComplex32,
            tw_rows: *const VkspComplex32,
        ) -> i32,
    >,
    /// Inverse FFT in place, with conjugate tables; each 1-D transform is
    /// followed by multiplication with 2^-m.
    pub ifft: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            data: *mut VkspComplex32,
            batch: usize,
            canvas: VkspCanvas,
            tw_cols: *const VkspComplex32,
            tw_rows: *const VkspComplex32,
        ) -> i32,
    >,
    /// Multiply canvas `b` by filter `index[b]` of `bank` (`n_filters` filters of
    /// rows * cols floats): (re, im) -> (re * h, im * h).
    pub mul_real_filter: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            data: *mut VkspComplex32,
            batch: usize,
            canvas: VkspCanvas,
            bank: *const f32,
            n_filters: usize,
            index: *const u32,
        ) -> i32,
    >,
    /// (re, im) -> (sqrt(re * re + im * im), +0) for len values in place.
    pub modulus:
        Option<unsafe extern "C" fn(ctx: *mut c_void, data: *mut VkspComplex32, len: usize) -> i32>,
    /// `out[b][a][c] = input[b][f * (o_r + a)][f * (o_c + c)].re`.
    pub subsample: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            input: *const VkspComplex32,
            batch: usize,
            canvas: VkspCanvas,
            spec: VkspSubsample,
            out: *mut f32,
        ) -> i32,
    >,
    /// Release `ctx`; may be null.
    pub destroy: Option<unsafe extern "C" fn(ctx: *mut c_void)>,
}

fn status(r: Result<(), BackendError>) -> i32 {
    match r {
        Ok(()) => VKSP_OK,
        Err(BackendError::InvalidArgument(_)) => VKSP_INVALID_ARGUMENT,
        Err(BackendError::Device(_)) => VKSP_DEVICE_ERROR,
    }
}

fn canvas_of(c: VkspCanvas) -> Canvas {
    Canvas::two_d(c.rows as usize, c.cols as usize)
}

const fn half(n: u32) -> usize {
    if n > 1 { (n / 2) as usize } else { 0 }
}

/// Build a slice from a possibly-null pointer.
///
/// # Safety
/// If `len > 0`, `ptr` must be valid for `len` reads of `T`.
unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if len == 0 || ptr.is_null() {
        &[]
    } else {
        // SAFETY: guaranteed by the caller.
        unsafe { core::slice::from_raw_parts(ptr, len) }
    }
}

/// Mutable variant of [`slice`].
///
/// # Safety
/// If `len > 0`, `ptr` must be valid for `len` reads and writes of `T` and
/// not aliased.
unsafe fn slice_mut<'a, T>(ptr: *mut T, len: usize) -> &'a mut [T] {
    if len == 0 || ptr.is_null() {
        &mut []
    } else {
        // SAFETY: guaranteed by the caller.
        unsafe { core::slice::from_raw_parts_mut(ptr, len) }
    }
}

// ---------------------------------------------------------------------------
// The CPU reference exposed through the C ABI.

static CPU_NAME: &[u8] = b"cpu\0";

/// # Safety
/// `ctx` must come from [`vksp_cpu_backend_v1`].
unsafe fn cpu<'a>(ctx: *mut c_void) -> &'a CpuBackend {
    // SAFETY: `ctx` is the boxed CpuBackend created by vksp_cpu_backend_v1.
    unsafe { &*ctx.cast::<CpuBackend>() }
}

unsafe extern "C" fn cpu_name(_ctx: *mut c_void) -> *const c_char {
    CPU_NAME.as_ptr().cast()
}

unsafe extern "C" fn cpu_numerics_version(ctx: *mut c_void) -> u32 {
    // SAFETY: table contract.
    unsafe { cpu(ctx) }.numerics_version()
}

unsafe extern "C" fn cpu_capabilities(ctx: *mut c_void) -> VkspCapabilities {
    // SAFETY: table contract.
    let c = unsafe { cpu(ctx) }.capabilities();
    VkspCapabilities {
        max_log2_len: c.max_log2_len,
        parallel: u8::from(c.parallel),
        preserves_subnormals: u8::from(c.preserves_subnormals),
    }
}

unsafe fn cpu_transform(
    ctx: *mut c_void,
    data: *mut VkspComplex32,
    batch: usize,
    canvas: VkspCanvas,
    tw_cols: *const VkspComplex32,
    tw_rows: *const VkspComplex32,
    inverse: bool,
) -> i32 {
    let cv = canvas_of(canvas);
    // SAFETY: the caller passes `batch` canvases and tables of the
    // documented lengths; VkspComplex32 and Complex32 are both
    // #[repr(C)] { f32, f32 }.
    let (d, tc, tr) = unsafe {
        (
            slice_mut(data.cast::<Complex32>(), batch * cv.len()),
            slice(tw_cols.cast::<Complex32>(), half(canvas.cols)),
            slice(tw_rows.cast::<Complex32>(), half(canvas.rows)),
        )
    };
    let tw = Twiddles { cols: tc, rows: tr };
    // SAFETY: table contract.
    let b = unsafe { cpu(ctx) };
    status(if inverse {
        b.ifft(d, cv, &tw)
    } else {
        b.fft(d, cv, &tw)
    })
}

unsafe extern "C" fn cpu_fft(
    ctx: *mut c_void,
    data: *mut VkspComplex32,
    batch: usize,
    canvas: VkspCanvas,
    tw_cols: *const VkspComplex32,
    tw_rows: *const VkspComplex32,
) -> i32 {
    // SAFETY: forwarded caller contract.
    unsafe { cpu_transform(ctx, data, batch, canvas, tw_cols, tw_rows, false) }
}

unsafe extern "C" fn cpu_ifft(
    ctx: *mut c_void,
    data: *mut VkspComplex32,
    batch: usize,
    canvas: VkspCanvas,
    tw_cols: *const VkspComplex32,
    tw_rows: *const VkspComplex32,
) -> i32 {
    // SAFETY: forwarded caller contract.
    unsafe { cpu_transform(ctx, data, batch, canvas, tw_cols, tw_rows, true) }
}

unsafe extern "C" fn cpu_mul_real_filter(
    ctx: *mut c_void,
    data: *mut VkspComplex32,
    batch: usize,
    canvas: VkspCanvas,
    bank: *const f32,
    n_filters: usize,
    index: *const u32,
) -> i32 {
    let cv = canvas_of(canvas);
    // SAFETY: caller contract (lengths as documented); layouts match.
    let (d, bk, ix) = unsafe {
        (
            slice_mut(data.cast::<Complex32>(), batch * cv.len()),
            slice(bank, n_filters * cv.len()),
            slice(index, batch),
        )
    };
    // SAFETY: table contract.
    status(unsafe { cpu(ctx) }.mul_real_filter(d, cv, bk, ix))
}

unsafe extern "C" fn cpu_modulus(ctx: *mut c_void, data: *mut VkspComplex32, len: usize) -> i32 {
    // SAFETY: caller contract; layouts match.
    let d = unsafe { slice_mut(data.cast::<Complex32>(), len) };
    // SAFETY: table contract.
    status(unsafe { cpu(ctx) }.modulus(d))
}

unsafe extern "C" fn cpu_subsample(
    ctx: *mut c_void,
    input: *const VkspComplex32,
    batch: usize,
    canvas: VkspCanvas,
    spec: VkspSubsample,
    out: *mut f32,
) -> i32 {
    let cv = canvas_of(canvas);
    let s = SubsampleSpec {
        factor: spec.factor as usize,
        offset_rows: spec.offset_rows as usize,
        offset_cols: spec.offset_cols as usize,
        out_rows: spec.out_rows as usize,
        out_cols: spec.out_cols as usize,
    };
    // SAFETY: caller contract; layouts match.
    let (i, o) = unsafe {
        (
            slice(input.cast::<Complex32>(), batch * cv.len()),
            slice_mut(out, batch * s.out_len()),
        )
    };
    // SAFETY: table contract.
    status(unsafe { cpu(ctx) }.subsample(i, cv, s, o))
}

unsafe extern "C" fn cpu_destroy(ctx: *mut c_void) {
    if !ctx.is_null() {
        // SAFETY: `ctx` was produced by Box::into_raw in vksp_cpu_backend_v1.
        drop(unsafe { Box::from_raw(ctx.cast::<CpuBackend>()) });
    }
}

/// The CPU reference backend as a C table. Call `destroy` when done.
#[unsafe(no_mangle)]
pub extern "C" fn vksp_cpu_backend_v1() -> VkspBackendV1 {
    VkspBackendV1 {
        abi_version: VKSP_ABI_VERSION,
        ctx: Box::into_raw(Box::new(CpuBackend::new())).cast(),
        name: Some(cpu_name),
        numerics_version: Some(cpu_numerics_version),
        capabilities: Some(cpu_capabilities),
        fft: Some(cpu_fft),
        ifft: Some(cpu_ifft),
        mul_real_filter: Some(cpu_mul_real_filter),
        modulus: Some(cpu_modulus),
        subsample: Some(cpu_subsample),
        destroy: Some(cpu_destroy),
    }
}

// ---------------------------------------------------------------------------
// Driving a foreign table from Rust.

/// A [`ScatterBackend`] backed by a [`VkspBackendV1`] table.
#[derive(Debug)]
pub struct ForeignBackend {
    table: VkspBackendV1,
    name: String,
}

// SAFETY: the table contract (see module docs) requires a backend used
// through ForeignBackend to accept calls from any thread; the Rust side never
// calls one table concurrently from two threads because ScatterBackend
// methods are invoked sequentially by the host driver.
unsafe impl Send for ForeignBackend {}
// SAFETY: see above.
unsafe impl Sync for ForeignBackend {}

impl ForeignBackend {
    /// Wrap a table.
    ///
    /// # Safety
    /// Every non-null function pointer must implement the documented
    /// contract for the table's `ctx`, and the table must stay valid until
    /// this value is dropped (which calls `destroy`).
    pub unsafe fn new(table: VkspBackendV1) -> Result<Self, BackendError> {
        if table.abi_version != VKSP_ABI_VERSION {
            return Err(BackendError::InvalidArgument(format!(
                "ABI version {} (expected {VKSP_ABI_VERSION})",
                table.abi_version
            )));
        }
        let name = match table.name {
            // SAFETY: the function returns a NUL-terminated string valid for
            // the lifetime of the table (contract).
            Some(f) => unsafe { core::ffi::CStr::from_ptr(f(table.ctx)) }
                .to_string_lossy()
                .into_owned(),
            None => String::from("foreign"),
        };
        Ok(Self { table, name })
    }

    fn missing(what: &str) -> BackendError {
        BackendError::InvalidArgument(format!("backend does not implement {what}"))
    }

    fn check(code: i32) -> Result<(), BackendError> {
        match code {
            VKSP_OK => Ok(()),
            VKSP_INVALID_ARGUMENT => {
                Err(BackendError::InvalidArgument("rejected by backend".into()))
            }
            other => Err(BackendError::Device(format!("backend status {other}"))),
        }
    }

    fn transform(
        &self,
        f: Option<VkspFftFn>,
        data: &mut [Complex32],
        canvas: Canvas,
        tw: &Twiddles<'_>,
    ) -> Result<(), BackendError> {
        let batch = vikshep_backend_api::validate::fft(data, canvas, tw)?;
        let f = f.ok_or_else(|| Self::missing("fft"))?;
        let rows_ptr = if tw.rows.is_empty() {
            core::ptr::null()
        } else {
            tw.rows.as_ptr().cast()
        };
        // SAFETY: buffers are valid for the lengths the contract requires
        // (checked by validate::fft); layouts match.
        Self::check(unsafe {
            f(
                self.table.ctx,
                data.as_mut_ptr().cast(),
                batch,
                vk_canvas(canvas)?,
                tw.cols.as_ptr().cast(),
                rows_ptr,
            )
        })
    }
}

fn vk_canvas(c: Canvas) -> Result<VkspCanvas, BackendError> {
    let conv = |n: usize| {
        u32::try_from(n).map_err(|_| BackendError::InvalidArgument("canvas too large".into()))
    };
    Ok(VkspCanvas {
        rows: conv(c.rows)?,
        cols: conv(c.cols)?,
    })
}

impl Drop for ForeignBackend {
    fn drop(&mut self) {
        if let Some(d) = self.table.destroy {
            // SAFETY: called once, as the contract allows.
            unsafe { d(self.table.ctx) };
        }
    }
}

impl ScatterBackend for ForeignBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn numerics_version(&self) -> u32 {
        // SAFETY: table contract.
        self.table
            .numerics_version
            .map_or(0, |f| unsafe { f(self.table.ctx) })
    }

    fn capabilities(&self) -> Capabilities {
        // SAFETY: table contract.
        let c = self
            .table
            .capabilities
            .map(|f| unsafe { f(self.table.ctx) });
        c.map_or(
            Capabilities {
                max_log2_len: 0,
                parallel: false,
                preserves_subnormals: false,
            },
            |c| Capabilities {
                max_log2_len: c.max_log2_len,
                parallel: c.parallel != 0,
                preserves_subnormals: c.preserves_subnormals != 0,
            },
        )
    }

    fn fft(
        &self,
        data: &mut [Complex32],
        canvas: Canvas,
        tw: &Twiddles<'_>,
    ) -> Result<(), BackendError> {
        self.transform(self.table.fft, data, canvas, tw)
    }

    fn ifft(
        &self,
        data: &mut [Complex32],
        canvas: Canvas,
        tw_conj: &Twiddles<'_>,
    ) -> Result<(), BackendError> {
        self.transform(self.table.ifft, data, canvas, tw_conj)
    }

    fn mul_real_filter(
        &self,
        data: &mut [Complex32],
        canvas: Canvas,
        bank: &[f32],
        index: &[u32],
    ) -> Result<(), BackendError> {
        let batch = vikshep_backend_api::validate::mul_real_filter(data, canvas, bank, index)?;
        let f = self
            .table
            .mul_real_filter
            .ok_or_else(|| Self::missing("mul_real_filter"))?;
        // SAFETY: lengths checked by validate::mul_real_filter; layouts match.
        Self::check(unsafe {
            f(
                self.table.ctx,
                data.as_mut_ptr().cast(),
                batch,
                vk_canvas(canvas)?,
                bank.as_ptr(),
                bank.len() / canvas.len(),
                index.as_ptr(),
            )
        })
    }

    fn modulus(&self, data: &mut [Complex32]) -> Result<(), BackendError> {
        let f = self.table.modulus.ok_or_else(|| Self::missing("modulus"))?;
        // SAFETY: `data` is valid for `data.len()` elements; layouts match.
        Self::check(unsafe { f(self.table.ctx, data.as_mut_ptr().cast(), data.len()) })
    }

    fn subsample(
        &self,
        input: &[Complex32],
        canvas: Canvas,
        spec: SubsampleSpec,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let batch = vikshep_backend_api::validate::subsample(input, canvas, spec, out)?;
        let f = self
            .table
            .subsample
            .ok_or_else(|| Self::missing("subsample"))?;
        let conv = |n: usize| {
            u32::try_from(n).map_err(|_| BackendError::InvalidArgument("too large".into()))
        };
        let s = VkspSubsample {
            factor: conv(spec.factor)?,
            offset_rows: conv(spec.offset_rows)?,
            offset_cols: conv(spec.offset_cols)?,
            out_rows: conv(spec.out_rows)?,
            out_cols: conv(spec.out_cols)?,
        };
        // SAFETY: lengths checked by validate::subsample; layouts match.
        Self::check(unsafe {
            f(
                self.table.ctx,
                input.as_ptr().cast(),
                batch,
                vk_canvas(canvas)?,
                s,
                out.as_mut_ptr(),
            )
        })
    }
}

/// The CPU reference driven through its own C table.
#[must_use]
pub fn cpu_through_c_abi() -> ForeignBackend {
    // SAFETY: vksp_cpu_backend_v1 returns a table that satisfies the
    // contract and is destroyed by ForeignBackend::drop.
    unsafe { ForeignBackend::new(vksp_cpu_backend_v1()) }
        .expect("CPU table has the current ABI version")
}

#[cfg(test)]
mod tests {
    use super::*;
    use vikshep_numerics::fft::Real;

    #[test]
    fn layouts_match() {
        assert_eq!(core::mem::size_of::<VkspComplex32>(), 8);
        assert_eq!(core::mem::align_of::<VkspComplex32>(), 4);
        assert_eq!(core::mem::size_of::<Complex32>(), 8);
        assert_eq!(core::mem::align_of::<Complex32>(), 4);
    }

    #[test]
    fn round_trip_equals_direct_cpu() {
        let foreign = cpu_through_c_abi();
        assert_eq!(foreign.name(), "cpu");
        assert_eq!(
            foreign.numerics_version(),
            vikshep_numerics::NUMERICS_VERSION
        );
        let canvas = Canvas::two_d(8, 16);
        let mut a: Vec<Complex32> = (0..canvas.len() * 3)
            .map(|i| Complex32::new((i as f32 * 0.37).fract() - 0.5, (i as f32 * 0.11).fract()))
            .collect();
        let mut b = a.clone();
        let tw = Twiddles {
            cols: f32::twiddles(4),
            rows: f32::twiddles(3),
        };
        foreign.fft(&mut a, canvas, &tw).unwrap();
        CpuBackend::new().fft(&mut b, canvas, &tw).unwrap();
        assert_eq!(a, b);
        foreign.modulus(&mut a).unwrap();
        CpuBackend::new().modulus(&mut b).unwrap();
        assert_eq!(a, b);
        // invalid arguments are reported, not executed
        let bad = Twiddles {
            cols: &[],
            rows: &[],
        };
        assert!(foreign.fft(&mut a, canvas, &bad).is_err());
    }
}
