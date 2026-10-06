//! C ABI of Vikshep compute.
//!
//! Two headers are generated from this crate with cbindgen:
//!
//! * `include/vikshep_backend.h` (this file, `cbindgen.toml`): the backend
//!   interface, a mirror of the five Tier-1 kernels (`spec/VDS-1.md` section
//!   14.6). A backend written in C, C++/CUDA or Objective-C/Metal fills a
//!   [`VkspBackendV1`] table; Rust code drives it through [`ForeignBackend`],
//!   which implements the `ScatterBackend` trait. [`vksp_cpu_backend_v1`]
//!   exposes the CPU reference through the same table, which the conformance
//!   runner uses to test the ABI round trip (`vikshep-conformance run
//!   --backend capi-cpu`). [`vksp_register_backend`] hands an external table
//!   to this library, after which every host function of `vikshep.h` can use
//!   it by name (`docs/external_backends.md`).
//! * `include/vikshep.h` ([`api`], `cbindgen-vikshep.toml`): the stable host
//!   API (scattering, r2, fingerprint, provenance manifest, conformance
//!   self-test, version information).
//!
//! Status codes are shared by both headers. No Rust panic crosses the
//! boundary: every exported function catches unwinding and reports
//! `VKSP_INTERNAL_ERROR`.
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

use core::ffi::{CStr, c_char, c_void};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, PoisonError};

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
/// An output buffer is too small; the required size was written.
pub const VKSP_BUFFER_TOO_SMALL: i32 = 3;
/// No backend is registered under the requested name.
pub const VKSP_UNKNOWN_BACKEND: i32 = 4;
/// A backend with that name is already registered (or the name is reserved).
pub const VKSP_ALREADY_REGISTERED: i32 = 5;
/// The conformance self-test ran and at least one case failed.
pub const VKSP_CONFORMANCE_FAILED: i32 = 6;
/// Internal error (a caught Rust panic); nothing unwound across the boundary.
pub const VKSP_INTERNAL_ERROR: i32 = 7;

pub mod api;

/// Run `f`, converting a panic into `VKSP_INTERNAL_ERROR`.
fn shield(f: impl FnOnce() -> i32) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or(VKSP_INTERNAL_ERROR)
}

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
    /// Non-zero if IEEE subnormals are preserved; every VDS-1.1 backend
    /// flushes them (section 8.1) and reports 0.
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
    shield(|| unsafe { cpu_transform(ctx, data, batch, canvas, tw_cols, tw_rows, false) })
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
    shield(|| unsafe { cpu_transform(ctx, data, batch, canvas, tw_cols, tw_rows, true) })
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
    shield(|| {
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
    })
}

unsafe extern "C" fn cpu_modulus(ctx: *mut c_void, data: *mut VkspComplex32, len: usize) -> i32 {
    shield(|| {
        // SAFETY: caller contract; layouts match.
        let d = unsafe { slice_mut(data.cast::<Complex32>(), len) };
        // SAFETY: table contract.
        status(unsafe { cpu(ctx) }.modulus(d))
    })
}

unsafe extern "C" fn cpu_subsample(
    ctx: *mut c_void,
    input: *const VkspComplex32,
    batch: usize,
    canvas: VkspCanvas,
    spec: VkspSubsample,
    out: *mut f32,
) -> i32 {
    shield(|| {
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
    })
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

// ---------------------------------------------------------------------------
// Registry of external backends.

type Registered = Arc<Mutex<ForeignBackend>>;

static REGISTRY: Mutex<BTreeMap<String, Registered>> = Mutex::new(BTreeMap::new());

/// Name reserved for the built-in CPU reference.
pub const BUILTIN_BACKEND: &str = "cpu";

fn registry() -> std::sync::MutexGuard<'static, BTreeMap<String, Registered>> {
    REGISTRY.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The backend registered under `name`, if any. Calls through the returned
/// value must hold its lock, which serializes them.
#[must_use]
pub fn registered_backend(name: &str) -> Option<Registered> {
    registry().get(name).cloned()
}

/// Names of the registered external backends, sorted.
#[must_use]
pub fn registered_backends() -> Vec<String> {
    registry().keys().cloned().collect()
}

/// Check a table and read its name without taking ownership.
///
/// # Safety
/// As for [`vksp_register_backend`].
unsafe fn validate_table(t: &VkspBackendV1) -> Result<String, (i32, String)> {
    let invalid = |m: &str| Err((VKSP_INVALID_ARGUMENT, m.to_string()));
    if t.abi_version != VKSP_ABI_VERSION {
        return invalid("abi_version differs from VKSP_ABI_VERSION");
    }
    if t.fft.is_none()
        || t.ifft.is_none()
        || t.mul_real_filter.is_none()
        || t.modulus.is_none()
        || t.subsample.is_none()
    {
        return invalid("all five kernels must be non-null");
    }
    let Some(nv) = t.numerics_version else {
        return invalid("numerics_version must be non-null");
    };
    // SAFETY: table contract (the caller vouches for every pointer).
    let v = unsafe { nv(t.ctx) };
    if v != vikshep_numerics::NUMERICS_VERSION {
        return invalid("backend implements a different numerics_version");
    }
    let Some(name_fn) = t.name else {
        return invalid("name must be non-null");
    };
    // SAFETY: table contract.
    let p = unsafe { name_fn(t.ctx) };
    if p.is_null() {
        return invalid("name returned null");
    }
    // SAFETY: the name is NUL-terminated and valid as long as the table.
    let name = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    if name.is_empty() || name.len() > 64 {
        return invalid("name must have 1 to 64 bytes");
    }
    if name == BUILTIN_BACKEND {
        return Err((VKSP_ALREADY_REGISTERED, "the name cpu is reserved".into()));
    }
    Ok(name)
}

/// Hand an external backend to this library. The table is copied; on
/// success the library owns it and calls `destroy` (if non-null) when the
/// backend is unregistered and no call is using it. On failure the caller
/// keeps ownership. The backend is then available by its `name()` to every
/// host function that takes a backend name (`vikshep.h`). Calls into one
/// registered backend are serialized by the library.
///
/// Requirements: `abi_version == VKSP_ABI_VERSION`, all five kernels and
/// `name`/`numerics_version` non-null, `numerics_version()` equal to the
/// library's, a name of 1 to 64 bytes other than `cpu` and not yet registered.
///
/// # Safety
/// `table` must point to a table whose non-null function pointers implement
/// the documented contract for its `ctx`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vksp_register_backend(table: *const VkspBackendV1) -> i32 {
    api::guard(|| {
        if table.is_null() {
            return api::fail(VKSP_INVALID_ARGUMENT, "table is null");
        }
        // SAFETY: non-null pointer to a table (caller contract).
        let t = unsafe { *table };
        // SAFETY: caller contract.
        let name = match unsafe { validate_table(&t) } {
            Ok(n) => n,
            Err((code, msg)) => return api::fail(code, &msg),
        };
        let mut reg = registry();
        if reg.contains_key(&name) {
            return api::fail(
                VKSP_ALREADY_REGISTERED,
                "a backend with this name is registered",
            );
        }
        // SAFETY: validated above; ownership passes to the registry.
        match unsafe { ForeignBackend::new(t) } {
            Ok(fb) => {
                reg.insert(name, Arc::new(Mutex::new(fb)));
                VKSP_OK
            }
            Err(e) => api::fail(VKSP_INVALID_ARGUMENT, &e.to_string()),
        }
    })
}

/// Remove a registered backend. Its `destroy` runs once no call is using it.
///
/// # Safety
/// `name` must be a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vksp_unregister_backend(name: *const c_char) -> i32 {
    api::guard(|| {
        // SAFETY: caller contract.
        let Some(n) = (unsafe { api::opt_str(name) }) else {
            return api::fail(VKSP_INVALID_ARGUMENT, "name is null or not UTF-8");
        };
        match registry().remove(n) {
            Some(_) => VKSP_OK,
            None => api::fail(
                VKSP_UNKNOWN_BACKEND,
                "no backend registered under this name",
            ),
        }
    })
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

    static DESTROYED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    unsafe extern "C" fn fwd_name(_ctx: *mut c_void) -> *const c_char {
        c"test-forward".as_ptr()
    }

    unsafe extern "C" fn counting_destroy(ctx: *mut c_void) {
        DESTROYED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // SAFETY: ctx comes from vksp_cpu_backend_v1.
        unsafe { cpu_destroy(ctx) };
    }

    #[test]
    fn register_use_and_unregister() {
        let mut t = vksp_cpu_backend_v1();
        t.name = Some(fwd_name);
        t.destroy = Some(counting_destroy);
        // rejected tables stay with the caller
        let mut bad = t;
        bad.abi_version = 99;
        // SAFETY: valid table pointers.
        unsafe {
            assert_eq!(
                vksp_register_backend(core::ptr::null()),
                VKSP_INVALID_ARGUMENT
            );
            assert_eq!(vksp_register_backend(&bad), VKSP_INVALID_ARGUMENT);
            let mut no_fft = t;
            no_fft.fft = None;
            assert_eq!(vksp_register_backend(&no_fft), VKSP_INVALID_ARGUMENT);
            let reserved = vksp_cpu_backend_v1();
            assert_eq!(vksp_register_backend(&reserved), VKSP_ALREADY_REGISTERED);
            cpu_destroy(reserved.ctx);
            assert_eq!(vksp_register_backend(&t), VKSP_OK);
            assert_eq!(vksp_register_backend(&t), VKSP_ALREADY_REGISTERED);
        }
        assert_eq!(DESTROYED.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(registered_backends().contains(&"test-forward".to_string()));
        // scattering through the registered table equals the CPU reference
        let mut c = api::VkspScatterConfig::default();
        let x: Vec<f32> = (0..256)
            .map(|i| ((i * 37) % 101) as f32 / 101.0 - 0.5)
            .collect();
        let mut n = 0usize;
        let (mut oa, mut ob) = ([0 as c_char; 29], [0 as c_char; 29]);
        // SAFETY: valid pointers.
        unsafe {
            assert_eq!(
                api::vksp_config_1d(256, 3, 2, api::VKSP_PAD_CIRCULAR, &mut c),
                VKSP_OK
            );
            api::vksp_scatter(
                &c,
                core::ptr::null(),
                x.as_ptr(),
                256,
                core::ptr::null_mut(),
                0,
                &mut n,
                core::ptr::null_mut(),
            );
            let mut out = vec![0.0f32; n];
            assert_eq!(
                api::vksp_scatter(
                    &c,
                    core::ptr::null(),
                    x.as_ptr(),
                    256,
                    out.as_mut_ptr(),
                    n,
                    &mut n,
                    oa.as_mut_ptr()
                ),
                VKSP_OK
            );
            assert_eq!(
                api::vksp_scatter(
                    &c,
                    c"test-forward".as_ptr(),
                    x.as_ptr(),
                    256,
                    out.as_mut_ptr(),
                    n,
                    &mut n,
                    ob.as_mut_ptr()
                ),
                VKSP_OK
            );
        }
        assert_eq!(oa, ob);
        // SAFETY: valid strings.
        unsafe {
            assert_eq!(vksp_unregister_backend(c"test-forward".as_ptr()), VKSP_OK);
            assert_eq!(
                vksp_unregister_backend(c"test-forward".as_ptr()),
                VKSP_UNKNOWN_BACKEND
            );
        }
        assert_eq!(DESTROYED.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}
