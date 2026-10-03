//! Stable host API (`include/vikshep.h`).
//!
//! # Conventions
//!
//! * Every function returns a status code (`VKSP_OK` on success); details of
//!   the last failure on the calling thread are available from
//!   [`vksp_last_error`]. No Rust panic crosses the boundary.
//! * Outputs go to caller-allocated buffers. For an array output the caller
//!   passes `out`, its capacity `out_cap` (in elements) and `out_len`, which
//!   always receives the required number of elements. With `out == NULL` the
//!   call is a size query and returns `VKSP_OK` without computing; with
//!   `out_cap` too small it returns `VKSP_BUFFER_TOO_SMALL`. An empty output
//!   (for example r2 of a configuration without ratio pairs) is always
//!   computed, so its OID is written even with `out == NULL`. Text outputs
//!   work the same way in bytes: `len` receives the length without the
//!   terminating NUL and the buffer needs `len + 1` bytes.
//! * `oid_out` (may be NULL) receives the 28-character OID of the output
//!   tensor and a NUL: 29 bytes (`VKSP_OID_LEN + 1`).
//! * `backend` (may be NULL for the CPU reference `"cpu"`) names a backend
//!   registered with `vksp_register_backend` (`vikshep_backend.h`).
//! * Input arrays are read in the documented row-major order; results do not
//!   depend on their alignment beyond that of `float`/`double`.

use core::ffi::{CStr, c_char, c_void};
use std::cell::RefCell;

use vikshep_backend_api::ScatterBackend;
use vikshep_cpu::CpuBackend;
use vikshep_numerics::oid::oid;
use vikshep_numerics::provenance::{Execution, Executor, PLATFORM};
use vikshep_numerics::{NUMERICS_VERSION, TIER2_VERSION};
use vikshep_scatter::manifest::scattering_manifest;
use vikshep_scatter::reduce::{log_mean, log_mean_bytes, r2};
use vikshep_scatter::{Group, PadPolicy, ScatterConfig, ScatterOutput, Scattering};

use crate::{
    BUILTIN_BACKEND, VKSP_ABI_VERSION, VKSP_BUFFER_TOO_SMALL, VKSP_CONFORMANCE_FAILED,
    VKSP_DEVICE_ERROR, VKSP_INTERNAL_ERROR, VKSP_INVALID_ARGUMENT, VKSP_OK, VKSP_UNKNOWN_BACKEND,
    registered_backend,
};

/// Version of this host API.
pub const VKSP_API_VERSION: u32 = 1;
/// Characters of an OID (a buffer needs one more byte for the NUL).
pub const VKSP_OID_LEN: u32 = 28;
/// Characters of a hex SHA3-256 digest (a buffer needs one more byte).
pub const VKSP_SHA3_HEX_LEN: u32 = 64;
/// Pad policy: periodic boundary (power-of-two length).
pub const VKSP_PAD_CIRCULAR: u32 = 0;
/// Pad policy: zero padding with a halo of `2^J` per side.
pub const VKSP_PAD_ZERO: u32 = 1;
/// Group: keep every path.
pub const VKSP_GROUP_TRIVIAL: u32 = 0;
/// Group: 2-D, pooled over absolute orientation.
pub const VKSP_GROUP_SO2_RELATIVE: u32 = 1;
/// Executor recorded in a manifest: the local machine.
pub const VKSP_EXECUTOR_LOCAL: u32 = 0;
/// Executor recorded in a manifest: a remote worker.
pub const VKSP_EXECUTOR_CLOUD: u32 = 1;

/// A scattering configuration (VDS-1 section 14.1). Fill it with
/// `vksp_config_1d` or `vksp_config_2d`, then adjust fields if needed.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VkspScatterConfig {
    /// Must equal `sizeof(VkspScatterConfig)`.
    pub struct_size: u32,
    /// 1 or 2.
    pub dim: u32,
    /// Number of octaves `J`.
    pub j: u32,
    /// Wavelets per octave `Q` (1-D; 1 in 2-D).
    pub q: u32,
    /// Orientations `L` (2-D, even; 1 in 1-D).
    pub l: u32,
    /// Highest order, 1 or 2.
    pub max_order: u32,
    /// `VKSP_GROUP_TRIVIAL` or `VKSP_GROUP_SO2_RELATIVE`.
    pub group: u32,
    /// First scale kept by r2 (default 1).
    pub carrier_cutoff: u32,
    /// Pad policy per axis (rows first; `pad[0]` only in 1-D).
    pub pad: [u32; 2],
    /// Signal shape (rows, cols; `shape[0]` = length in 1-D).
    pub shape: [u32; 2],
}

/// Library version information.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VkspVersionInfo {
    /// Must equal `sizeof(VkspVersionInfo)`.
    pub struct_size: u32,
    /// `VKSP_API_VERSION`.
    pub api_version: u32,
    /// `VKSP_ABI_VERSION` of the backend table.
    pub backend_abi_version: u32,
    /// Tier-1 numerics version.
    pub numerics_version: u32,
    /// Tier-2 algorithm version.
    pub tier2_version: u32,
    /// Package version, static NUL-terminated string.
    pub package_version: *const c_char,
    /// Target triple of this build, static NUL-terminated string.
    pub platform: *const c_char,
}

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Record `msg` as the last error of this thread and return `code`.
pub(crate) fn fail(code: i32, msg: &str) -> i32 {
    LAST_ERROR.with(|e| *e.borrow_mut() = msg.to_string());
    code
}

/// Run `f`, converting a panic into `VKSP_INTERNAL_ERROR`.
pub(crate) fn guard(f: impl FnOnce() -> i32) -> i32 {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(code) => code,
        Err(_) => fail(VKSP_INTERNAL_ERROR, "internal error (caught panic)"),
    }
}

/// Optional NUL-terminated UTF-8 string.
///
/// # Safety
/// `p` is null or points to a NUL-terminated string.
pub(crate) unsafe fn opt_str<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        None
    } else {
        // SAFETY: caller contract.
        unsafe { CStr::from_ptr(p) }.to_str().ok()
    }
}

/// Write `value` into an optional out-parameter.
///
/// # Safety
/// `p` is null or valid for one write.
unsafe fn put<T>(p: *mut T, value: T) {
    if !p.is_null() {
        // SAFETY: caller contract.
        unsafe { p.write(value) };
    }
}

/// Copy `text` and a NUL into an optional buffer of `n + 1` bytes.
///
/// # Safety
/// `p` is null or valid for `text.len() + 1` writes.
unsafe fn put_text(p: *mut c_char, text: &str) {
    if !p.is_null() {
        // SAFETY: caller contract.
        unsafe {
            core::ptr::copy_nonoverlapping(text.as_ptr(), p.cast::<u8>(), text.len());
            p.add(text.len()).write(0);
        }
    }
}

/// Size-query protocol for an array output: writes the required length and
/// decides whether to compute. An empty output is always computed (its OID
/// is still written).
///
/// # Safety
/// `out_len` is null or valid for one write.
unsafe fn sized<T>(out: *mut T, out_cap: usize, out_len: *mut usize, needed: usize) -> Option<i32> {
    // SAFETY: caller contract.
    unsafe { put(out_len, needed) };
    if needed == 0 {
        None
    } else if out.is_null() {
        Some(VKSP_OK)
    } else if out_cap < needed {
        Some(fail(VKSP_BUFFER_TOO_SMALL, "output buffer too small"))
    } else {
        None
    }
}

/// Copy `src` to `out` (nothing when empty).
///
/// # Safety
/// `out` is valid for `src.len()` writes when `src` is not empty.
unsafe fn copy_out<T: Copy>(src: &[T], out: *mut T) {
    if !src.is_empty() {
        // SAFETY: caller contract.
        unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), out, src.len()) };
    }
}

/// Copy a text output under the size-query protocol.
///
/// # Safety
/// `buf` is null or valid for `cap` writes; `len` is null or valid.
unsafe fn text_out(buf: *mut c_char, cap: usize, len: *mut usize, text: &str) -> i32 {
    // SAFETY: caller contract.
    unsafe { put(len, text.len()) };
    if buf.is_null() {
        VKSP_OK
    } else if cap < text.len() + 1 {
        fail(VKSP_BUFFER_TOO_SMALL, "text buffer too small")
    } else {
        // SAFETY: capacity checked.
        unsafe { put_text(buf, text) };
        VKSP_OK
    }
}

fn pad_of(v: u32) -> Option<PadPolicy> {
    match v {
        VKSP_PAD_CIRCULAR => Some(PadPolicy::Circular),
        VKSP_PAD_ZERO => Some(PadPolicy::ZeroPad),
        _ => None,
    }
}

fn config_of(c: &VkspScatterConfig) -> Result<ScatterConfig, String> {
    if c.struct_size as usize != core::mem::size_of::<VkspScatterConfig>() {
        return Err("struct_size must equal sizeof(VkspScatterConfig)".into());
    }
    let dim = c.dim as usize;
    if dim != 1 && dim != 2 {
        return Err("dim must be 1 or 2".into());
    }
    let group = match c.group {
        VKSP_GROUP_TRIVIAL => Group::Trivial,
        VKSP_GROUP_SO2_RELATIVE => Group::So2Relative,
        _ => return Err("unknown group".into()),
    };
    let mut pad = Vec::with_capacity(dim);
    for &p in &c.pad[..dim] {
        pad.push(pad_of(p).ok_or("unknown pad policy")?);
    }
    let cfg = ScatterConfig {
        dim,
        group,
        j: c.j,
        q: c.q,
        l: c.l,
        max_order: c.max_order,
        pad,
        shape: c.shape[..dim].iter().map(|&n| n as usize).collect(),
        carrier_cutoff: c.carrier_cutoff,
    };
    cfg.validate().map_err(|e| e.0)?;
    Ok(cfg)
}

/// # Safety
/// `cfg` is null or points to a configuration.
unsafe fn scattering(cfg: *const VkspScatterConfig) -> Result<Scattering, i32> {
    if cfg.is_null() {
        return Err(fail(VKSP_INVALID_ARGUMENT, "cfg is null"));
    }
    // SAFETY: caller contract.
    let c = config_of(unsafe { &*cfg }).map_err(|m| fail(VKSP_INVALID_ARGUMENT, &m))?;
    Scattering::new(c).map_err(|e| fail(VKSP_INVALID_ARGUMENT, &e.0))
}

/// Run `f` with the backend named `name` (`None` or `"cpu"`: the CPU
/// reference).
pub(crate) fn with_backend<R>(
    name: Option<&str>,
    f: impl FnOnce(&dyn ScatterBackend) -> R,
) -> Result<R, i32> {
    match name {
        None | Some(BUILTIN_BACKEND) => Ok(f(&CpuBackend::new())),
        Some(n) => {
            let b = registered_backend(n).ok_or_else(|| {
                fail(
                    VKSP_UNKNOWN_BACKEND,
                    "no backend registered under this name",
                )
            })?;
            let guard = b.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            Ok(f(&*guard))
        }
    }
}

/// Input slice of `len` elements.
///
/// # Safety
/// `p` is valid for `len` reads when `len > 0`.
unsafe fn input<'a, T>(p: *const T, len: usize) -> Result<&'a [T], i32> {
    if len == 0 {
        return Ok(&[]);
    }
    if p.is_null() {
        return Err(fail(VKSP_INVALID_ARGUMENT, "input is null"));
    }
    // SAFETY: caller contract.
    Ok(unsafe { core::slice::from_raw_parts(p, len) })
}

/// Coefficients `[batch][paths][out]` rebuilt as a `ScatterOutput`.
fn coefficients(sc: &Scattering, coeffs: &[f32]) -> Result<ScatterOutput, i32> {
    let per = sc.paths().len() * sc.config().out_len();
    if coeffs.is_empty() || !coeffs.len().is_multiple_of(per) {
        return Err(fail(
            VKSP_INVALID_ARGUMENT,
            "coefficient count is not a positive multiple of paths * out_len",
        ));
    }
    Ok(ScatterOutput {
        paths: sc.paths().to_vec(),
        out_shape: sc.config().out_shape(),
        batch: coeffs.len() / per,
        coefficients: coeffs.to_vec(),
    })
}

macro_rules! tri {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(code) => return code,
        }
    };
}

static PACKAGE_VERSION: &CStr =
    match CStr::from_bytes_with_nul(concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes()) {
        Ok(s) => s,
        Err(_) => panic!("package version contains NUL"),
    };

static PLATFORM_C: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();

/// Fill `out` with version information.
///
/// # Safety
/// `out` points to a `VkspVersionInfo` whose `struct_size` is set.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vksp_version_info(out: *mut VkspVersionInfo) -> i32 {
    guard(|| {
        if out.is_null() {
            return fail(VKSP_INVALID_ARGUMENT, "out is null");
        }
        // SAFETY: caller contract.
        let size = unsafe { (*out).struct_size } as usize;
        if size != core::mem::size_of::<VkspVersionInfo>() {
            return fail(
                VKSP_INVALID_ARGUMENT,
                "struct_size must equal sizeof(VkspVersionInfo)",
            );
        }
        let platform = PLATFORM_C.get_or_init(|| {
            std::ffi::CString::new(PLATFORM).unwrap_or_else(|_| c"unknown".to_owned())
        });
        // SAFETY: caller contract.
        unsafe {
            out.write(VkspVersionInfo {
                struct_size: size as u32,
                api_version: VKSP_API_VERSION,
                backend_abi_version: VKSP_ABI_VERSION,
                numerics_version: NUMERICS_VERSION,
                tier2_version: TIER2_VERSION,
                package_version: PACKAGE_VERSION.as_ptr(),
                platform: platform.as_ptr(),
            });
        }
        VKSP_OK
    })
}

/// Static description of a status code.
#[unsafe(no_mangle)]
pub extern "C" fn vksp_status_string(status: i32) -> *const c_char {
    let s: &'static CStr = match status {
        VKSP_OK => c"ok",
        VKSP_INVALID_ARGUMENT => c"invalid argument",
        VKSP_DEVICE_ERROR => c"device error",
        VKSP_BUFFER_TOO_SMALL => c"buffer too small",
        VKSP_UNKNOWN_BACKEND => c"unknown backend",
        crate::VKSP_ALREADY_REGISTERED => c"already registered",
        VKSP_CONFORMANCE_FAILED => c"conformance failed",
        VKSP_INTERNAL_ERROR => c"internal error",
        _ => c"unknown status",
    };
    s.as_ptr()
}

/// Message of the last failure on this thread (empty if none), under the
/// text size-query protocol.
///
/// # Safety
/// `buf` is null or valid for `cap` writes; `len` is null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vksp_last_error(buf: *mut c_char, cap: usize, len: *mut usize) -> i32 {
    let msg = LAST_ERROR.with(|e| e.borrow().clone());
    // SAFETY: caller contract.
    guard(|| unsafe { text_out(buf, cap, len, &msg) })
}

/// A 1-D configuration with defaults: trivial group, order 2, carrier
/// cutoff 1, `L = 1`.
///
/// # Safety
/// `out` is valid for one write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vksp_config_1d(
    n: u32,
    j: u32,
    q: u32,
    pad: u32,
    out: *mut VkspScatterConfig,
) -> i32 {
    guard(|| {
        if out.is_null() || pad_of(pad).is_none() {
            return fail(VKSP_INVALID_ARGUMENT, "out is null or pad is unknown");
        }
        let c = VkspScatterConfig {
            struct_size: core::mem::size_of::<VkspScatterConfig>() as u32,
            dim: 1,
            j,
            q,
            l: 1,
            max_order: 2,
            group: VKSP_GROUP_TRIVIAL,
            carrier_cutoff: vikshep_scatter::config::DEFAULT_CARRIER_CUTOFF,
            pad: [pad, VKSP_PAD_CIRCULAR],
            shape: [n, 1],
        };
        // SAFETY: caller contract.
        unsafe { out.write(c) };
        VKSP_OK
    })
}

/// A 2-D configuration with defaults: order 2, carrier cutoff 1, `Q = 1`.
///
/// # Safety
/// `out` is valid for one write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vksp_config_2d(
    rows: u32,
    cols: u32,
    j: u32,
    l: u32,
    pad_rows: u32,
    pad_cols: u32,
    group: u32,
    out: *mut VkspScatterConfig,
) -> i32 {
    guard(|| {
        if out.is_null() || pad_of(pad_rows).is_none() || pad_of(pad_cols).is_none() {
            return fail(VKSP_INVALID_ARGUMENT, "out is null or a pad is unknown");
        }
        let c = VkspScatterConfig {
            struct_size: core::mem::size_of::<VkspScatterConfig>() as u32,
            dim: 2,
            j,
            q: 1,
            l,
            max_order: 2,
            group,
            carrier_cutoff: vikshep_scatter::config::DEFAULT_CARRIER_CUTOFF,
            pad: [pad_rows, pad_cols],
            shape: [rows, cols],
        };
        // SAFETY: caller contract.
        unsafe { out.write(c) };
        VKSP_OK
    })
}

/// Validate `cfg` and report its sizes: samples per signal, output paths,
/// output samples per path, and r2 ratio pairs (each optional).
///
/// # Safety
/// `cfg` points to a configuration; outputs are null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vksp_scatter_info(
    cfg: *const VkspScatterConfig,
    signal_len: *mut usize,
    n_paths: *mut usize,
    out_len: *mut usize,
    n_r2: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let sc = tri!(unsafe { scattering(cfg) });
        let probe = ScatterOutput {
            paths: sc.paths().to_vec(),
            out_shape: sc.config().out_shape(),
            batch: 0,
            coefficients: Vec::new(),
        };
        let pairs = r2(&probe, sc.config().carrier_cutoff).pairs.len();
        // SAFETY: caller contract.
        unsafe {
            put(signal_len, sc.config().signal_len());
            put(n_paths, sc.paths().len());
            put(out_len, sc.config().out_len());
            put(n_r2, pairs);
        }
        VKSP_OK
    })
}

/// OID of `len` raw bytes (29 bytes written to `oid_out`).
///
/// # Safety
/// `bytes` is valid for `len` reads; `oid_out` for 29 writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vksp_oid(bytes: *const c_void, len: usize, oid_out: *mut c_char) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let b = tri!(unsafe { input(bytes.cast::<u8>(), len) });
        if oid_out.is_null() {
            return fail(VKSP_INVALID_ARGUMENT, "oid_out is null");
        }
        // SAFETY: caller contract (29 bytes).
        unsafe { put_text(oid_out, &oid(b)) };
        VKSP_OK
    })
}

/// Scattering coefficients of `input_len / signal_len` signals (row-major,
/// back to back): `out` receives `[batch][paths][out_len]` binary32 values,
/// `oid_out` the OID of their bytes. `backend`: NULL or `"cpu"` for the CPU
/// reference, or a registered backend name.
///
/// # Safety
/// Pointers valid for the documented lengths.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vksp_scatter(
    cfg: *const VkspScatterConfig,
    backend: *const c_char,
    input_ptr: *const f32,
    input_len: usize,
    out: *mut f32,
    out_cap: usize,
    out_len: *mut usize,
    oid_out: *mut c_char,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let sc = tri!(unsafe { scattering(cfg) });
        let sig = sc.config().signal_len();
        if input_len == 0 || !input_len.is_multiple_of(sig) {
            return fail(
                VKSP_INVALID_ARGUMENT,
                "input_len is not a positive multiple of the signal length",
            );
        }
        let needed = input_len / sig * sc.paths().len() * sc.config().out_len();
        // SAFETY: caller contract.
        if let Some(code) = unsafe { sized(out, out_cap, out_len, needed) } {
            return code;
        }
        // SAFETY: caller contract.
        let x = tri!(unsafe { input(input_ptr, input_len) });
        // SAFETY: caller contract.
        let name = unsafe { opt_str(backend) };
        let res = tri!(with_backend(name, |b| sc.run(b, x)));
        let s = match res {
            Ok(s) => s,
            Err(e) => return fail(VKSP_DEVICE_ERROR, &e.to_string()),
        };
        // SAFETY: capacity checked by `sized`.
        unsafe {
            copy_out(&s.coefficients[..needed], out);
            put_text(oid_out, &oid(&s.canonical_bytes()));
        }
        VKSP_OK
    })
}

/// r2 ratios (VDS-1 section 14.8) of coefficients produced with `cfg`:
/// `[batch][pairs][out_len]` binary32 values.
///
/// # Safety
/// Pointers valid for the documented lengths.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vksp_r2(
    cfg: *const VkspScatterConfig,
    coeffs: *const f32,
    coeffs_len: usize,
    out: *mut f32,
    out_cap: usize,
    out_len: *mut usize,
    oid_out: *mut c_char,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let sc = tri!(unsafe { scattering(cfg) });
        // SAFETY: caller contract.
        let c = tri!(unsafe { input(coeffs, coeffs_len) });
        let s = tri!(coefficients(&sc, c));
        let rr = r2(&s, sc.config().carrier_cutoff);
        // SAFETY: caller contract.
        if let Some(code) = unsafe { sized(out, out_cap, out_len, rr.values.len()) } {
            return code;
        }
        // SAFETY: capacity checked.
        unsafe {
            copy_out(&rr.values, out);
            put_text(oid_out, &oid(&rr.canonical_bytes()));
        }
        VKSP_OK
    })
}

/// Log-mean fingerprint (VDS-1 section 14.9) of coefficients produced with
/// `cfg`: `[batch][paths]` binary64 values.
///
/// # Safety
/// Pointers valid for the documented lengths.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vksp_fingerprint(
    cfg: *const VkspScatterConfig,
    coeffs: *const f32,
    coeffs_len: usize,
    out: *mut f64,
    out_cap: usize,
    out_len: *mut usize,
    oid_out: *mut c_char,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let sc = tri!(unsafe { scattering(cfg) });
        // SAFETY: caller contract.
        let c = tri!(unsafe { input(coeffs, coeffs_len) });
        let s = tri!(coefficients(&sc, c));
        let fp = log_mean(&s);
        // SAFETY: caller contract.
        if let Some(code) = unsafe { sized(out, out_cap, out_len, fp.len()) } {
            return code;
        }
        // SAFETY: capacity checked.
        unsafe {
            copy_out(&fp, out);
            put_text(oid_out, &oid(&log_mean_bytes(&fp)));
        }
        VKSP_OK
    })
}

fn is_oid(s: &str) -> bool {
    s.len() == VKSP_OID_LEN as usize
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Provenance manifest (`spec/provenance.schema.json`) of a scattering run
/// of `batch` signals with `cfg`, from the input and output OIDs: canonical
/// JSON under the text size-query protocol, and its `manifest_hash` (65
/// bytes, may be NULL). `backend_name`/`backend_version` may be NULL (CPU
/// reference, this package's version); `executor` is `VKSP_EXECUTOR_LOCAL`
/// or `VKSP_EXECUTOR_CLOUD`; wall-clock times are milliseconds since the
/// Unix epoch, 0 when not recorded (never part of the hash).
///
/// # Safety
/// Strings NUL-terminated; buffers valid for their capacities.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vksp_provenance_manifest(
    cfg: *const VkspScatterConfig,
    batch: usize,
    input_oid: *const c_char,
    output_oid: *const c_char,
    backend_name: *const c_char,
    backend_version: *const c_char,
    executor: u32,
    started_unix_ms: u64,
    finished_unix_ms: u64,
    json_out: *mut c_char,
    cap: usize,
    len: *mut usize,
    hash_out: *mut c_char,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let sc = tri!(unsafe { scattering(cfg) });
        // SAFETY: caller contract.
        let (Some(i), Some(o)) = (unsafe { opt_str(input_oid) }, unsafe {
            opt_str(output_oid)
        }) else {
            return fail(
                VKSP_INVALID_ARGUMENT,
                "input_oid and output_oid are required",
            );
        };
        if !is_oid(i) || !is_oid(o) || batch == 0 {
            return fail(
                VKSP_INVALID_ARGUMENT,
                "OIDs must be 28 lowercase hex characters and batch >= 1",
            );
        }
        let executor = match executor {
            VKSP_EXECUTOR_LOCAL => Executor::Local,
            VKSP_EXECUTOR_CLOUD => Executor::Cloud,
            _ => return fail(VKSP_INVALID_ARGUMENT, "unknown executor"),
        };
        // SAFETY: caller contract.
        let name = unsafe { opt_str(backend_name) }.unwrap_or(BUILTIN_BACKEND);
        // SAFETY: caller contract.
        let version = unsafe { opt_str(backend_version) }.unwrap_or(env!("CARGO_PKG_VERSION"));
        let mut ex = Execution::local(name, version);
        ex.executor = executor;
        ex.started_unix_ms = (started_unix_ms != 0).then_some(started_unix_ms);
        ex.finished_unix_ms = (finished_unix_ms != 0).then_some(finished_unix_ms);
        let m = scattering_manifest(&sc, batch, i, o, ex);
        let (json, hash) = match (m.to_json(), m.manifest_hash()) {
            (Ok(j), Ok(h)) => (j, h),
            _ => return fail(VKSP_INTERNAL_ERROR, "manifest serialization failed"),
        };
        // SAFETY: caller contract (65 bytes).
        unsafe { put_text(hash_out, &hash) };
        // SAFETY: caller contract.
        unsafe { text_out(json_out, cap, len, &json) }
    })
}

/// Run the conformance self-test against the compiled-in vectors.
/// `subset`: `"quick"` (default when NULL) or `"full"`; `backend`: NULL or
/// `"cpu"` for the CPU reference, or a registered backend. Writes the case
/// and failure counts, and the JSON report under the text size-query
/// protocol (`report_out` may be NULL; a size query still runs the test).
/// Returns `VKSP_CONFORMANCE_FAILED` if any case failed.
///
/// # Safety
/// Strings NUL-terminated; buffers valid for their capacities.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vksp_conformance_selftest(
    subset: *const c_char,
    backend: *const c_char,
    n_cases: *mut usize,
    n_failed: *mut usize,
    report_out: *mut c_char,
    cap: usize,
    len: *mut usize,
) -> i32 {
    guard(|| {
        // SAFETY: caller contract.
        let subset = unsafe { opt_str(subset) }.unwrap_or("quick");
        // SAFETY: caller contract.
        let name = unsafe { opt_str(backend) };
        let report = match tri!(with_backend(name, |b| {
            vikshep_conformance_core::suite::selftest(b, subset)
        })) {
            Ok(r) => r,
            Err(e) => return fail(VKSP_INVALID_ARGUMENT, &e),
        };
        // SAFETY: caller contract.
        unsafe {
            put(n_cases, report.summary.cases);
            put(n_failed, report.summary.fail);
        }
        // SAFETY: caller contract.
        let code = unsafe { text_out(report_out, cap, len, &report.to_json()) };
        if code != VKSP_OK {
            return code;
        }
        if report.summary.fail > 0 {
            return fail(VKSP_CONFORMANCE_FAILED, "conformance cases failed");
        }
        VKSP_OK
    })
}

/// SHA3-256 (hex) of bytes, for tests.
#[cfg(test)]
fn sha_hex(b: &[u8]) -> String {
    vikshep_numerics::oid::hex(&vikshep_numerics::oid::sha3_256(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_1d() -> VkspScatterConfig {
        let mut c = VkspScatterConfig::default();
        // SAFETY: valid out pointer.
        let code = unsafe { vksp_config_1d(256, 4, 1, VKSP_PAD_ZERO, &mut c) };
        assert_eq!(code, VKSP_OK);
        c
    }

    #[test]
    fn scatter_matches_conformance_vector() {
        let id = "scatter/1d/256/J4-Q1-L1/zero_pad/trivial/o2/uniform";
        let (_, x) = vikshep_conformance_core::suite::scatter_case(id).unwrap();
        let c = cfg_1d();
        let mut n = 0usize;
        // size query
        // SAFETY: valid pointers.
        let code = unsafe {
            vksp_scatter(
                &c,
                core::ptr::null(),
                x.as_ptr(),
                x.len(),
                core::ptr::null_mut(),
                0,
                &mut n,
                core::ptr::null_mut(),
            )
        };
        assert_eq!(code, VKSP_OK);
        let mut out = vec![0.0f32; n];
        let mut small = vec![0.0f32; n - 1];
        // SAFETY: valid pointers.
        let code = unsafe {
            vksp_scatter(
                &c,
                core::ptr::null(),
                x.as_ptr(),
                x.len(),
                small.as_mut_ptr(),
                small.len(),
                &mut n,
                core::ptr::null_mut(),
            )
        };
        assert_eq!(code, VKSP_BUFFER_TOO_SMALL);
        let mut oid_buf = [0 as c_char; 29];
        // SAFETY: valid pointers.
        let code = unsafe {
            vksp_scatter(
                &c,
                c"cpu".as_ptr(),
                x.as_ptr(),
                x.len(),
                out.as_mut_ptr(),
                out.len(),
                &mut n,
                oid_buf.as_mut_ptr(),
            )
        };
        assert_eq!(code, VKSP_OK);
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (sha, _) = vikshep_conformance_core::suite::embedded_expected_output(id, "S").unwrap();
        assert_eq!(sha_hex(&bytes), sha);
        // SAFETY: NUL-terminated by vksp_scatter.
        let o = unsafe { CStr::from_ptr(oid_buf.as_ptr()) }
            .to_str()
            .unwrap();
        assert_eq!(o, &sha[..28]);

        let mut fp = vec![0.0f64; 64];
        let mut m = 0usize;
        // SAFETY: valid pointers.
        let code = unsafe {
            vksp_fingerprint(
                &c,
                out.as_ptr(),
                out.len(),
                fp.as_mut_ptr(),
                fp.len(),
                &mut m,
                core::ptr::null_mut(),
            )
        };
        assert_eq!(code, VKSP_OK);
        let fpb: Vec<u8> = fp[..m].iter().flat_map(|v| v.to_le_bytes()).collect();
        let (sha, _) =
            vikshep_conformance_core::suite::embedded_expected_output(id, "log_mean").unwrap();
        assert_eq!(sha_hex(&fpb), sha);
    }

    #[test]
    fn errors_are_reported_not_panicked() {
        let mut c = cfg_1d();
        c.j = 0;
        let mut n = 0usize;
        // SAFETY: valid pointers.
        let code = unsafe {
            vksp_scatter_info(
                &c,
                &mut n,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            )
        };
        assert_eq!(code, VKSP_INVALID_ARGUMENT);
        let mut buf = [0 as c_char; 128];
        let mut len = 0usize;
        // SAFETY: valid pointers.
        let code = unsafe { vksp_last_error(buf.as_mut_ptr(), buf.len(), &mut len) };
        assert_eq!(code, VKSP_OK);
        // SAFETY: NUL-terminated.
        let msg = unsafe { CStr::from_ptr(buf.as_ptr()) };
        assert!(msg.to_str().unwrap().contains('J'));
        let x = [0.0f32; 256];
        let c = cfg_1d();
        // SAFETY: valid pointers.
        let code = unsafe {
            vksp_scatter(
                &c,
                c"nope".as_ptr(),
                x.as_ptr(),
                x.len(),
                core::ptr::null_mut(),
                0,
                &mut n,
                core::ptr::null_mut(),
            )
        };
        assert_eq!(code, VKSP_OK, "a size query does not touch the backend");
        let mut out = vec![0.0f32; n];
        // SAFETY: valid pointers.
        let code = unsafe {
            vksp_scatter(
                &c,
                c"nope".as_ptr(),
                x.as_ptr(),
                x.len(),
                out.as_mut_ptr(),
                out.len(),
                &mut n,
                core::ptr::null_mut(),
            )
        };
        assert_eq!(code, VKSP_UNKNOWN_BACKEND);
        assert_eq!(guard(|| panic!("boom")), VKSP_INTERNAL_ERROR);
    }

    #[test]
    fn manifest_and_version() {
        let c = cfg_1d();
        let mut len = 0usize;
        let a = c"0123456789abcdef0123456789ab";
        // SAFETY: valid pointers.
        let code = unsafe {
            vksp_provenance_manifest(
                &c,
                1,
                a.as_ptr(),
                a.as_ptr(),
                core::ptr::null(),
                core::ptr::null(),
                VKSP_EXECUTOR_CLOUD,
                5,
                6,
                core::ptr::null_mut(),
                0,
                &mut len,
                core::ptr::null_mut(),
            )
        };
        assert_eq!(code, VKSP_OK);
        let mut buf = vec![0 as c_char; len + 1];
        let mut hash = [0 as c_char; 65];
        // SAFETY: valid pointers.
        let code = unsafe {
            vksp_provenance_manifest(
                &c,
                1,
                a.as_ptr(),
                a.as_ptr(),
                core::ptr::null(),
                core::ptr::null(),
                VKSP_EXECUTOR_CLOUD,
                5,
                6,
                buf.as_mut_ptr(),
                buf.len(),
                &mut len,
                hash.as_mut_ptr(),
            )
        };
        assert_eq!(code, VKSP_OK);
        // SAFETY: NUL-terminated.
        let json = unsafe { CStr::from_ptr(buf.as_ptr()) }.to_str().unwrap();
        // SAFETY: NUL-terminated.
        let h = unsafe { CStr::from_ptr(hash.as_ptr()) }.to_str().unwrap();
        assert!(json.contains(&format!(r#""manifest_hash":"{h}""#)));
        assert!(json.contains(r#""executor":"cloud""#) && json.contains(r#""started_unix_ms":5"#));
        let mut v = VkspVersionInfo {
            struct_size: core::mem::size_of::<VkspVersionInfo>() as u32,
            api_version: 0,
            backend_abi_version: 0,
            numerics_version: 0,
            tier2_version: 0,
            package_version: core::ptr::null(),
            platform: core::ptr::null(),
        };
        // SAFETY: valid pointer.
        assert_eq!(unsafe { vksp_version_info(&mut v) }, VKSP_OK);
        assert_eq!(v.numerics_version, NUMERICS_VERSION);
        // SAFETY: static strings.
        let platform = unsafe { CStr::from_ptr(v.platform) };
        assert_eq!(platform.to_str().unwrap(), PLATFORM);
    }
}
