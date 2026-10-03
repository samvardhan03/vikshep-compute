//! Tensor stores keyed by OID (`docs/mcp.md`).
//!
//! A tensor lives in a segment named by its OID whose first bytes are the
//! raw tensor bytes (row-major, little-endian, VDS-1 section 9.1). Dtype and
//! shape travel in the JSON messages, never in the segment. A segment may be
//! longer than the tensor (macOS rounds shared-memory sizes up to whole
//! pages; producers may allocate at least one byte); readers take the
//! number of bytes the message implies and check that they hash to the OID.
//!
//! * [`ShmStore`] (Linux, macOS): POSIX shared memory, object name
//!   `/<oid>` (29 characters, within macOS's 31-character limit). This is
//!   the object `multiprocessing.shared_memory.SharedMemory(name=oid)`
//!   opens in Python.
//! * [`FileStore`] (default on Windows, which has no POSIX shared memory;
//!   selectable everywhere): one file per tensor named `<oid>` in a
//!   directory, by default `<temp dir>/vikshep-shm` or `$VIKSHEP_SHM_DIR`.

use std::path::{Path, PathBuf};

use vikshep_numerics::oid::oid;

/// A store of tensors keyed by OID.
pub trait TensorStore {
    /// `"posix_shm"` or `"file"`.
    fn kind(&self) -> &'static str;

    /// The first `len` bytes of the segment `oid`, checked against the OID.
    fn read(&self, oid: &str, len: usize) -> Result<Vec<u8>, String>;

    /// Store `bytes` under `oid` (which must be their OID). An existing
    /// segment with the same leading bytes is accepted as is. Returns true
    /// if this call created the segment.
    fn write(&mut self, oid: &str, bytes: &[u8]) -> Result<bool, String>;

    /// Remove the segment `oid`.
    fn remove(&mut self, oid: &str) -> Result<(), String>;
}

/// True for a 28-character lowercase hex OID.
#[must_use]
pub fn is_oid(s: &str) -> bool {
    s.len() == 28
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn checked(oid_str: &str, bytes: Vec<u8>) -> Result<Vec<u8>, String> {
    if oid(&bytes) == oid_str {
        Ok(bytes)
    } else {
        Err(format!(
            "segment {oid_str}: its first {} bytes do not hash to the OID (wrong length or contents)",
            bytes.len()
        ))
    }
}

fn require_oid(s: &str) -> Result<(), String> {
    if is_oid(s) {
        Ok(())
    } else {
        Err(format!("{s:?} is not a 28-character lowercase hex OID"))
    }
}

// ---------------------------------------------------------------------------
// File-backed store

/// One file per tensor, named by OID, in a directory.
#[derive(Debug)]
pub struct FileStore {
    dir: PathBuf,
}

impl FileStore {
    /// Default directory: `$VIKSHEP_SHM_DIR` or `<temp dir>/vikshep-shm`.
    #[must_use]
    pub fn default_dir() -> PathBuf {
        std::env::var_os("VIKSHEP_SHM_DIR")
            .map_or_else(|| std::env::temp_dir().join("vikshep-shm"), PathBuf::from)
    }

    /// A store in `dir` (created if missing).
    pub fn new(dir: &Path) -> Result<Self, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        Ok(Self {
            dir: dir.to_path_buf(),
        })
    }

    /// The directory.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

impl TensorStore for FileStore {
    fn kind(&self) -> &'static str {
        "file"
    }

    fn read(&self, oid_str: &str, len: usize) -> Result<Vec<u8>, String> {
        require_oid(oid_str)?;
        let path = self.dir.join(oid_str);
        let mut bytes = std::fs::read(&path)
            .map_err(|e| format!("tensor {oid_str} not found in {}: {e}", self.dir.display()))?;
        if bytes.len() < len {
            return Err(format!(
                "tensor {oid_str}: {} bytes, expected at least {len}",
                bytes.len()
            ));
        }
        bytes.truncate(len);
        checked(oid_str, bytes)
    }

    fn write(&mut self, oid_str: &str, bytes: &[u8]) -> Result<bool, String> {
        require_oid(oid_str)?;
        let path = self.dir.join(oid_str);
        if path.exists() {
            self.read(oid_str, bytes.len())?;
            return Ok(false);
        }
        let tmp = self
            .dir
            .join(format!(".{oid_str}.{}.tmp", std::process::id()));
        std::fs::write(&tmp, bytes).map_err(|e| format!("{}: {e}", tmp.display()))?;
        if let Err(e) = std::fs::rename(&tmp, &path) {
            let _ = std::fs::remove_file(&tmp);
            if path.exists() {
                self.read(oid_str, bytes.len())?;
                return Ok(false);
            }
            return Err(format!("{}: {e}", path.display()));
        }
        Ok(true)
    }

    fn remove(&mut self, oid_str: &str) -> Result<(), String> {
        require_oid(oid_str)?;
        std::fs::remove_file(self.dir.join(oid_str)).map_err(|e| e.to_string())
    }
}

// ---------------------------------------------------------------------------
// POSIX shared memory

/// POSIX shared memory (`shm_open` name `/<oid>`).
#[cfg(unix)]
#[derive(Debug, Default)]
pub struct ShmStore;

#[cfg(unix)]
impl ShmStore {
    /// The store.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[cfg(unix)]
impl TensorStore for ShmStore {
    fn kind(&self) -> &'static str {
        "posix_shm"
    }

    fn read(&self, oid_str: &str, len: usize) -> Result<Vec<u8>, String> {
        require_oid(oid_str)?;
        checked(oid_str, shm::read(oid_str, len)?)
    }

    fn write(&mut self, oid_str: &str, bytes: &[u8]) -> Result<bool, String> {
        require_oid(oid_str)?;
        if shm::create(oid_str, bytes)? {
            Ok(true)
        } else {
            self.read(oid_str, bytes.len())?;
            Ok(false)
        }
    }

    fn remove(&mut self, oid_str: &str) -> Result<(), String> {
        require_oid(oid_str)?;
        shm::unlink(oid_str)
    }
}

#[cfg(unix)]
#[allow(unsafe_code)]
mod shm {
    use std::ffi::CString;
    use std::io::Error;

    fn name(oid: &str) -> CString {
        CString::new(format!("/{oid}")).expect("OIDs contain no NUL")
    }

    /// Open an object; `mode` applies when creating.
    fn open(name: &CString, flags: libc::c_int) -> libc::c_int {
        #[cfg(target_vendor = "apple")]
        // SAFETY: `name` is NUL-terminated; macOS declares shm_open variadic,
        // and the mode is passed as an unsigned int (default promotion).
        let fd = unsafe { libc::shm_open(name.as_ptr(), flags, 0o600 as libc::c_uint) };
        #[cfg(not(target_vendor = "apple"))]
        // SAFETY: `name` is NUL-terminated.
        let fd = unsafe { libc::shm_open(name.as_ptr(), flags, 0o600 as libc::mode_t) };
        fd
    }

    fn close(fd: libc::c_int) {
        // SAFETY: `fd` is an open descriptor owned by the caller.
        unsafe { libc::close(fd) };
    }

    /// Create `/<oid>` holding `bytes`; false if it already exists.
    pub(super) fn create(oid: &str, bytes: &[u8]) -> Result<bool, String> {
        let n = name(oid);
        let fd = open(&n, libc::O_CREAT | libc::O_EXCL | libc::O_RDWR);
        if fd < 0 {
            let e = Error::last_os_error();
            return if e.raw_os_error() == Some(libc::EEXIST) {
                Ok(false)
            } else {
                Err(format!("shm_open(/{oid}): {e}"))
            };
        }
        let size = bytes.len().max(1);
        let fail = |what: &str| {
            let e = Error::last_os_error();
            close(fd);
            // SAFETY: `n` is NUL-terminated; removes the half-made object.
            unsafe { libc::shm_unlink(n.as_ptr()) };
            Err(format!("{what}(/{oid}): {e}"))
        };
        let Ok(off) = libc::off_t::try_from(size) else {
            return fail("size");
        };
        // SAFETY: `fd` is open read-write.
        if unsafe { libc::ftruncate(fd, off) } != 0 {
            return fail("ftruncate");
        }
        // SAFETY: mapping `size` bytes of an object of exactly that size.
        let p = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return fail("mmap");
        }
        // SAFETY: the mapping is valid for `size >= bytes.len()` bytes and
        // does not overlap `bytes`; it is unmapped once.
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), p.cast::<u8>(), bytes.len());
            libc::munmap(p, size);
        }
        close(fd);
        Ok(true)
    }

    /// The first `len` bytes of `/<oid>`.
    pub(super) fn read(oid: &str, len: usize) -> Result<Vec<u8>, String> {
        let n = name(oid);
        // SAFETY: `n` is NUL-terminated.
        let fd = unsafe { libc::shm_open(n.as_ptr(), libc::O_RDONLY, 0) };
        if fd < 0 {
            return Err(format!(
                "no shared-memory segment /{oid}: {}",
                Error::last_os_error()
            ));
        }
        // SAFETY: zeroed `stat` is a valid out-parameter; `fd` is open.
        let mut st: libc::stat = unsafe { core::mem::zeroed() };
        // SAFETY: as above.
        if unsafe { libc::fstat(fd, &mut st) } != 0 {
            let e = Error::last_os_error();
            close(fd);
            return Err(format!("fstat(/{oid}): {e}"));
        }
        let size = usize::try_from(st.st_size).unwrap_or(0);
        if size < len {
            close(fd);
            return Err(format!(
                "segment /{oid}: {size} bytes, expected at least {len}"
            ));
        }
        if len == 0 {
            close(fd);
            return Ok(Vec::new());
        }
        // SAFETY: mapping `len <= size` bytes read-only.
        let p = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        close(fd);
        if p == libc::MAP_FAILED {
            return Err(format!("mmap(/{oid}): {}", Error::last_os_error()));
        }
        // SAFETY: the mapping is valid for `len` bytes until unmapped.
        let out = unsafe { core::slice::from_raw_parts(p.cast::<u8>(), len) }.to_vec();
        // SAFETY: unmapping the mapping created above, once.
        unsafe { libc::munmap(p, len) };
        Ok(out)
    }

    /// Remove `/<oid>`.
    pub(super) fn unlink(oid: &str) -> Result<(), String> {
        let n = name(oid);
        // SAFETY: `n` is NUL-terminated.
        if unsafe { libc::shm_unlink(n.as_ptr()) } == 0 {
            Ok(())
        } else {
            Err(format!("shm_unlink(/{oid}): {}", Error::last_os_error()))
        }
    }
}

/// The platform default: POSIX shared memory on Unix, the file store in
/// [`FileStore::default_dir`] on Windows.
pub fn default_store() -> Result<Box<dyn TensorStore>, String> {
    #[cfg(unix)]
    {
        Ok(Box::new(ShmStore::new()))
    }
    #[cfg(not(unix))]
    {
        Ok(Box::new(FileStore::new(&FileStore::default_dir())?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(store: &mut dyn TensorStore, tag: u8) {
        let bytes: Vec<u8> = (0..1000u32).map(|i| (i as u8) ^ tag).collect();
        let o = oid(&bytes);
        assert!(store.write(&o, &bytes).unwrap());
        assert!(
            !store.write(&o, &bytes).unwrap(),
            "second write reuses the segment"
        );
        assert_eq!(store.read(&o, bytes.len()).unwrap(), bytes);
        assert!(
            store.read(&o, 999).is_err(),
            "a wrong length does not hash to the OID"
        );
        assert!(store.read(&o, 4096 * 4).is_err());
        assert!(store.read("not-an-oid", 1).is_err());
        store.remove(&o).unwrap();
        assert!(store.read(&o, bytes.len()).is_err());
        let empty = oid(&[]);
        store.write(&empty, &[]).unwrap();
        assert_eq!(store.read(&empty, 0).unwrap(), Vec::<u8>::new());
        store.remove(&empty).unwrap();
    }

    #[test]
    fn file_store_roundtrip() {
        let dir = std::env::temp_dir().join(format!("vikshep-store-test-{}", std::process::id()));
        let mut s = FileStore::new(&dir).unwrap();
        roundtrip(&mut s, 0x5a);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn shm_store_roundtrip() {
        roundtrip(&mut ShmStore::new(), 0xa5 ^ (std::process::id() as u8));
    }
}
