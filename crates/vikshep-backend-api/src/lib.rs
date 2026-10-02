//! Backend interface: the Tier-1 operations (FFT, Fourier-domain filter
//! multiplication, modulus, subsampling) that every VDS-1 backend implements
//! bit-exactly. The CPU reference lives in `vikshep-cpu`; GPU backends are
//! separate, private implementations of the same interface.
//!
//! Placeholder created by milestone C0; no functionality yet.

#[cfg(test)]
mod tests {
    #[test]
    fn crate_identity() {
        assert_eq!(env!("CARGO_PKG_NAME"), "vikshep-backend-api");
    }
}
