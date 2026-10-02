//! CPU reference backend: the normative implementation of the Tier-1
//! operations of `spec/VDS-1.md` against which every other backend is
//! checked bit for bit.
//!
//! Placeholder created by milestone C0; no functionality yet.

#[cfg(test)]
mod tests {
    #[test]
    fn crate_identity() {
        assert_eq!(env!("CARGO_PKG_NAME"), "vikshep-cpu");
    }
}
