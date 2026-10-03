//! Python bindings of Vikshep compute: the `vikshep-compute` distribution,
//! imported as `vikshep_compute` (PyO3 + maturin, abi3 wheels for Python
//! 3.10 and later).
//!
//! The extension module is compiled only with the `python` feature, which
//! maturin enables (`crates/vikshep-py/pyproject.toml`); without it this
//! crate is empty, so `cargo test --workspace` never links against Python.
//!
//! Inputs are converted with numpy and copied into Rust-owned, contiguous,
//! row-major buffers before any computation, so numpy strides, memory
//! order and byte order never affect results (see `python.rs`).

#[cfg(feature = "python")]
mod python;
