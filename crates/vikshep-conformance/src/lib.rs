//! VDS-1 conformance suite, re-exported from `vikshep-conformance-core`
//! (case expansion, input generation, runner, expected vectors) for the
//! `vikshep-conformance` command, which adds the backends it can test
//! (including the CPU reference through the C ABI of `vikshep-capi`).

pub use vikshep_conformance_core::*;
