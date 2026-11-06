//! The open, deterministic compute core of Vikshep in one crate.
//!
//! Vikshep turns detector-physics data (Geant4 simulation output, detector
//! images) into fixed features with the wavelet scattering transform, and
//! every result is bit-identical on every conforming platform
//! (specification VDS-1). This crate re-exports the crates of
//! [vikshep-compute](https://github.com/samvardhan03/vikshep-compute):
//!
//! | Module | Crate | Contents |
//! |---|---|---|
//! | [`scatter`] | `vikshep-scatter` | configuration, filter banks, the order-0/1/2 cascade, r2, fingerprints |
//! | [`cpu`] | `vikshep-cpu` | CPU reference backend |
//! | [`backend_api`] | `vikshep-backend-api` | the `ScatterBackend` trait other backends implement |
//! | [`stats`] | `vikshep-stats` | weighted dCorr2 and its gradient, JSD, the benchmark report |
//! | [`train`] | `vikshep-train` | DisCo training, calibration |
//! | [`anomaly`] | `vikshep-anomaly` | Sliced Wasserstein-1, deterministic HNSW |
//! | [`numerics`] | `vikshep-numerics` | FFT, RNG streams, OIDs, canonical JSON, provenance |
//! | [`detmath`] | `vikshep-detmath` | portable transcendental functions |
//!
//! ```
//! use vikshep::cpu::CpuBackend;
//! use vikshep::scatter::{PadPolicy, ScatterConfig, Scattering};
//!
//! // One 256-sample signal, scattered to order 2 with J = 4, Q = 1.
//! let x: Vec<f32> = (0..256u16).map(|i| f32::from(i % 16)).collect();
//! let scattering = Scattering::new(ScatterConfig::one_d(256, 4, 1, PadPolicy::ZeroPad)).unwrap();
//! let out = scattering.run(&CpuBackend::new(), &x).unwrap();
//!
//! // The same bytes on every conforming platform; the OID names them.
//! let oid = vikshep::numerics::oid::oid(&out.canonical_bytes());
//! assert_eq!(oid.len(), 28);
//! ```
//!
//! Python: `pip install vikshep-compute`. C and C++: the stable C API
//! (`vikshep.h`), prebuilt for each release on the GitHub releases page.

pub use vikshep_anomaly as anomaly;
pub use vikshep_backend_api as backend_api;
pub use vikshep_cpu as cpu;
pub use vikshep_detmath as detmath;
pub use vikshep_numerics as numerics;
pub use vikshep_scatter as scatter;
pub use vikshep_stats as stats;
pub use vikshep_train as train;
