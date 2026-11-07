# vikshep

The open, deterministic compute core of Vikshep in one crate: wavelet
scattering, r2 and fingerprints, weighted distance correlation, DisCo
training, calibration and Sliced Wasserstein-1 anomaly search. Results are
bit-identical on every conforming platform (specification VDS-1).

```rust
use vikshep::cpu::CpuBackend;
use vikshep::scatter::{PadPolicy, ScatterConfig, Scattering};

let x: Vec<f32> = (0..256u16).map(|i| f32::from(i % 16)).collect();
let scattering = Scattering::new(ScatterConfig::one_d(256, 4, 1, PadPolicy::ZeroPad)).unwrap();
let out = scattering.run(&CpuBackend::new(), &x).unwrap();
let oid = vikshep::numerics::oid::oid(&out.canonical_bytes()); // 28-character content ID
```

Also available as a Python package (`pip install vikshep-compute`) and as a
stable C API for C and C++ (prebuilt on the GitHub releases page).

Licence: AGPL-3.0-or-later, or a commercial licence from the copyright
holder. Source, specification and conformance vectors:
https://github.com/samvardhan03/vikshep-compute
