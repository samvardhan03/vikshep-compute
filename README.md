# vikshep-compute

The open, deterministic compute core of Vikshep.

Vikshep turns detector-physics data (Geant4 simulation output, detector
images) into fixed, theorem-backed features using the wavelet scattering
transform: convolution with analytic Morlet wavelets, complex modulus, and
Gaussian low-pass averaging, repeated to orders 0, 1 and 2. Nothing in the
feature extractor is learned.

The goal is determinism: the same input and configuration produce the same
bytes on every supported machine. This repository is the open reference for
that goal:

- **Specification:** [`spec/VDS-1.md`](spec/VDS-1.md), the Vikshep Determinism
  Specification, version 1.
- **CPU reference implementation:** the Rust crates in `crates/`.
- **Conformance suite:** `vikshep-conformance` and the vectors in
  [`conformance/vectors/`](conformance/vectors/).

Other backends (for example GPU backends) implement the same specification
and must reproduce this reference bit for bit.

## Tier model

| Tier | Guarantee | Contents |
|---|---|---|
| 1 | Bit-exact on every conforming backend | FFT and inverse FFT, multiplication by a real Fourier-domain filter, complex modulus, subsampling, and therefore the scattering coefficients S0, S1, S2 |
| 2 | Bit-exact on every platform, because it always runs in this Rust CPU code | filter and twiddle construction, r2 ratios, reductions, statistics, training, calibration, anomaly search, reports |
| 3 | Reproducible only on the same build and platform; out of scope of conformance | external simulators such as Geant4 (provenance only) |

## Status

Milestone C1 (reference scattering core): Stockham FFT, Kymatio-parameterized
Morlet filter banks, the order-0/1/2 cascade with SO(2) pooling, r2 and
log-mean reductions, the backend interface with its C ABI, and conformance
suite v1. Next: C2. See [`STATUS.md`](STATUS.md).

## Layout

| Path | Role |
|---|---|
| `crates/vikshep-detmath` | portable `exp`, `ln`, `sin`, `cos` (the only source of transcendental functions) |
| `crates/vikshep-numerics` | Stockham FFT and twiddle tables, SplitMix64 and Philox4x32-10 streams, OIDs, fixed-order sums |
| `crates/vikshep-scatter` | configuration, filter banks, cascade driver, pooling, r2, log-mean, provenance manifests |
| `crates/vikshep-backend-api` | the `ScatterBackend` trait: the five Tier-1 kernels |
| `crates/vikshep-cpu` | CPU reference backend |
| `crates/vikshep-capi` | C ABI of the backend interface; header in `include/vikshep_backend.h` |
| `crates/vikshep-conformance` | conformance runner (binary) |
| `crates/vikshep-stats`, `-train`, `-anomaly`, `-py`, `-mcp` | placeholders for later milestones |
| `conformance/` | suite definition (`cases.toml`) and expected vectors |
| `oracles/` | Kymatio oracle script and fixtures (developer-run) |
| `spec/VDS-1.md` | the specification |

## Building and testing

The toolchain is pinned in `rust-toolchain.toml`; `rustup` installs it
automatically.

```sh
cargo test --workspace
scripts/determinism-lint.sh
```

## Example

```rust
use vikshep_cpu::CpuBackend;
use vikshep_scatter::{Group, PadPolicy, ScatterConfig, Scattering};
use vikshep_scatter::reduce::{log_mean, r2};

let cfg = ScatterConfig::two_d(64, 64, 3, 8, [PadPolicy::Circular; 2], Group::Trivial);
let sc = Scattering::new(cfg.clone())?;
let image = vec![0.0f32; 64 * 64];
let s = sc.run(&CpuBackend::new(), &image)?;   // [1, 217, 8, 8] binary32
let ratios = r2(&s, cfg.carrier_cutoff);
let fingerprint = log_mean(&s);
```

## Conformance

```sh
cargo run --release -p vikshep-conformance -- run --backend cpu   # full suite v1, JSON report
cargo run --release -p vikshep-conformance -- hashes              # C0 determinism sweeps
cargo run --release -p vikshep-conformance -- generate            # regenerate vectors (CPU reference)
```

Suite v1 has 313 cases: FFTs for N = 2..4096, the element-wise kernels,
1-D and 2-D scattering grids with adversarial inputs, and the portable-math
and random-stream sweeps (VDS-1 section 11). CI runs it on Linux x86_64,
Linux AArch64, macOS arm64 and Windows x86_64 and fails if any case fails or
any platform's report differs.

## Licence

Code: AGPL-3.0-or-later. `spec/`, `conformance/cases.toml` and
`conformance/vectors/`: CC-BY-4.0. A
commercial licence for the code is available from the copyright holder. See
[`LICENSING.md`](LICENSING.md).
