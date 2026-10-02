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

Milestone C0 (foundation): workspace, specification, portable math,
random numbers, determinism lint, cross-platform CI. The FFT and scattering
transform arrive in milestone C1. See [`STATUS.md`](STATUS.md).

## Layout

| Crate | Role |
|---|---|
| `vikshep-detmath` | portable `exp`, `ln`, `sin`, `cos` (the only source of transcendental functions) |
| `vikshep-numerics` | SplitMix64 and Philox4x32-10 streams, OIDs; FFT from C1 |
| `vikshep-conformance` | conformance runner (binary) |
| `vikshep-scatter`, `vikshep-stats`, `vikshep-train`, `vikshep-anomaly` | scattering and Tier-2 analysis (placeholders) |
| `vikshep-backend-api`, `vikshep-cpu` | backend interface and CPU reference backend (placeholders) |
| `vikshep-py`, `vikshep-capi`, `vikshep-mcp` | Python, C and MCP interfaces (placeholders) |

## Building and testing

The toolchain is pinned in `rust-toolchain.toml`; `rustup` installs it
automatically.

```sh
cargo test --workspace
scripts/determinism-lint.sh
```

## Conformance hashes

```sh
cargo run --release -p vikshep-conformance -- hashes   # print the report (JSON)
cargo run --release -p vikshep-conformance -- check    # compare with conformance/vectors/v1
```

The report contains the SHA3-256 of one million outputs of every portable
math function and of every random stream (VDS-1 sections 4.4 and 5.6). CI
produces it on Linux x86_64, Linux AArch64, macOS arm64 and Windows x86_64
and fails if any report differs from the committed vectors.

## Licence

Code: AGPL-3.0-or-later. `spec/` and `conformance/vectors/`: CC-BY-4.0. A
commercial licence for the code is available from the copyright holder. See
[`LICENSING.md`](LICENSING.md).
