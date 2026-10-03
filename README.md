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

Milestone C3a (bindings and data plane): the Python package
`vikshep-compute`, the stable C API (`include/vikshep.h`) with external
backend registration, the MCP data plane `vikshep-mcp`, and the shared
provenance manifest schema (`spec/provenance.schema.json`). Built on C2
(weighted distance correlation with its exact gradient, DisCo training,
calibration, the benchmark report, Sliced Wasserstein-1 anomaly search) and
C1 (Stockham FFT, Morlet filter banks, the order-0/1/2 cascade, r2 and
log-mean, conformance suite v1). See [`STATUS.md`](STATUS.md).

## Layout

| Path | Role |
|---|---|
| `crates/vikshep-detmath` | portable `exp`, `ln`, `sin`, `cos` (the only source of transcendental functions) |
| `crates/vikshep-numerics` | Stockham FFT and twiddle tables, SplitMix64 and Philox4x32-10 streams, OIDs, fixed-order sums, total-order sorting, RFC 8785 canonical JSON, provenance manifests |
| `crates/vikshep-scatter` | configuration, filter banks, cascade driver, pooling, r2, log-mean, path mean and std, scattering manifests |
| `crates/vikshep-backend-api` | the `ScatterBackend` trait: the five Tier-1 kernels |
| `crates/vikshep-cpu` | CPU reference backend |
| `crates/vikshep-capi` | C ABI: stable host API (`include/vikshep.h`) and backend interface with registration (`include/vikshep_backend.h`) |
| `crates/vikshep-stats` | weighted dCorr2 (exact and chunked) and its exact gradient, JSD, cuts, Asimov proxy, lambda frontier, benchmark report |
| `crates/vikshep-train` | logistic and MLP heads, DisCo training with Adam, lambda sweep, ridge calibration |
| `crates/vikshep-anomaly` | fingerprint distributions, Sliced Wasserstein-1, deterministic HNSW, detection, graph output |
| `crates/vikshep-py` | Python package `vikshep-compute` (PyO3, maturin; module `vikshep_compute`) |
| `crates/vikshep-mcp` | MCP data plane: JSON-RPC 2.0 over stdio, tensors by OID in shared memory |
| `crates/vikshep-conformance-core` | conformance suite library: cases, inputs, runner, compiled-in vectors, self-test |
| `crates/vikshep-conformance` | conformance runner (binary) |
| `conformance/` | suite definition (`cases.toml`) and expected vectors |
| `examples/` | C and C++ examples of the stable C API |
| `oracles/` | developer-run oracle generators and fixtures: Kymatio, the public Vikshep Python metric, ECMAScript number formatting |
| `docs/` | `disco_gradient.md` (derivation), `mcp.md` (data plane), `external_backends.md` (plugging in a CUDA or Metal backend) |
| `spec/` | `VDS-1.md` (the specification) and `provenance.schema.json` |

## Building and testing

The toolchain is pinned in `rust-toolchain.toml`; `rustup` installs it
automatically.

```sh
cargo test --workspace
scripts/determinism-lint.sh
```

## Quickstart: Python

Wheels are built in CI (abi3, Python 3.10 and later) for Linux x86_64 and
aarch64, macOS arm64 and x86_64, and Windows x86_64, as workflow artifacts;
nothing is published to PyPI. From a checkout (needs the Rust toolchain):

```sh
python -m pip install ./crates/vikshep-py
```

```python
import numpy as np
import vikshep_compute as vc

x = np.random.default_rng(0).standard_normal((4, 256)).astype(np.float32)
cfg = {"J": 4, "Q": 1, "pad": "zero_pad", "shape": [256]}
coeffs, coeffs_oid = vc.scatter(x, cfg)            # float32 (4, 13, 16), OID
ratios, _ = vc.r2(coeffs, cfg)                     # r2 ratios
fp, _ = vc.fingerprint(coeffs, cfg)                # log-mean fingerprint (4, 13)
manifest, manifest_hash = vc.provenance_manifest(x, cfg, coeffs_oid)

data = vc.synthetic_dataset(2000, 4, seed=1)
model = vc.train_tag(data["x"], data["y"], data["w"], data["m"], lam=1.0)
print(vc.dcorr2_w(model.predict(data["x"]), data["m"]))

index = vc.anomaly_index(vc.fingerprint_clouds(coeffs, cfg), tau=0.5, k=2)
print(vc.anomaly_query(index, vc.fingerprint_clouds(coeffs[:1], cfg))["flagged"])
print(vc.conformance_selftest("quick")["failed"])  # 0
```

Also: `calibrate`, `bench_report`, `scatter_paths`, `oid`. Inputs are
converted with numpy and copied in row-major order into buffers owned by the
library before any computation, so strides, memory order and byte order
never change a result (VDS-1 section 20.1).

## Quickstart: C and C++

```sh
cargo build --release -p vikshep-capi     # target/release/libvikshep_capi.{a,so,dylib}
```

```c
#include "vikshep.h"

VkspScatterConfig cfg;
vksp_config_1d(256, 4, 1, VKSP_PAD_ZERO, &cfg);
size_t n = 0;
vksp_scatter(&cfg, NULL, x, 256, NULL, 0, &n, NULL);          /* size query */
float *coeffs = malloc(n * sizeof *coeffs);
char oid[VKSP_OID_LEN + 1];
int32_t status = vksp_scatter(&cfg, NULL, x, 256, coeffs, n, &n, oid);
```

Every function returns a status code (`vksp_last_error` describes a
failure; no panic crosses the boundary), outputs go to caller-allocated
buffers with size queries, and external backends (for example CUDA or Metal,
built outside this repository) plug in with `vksp_register_backend`
(`docs/external_backends.md`). Complete programs: `examples/c/vikshep_example.c`
and `examples/cpp/vikshep_example.cpp`; CI builds and runs both on Linux and
macOS.

## MCP data plane

```sh
cargo build --release -p vikshep-mcp
target/release/vikshep-mcp            # JSON-RPC 2.0 on stdin/stdout
```

Tools `compute_scattering`, `reduce`, `compare` and `detect_anomaly`, with
the input fields of the public Vikshep contract (`contract/mcpSchemas.ts`).
Tensors are passed by OID in POSIX shared memory (`/<oid>`; a file per OID
on Windows); responses carry OIDs, small summaries, `numerics_version` and
the provenance `manifest_hash`. See [`docs/mcp.md`](docs/mcp.md).

## Rust

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

Tier 2 on top of the features:

```rust
use vikshep_stats::dcorr::{dcorr2_grad, weighted_dcorr2};
use vikshep_train::train::{train, GradientMode, TrainConfig};
use vikshep_anomaly::{graph::{AnomalyConfig, AnomalyIndex}, sw1::fingerprint_clouds};

let d = weighted_dcorr2(&scores, &mass, Some(&weights))?;      // exact up to 10,000 events
let (d, grad) = dcorr2_grad(&scores, &mass, Some(&weights))?;  // exact analytic gradient
let model = train(&dataset, TrainConfig { lambda: 1.0, gradient: GradientMode::Exact, ..Default::default() })?;
let index = AnomalyIndex::build(&fingerprint_clouds(&s)?, AnomalyConfig::default())?;
let graph_json = index.graph(&[])?.to_json()?;                  // nodes, edges, tau, config_digest
```

## Conformance

```sh
cargo run --release -p vikshep-conformance -- run --backend cpu   # full suite v1, JSON report
cargo run --release -p vikshep-conformance -- hashes              # C0 determinism sweeps
cargo run --release -p vikshep-conformance -- generate            # regenerate vectors (CPU reference)
```

Suite v1 has 331 cases: FFTs for N = 2..4096, the element-wise kernels,
1-D and 2-D scattering grids with adversarial inputs, the portable-math and
random-stream sweeps, and the Tier-2 operations (dCorr2, its gradient, JSD,
training runs, calibration, SW1, HNSW graphs and benchmark reports; VDS-1
section 11). CI runs it on Linux x86_64,
Linux AArch64, macOS arm64 and Windows x86_64 and fails if any case fails or
any platform's report differs. The vectors are also compiled into the
library: `vksp_conformance_selftest` (C, any registered backend) and
`vikshep_compute.conformance_selftest` (Python) run the suite without files.

## Licence

Code: AGPL-3.0-or-later. `spec/` (including `provenance.schema.json`),
`conformance/cases.toml` and `conformance/vectors/`: CC-BY-4.0. A
commercial licence for the code is available from the copyright holder. See
[`LICENSING.md`](LICENSING.md).
