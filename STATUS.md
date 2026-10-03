# Status

Current milestone: **C3a** (Python wheels, C ABI, open MCP data plane,
provenance schema). Next: **C3b** in `samvardhan03/Vikshep`, and **B1**,
**B2**, **D1** in the private engine repository.

## C0 (complete)

Gate met: CI run [37069011872](https://github.com/samvardhan03/vikshep-compute/actions/runs/37069011872)
on `main` (commit 9fb357f) passed every job, including `test` and `hashes` on
`ubuntu-latest`, `ubuntu-24.04-arm`, `macos-14` and `windows-latest`, and the
`hash-diff` job. `ubuntu-24.04-arm` is available to this repository.

C0 delivered the workspace (Rust 1.97.0, edition 2024), `spec/VDS-1.md`
sections 1 to 13, `vikshep-detmath` (Option A: `libm =0.2.16`, no
features), SplitMix64 and Philox4x32-10 with cited known-answer tests, the
determinism lint, and the cross-platform sweep hashes.

## C1 (complete)

Gate met: PR [#1](https://github.com/samvardhan03/vikshep-compute/pull/1)
merged (commit 250e058). CI run
[37104259050](https://github.com/samvardhan03/vikshep-compute/actions/runs/37104259050)
on the PR head and run
[37104411116](https://github.com/samvardhan03/vikshep-compute/actions/runs/37104411116)
on `main` passed every job: suite v1 313 / 313 on Linux x86_64, Linux arm64,
macOS arm64 and Windows x86_64, CPU and C ABI, with byte-identical reports
(SHA-256 `5a527b3b1f39966f4946e61f3112b223e1be5e8482e3bc28150c9a48f8f01444`).
The first run (37103963283) had the same identical reports; its `hash-diff`
failed only because the all-pass check grepped for `"fail": 0,`, now parsed
with `jq`.

C1 delivered the normative Stockham FFT, the Kymatio-parameterized Morlet
filter banks, the order 0/1/2 cascade with SO(2) pooling, r2 and the
log-mean fingerprint, RFC 8785 manifests, the `ScatterBackend` trait and its
C ABI, the CPU reference backend, conformance suite v1 (313 cases), the
Kymatio oracle (20 configurations within the stated tolerance) and the exact
power-of-two homogeneity property. Open item carried forward: per-runner
throughput from the manual `bench` CI job.

## C2 (complete)

Gate met: PR [#2](https://github.com/samvardhan03/vikshep-compute/pull/2)
merged (commit 6610961). CI run
[37122315966](https://github.com/samvardhan03/vikshep-compute/actions/runs/37122315966)
on the PR head and run
[37122321702](https://github.com/samvardhan03/vikshep-compute/actions/runs/37122321702)
on `main` passed every job, including the suite (331 cases, Tier-2 outputs
and benchmark reports included) on all four OS legs and `hash-diff`.

C2 delivered `vikshep-stats` (weighted dCorr2 exact and chunked, its exact
gradient, JSD, the Asimov proxy, the lambda frontier and the benchmark
report with its TRUE/FALSE win condition), `vikshep-train` (logistic and MLP
heads, DisCo training with Adam, ridge calibration), `vikshep-anomaly` (SW1,
deterministic HNSW, graph output contract), parity with the public Python
metric, 18 Tier-2 conformance cases and VDS-1 sections 15 to 19
(`tier2_version = 1`).

## C3a checklist

### A. Python package (`crates/vikshep-py`, PyO3 0.29 + maturin)
- [x] Distribution `vikshep-compute`, import `vikshep_compute` (extension module `vikshep_compute._core`); the name `vikshep` is not used
- [x] abi3 wheels for Python 3.10 and later; numpy arrays plus OIDs: `scatter(batch, config) -> (coeffs, oid)`, `r2`, `fingerprint`, `dcorr2_w(x, y, w)`, `train_tag`, `calibrate`, `anomaly_index`, `anomaly_query`, `bench_report`, `provenance_manifest`, `conformance_selftest(subset="quick")`; also `scatter_paths`, `fingerprint_clouds`, `synthetic_dataset`, `oid`, `TaggerModel`, `AnomalyIndex`
- [x] Inputs converted with numpy and copied element by element into Rust-owned contiguous row-major buffers before compute (VDS-1 section 20.1); tested with Fortran-ordered, strided, big-endian, float64 and list inputs (identical OIDs)
- [x] pytest: byte equality with `conformance/vectors/v1` for 6 scattering cases (S, r2, log_mean), the 4 training cases (model, scores) and the 2 benchmark reports (report.json, report.md); manifests validated against the schema
- [x] CI `wheels` job: Linux x86_64 and aarch64 (manylinux_2_28), macOS arm64 and x86_64 (cross-compiled on arm64), Windows x86_64; wheels kept as artifacts, not published; pytest on every target that can run natively
- [ ] Wheels build and pytest passes on all five targets in CI (first run on the C3a PR)

### B. C ABI (`crates/vikshep-capi`)
- [x] Stable functions in `include/vikshep.h` (cbindgen): `vksp_scatter`, `vksp_r2`, `vksp_fingerprint`, `vksp_provenance_manifest`, `vksp_conformance_selftest`, `vksp_version_info`, plus `vksp_config_1d/2d`, `vksp_scatter_info`, `vksp_oid`, `vksp_last_error`, `vksp_status_string`
- [x] Status codes, no panics across the boundary (`catch_unwind` on every export, including the CPU kernel table), caller-allocated output buffers with size queries
- [x] `examples/c/vikshep_example.c` and `examples/cpp/vikshep_example.cpp`, compiled with `-Werror -pedantic` and run in CI (`c-abi` job, Linux and macOS)
- [x] `vksp_register_backend` / `vksp_unregister_backend` in `include/vikshep_backend.h`: a host is handed an external `VkspBackendV1` and uses it by name in every host function, including the conformance self-test; `docs/external_backends.md` documents how a proprietary CUDA or Metal backend plugs in without this repository containing it
- [ ] C and C++ examples build and run in CI (first run on the C3a PR)

### C. MCP data plane (`crates/vikshep-mcp`)
- [x] JSON-RPC 2.0 over stdio, line-delimited; MCP handshake (`initialize`, `notifications/initialized`, `ping`, `tools/list`, `tools/call`); tools `compute_scattering`, `reduce`, `compare`, `detect_anomaly` with the contract's input field names
- [x] Inputs by OID from POSIX shared memory (name `/<oid>`); outputs written to new segments named by their OIDs; responses carry OIDs and small JSON summaries; every segment read is checked against its OID
- [x] Windows: file-backed store, one file per OID in `<temp dir>/vikshep-shm` (or `$VIKSHEP_SHM_DIR`), documented in `docs/mcp.md`; selectable on every platform with `--store file`
- [x] Integration test (`tests/stdio.rs`): spawns the binary, writes a conformance input tensor, and checks that the output OIDs of `compute_scattering` and `reduce` equal the conformance vectors (1-D zero-padded and 2-D `so2_relative` cases), plus `compare`, `detect_anomaly`, errors and segment cleanup; default store (shared memory on Linux and macOS, files on Windows) and file store
- [x] Every tool result includes `numerics_version`, `tier2_version` and the provenance `manifest_hash` (pinned in the test, platform independent)
- [ ] MCP integration test passes in CI on Linux and macOS (shared memory) and Windows (file store) (first run on the C3a PR)

### D. Provenance manifest
- [x] `spec/provenance.schema.json` (JSON Schema 2020-12): inputs (OID, dtype, shape), config, filter-bank SHA3, `numerics_version`, `tier2_version`, seeds, outputs, execution (backend name and version, platform triple, executor `local|cloud`, wall clock) and `manifest_hash`
- [x] Hashed subset specified (VDS-1 section 9.3): everything except `execution` and `manifest_hash`, so wall-clock times, backend, platform and executor never change the hash
- [x] One implementation (`vikshep_numerics::provenance`) behind the Rust, C, Python and MCP manifests

### E. Docs and status
- [x] README quickstart for Python and C/C++ (and the MCP binary); VDS-1 section 20 (bindings and data plane), sections 9.3, 11.2, 11.3, 14.9, 14.10 updated; `docs/mcp.md`, `docs/external_backends.md`
- [x] Nothing tagged, nothing published
- [x] STATUS updated

### Gate C3a
- [x] Locally: wheel builds (Linux x86_64) and pytest passes (21 tests, Python 3.10 and 3.11); C and C++ examples build and run (Linux); MCP integration test passes with POSIX shared memory and the file store (Linux) and with the file store under Wine (Windows build)
- [ ] Wheels build for all five targets in CI
- [ ] pytest byte equality with the conformance vectors passes in CI
- [ ] C and C++ examples build and run in CI
- [ ] MCP integration test passes on Linux and macOS, file-backed fallback passes on Windows, in CI
- [x] Provenance schema committed
- [ ] PR open

## Contract notes (`contract/mcpSchemas.ts`, public Vikshep commit 7882dfc)

Field names are mirrored exactly. What numerics_version 1 cannot support is
rejected with a tool error rather than renamed:

| Field | Contract | Supported here |
|---|---|---|
| `ScatterCfg.dim` | `"1"`, `"2"`, `"3"` | `"1"`, `"2"`; `"3"` rejected (no 3-D scattering in VDS-1) |
| `ScatterCfg.order` | 1 to 3 | 1, 2; 3 rejected |
| `ScatterCfg.group` | `trivial`, `so2`, `so3` | `trivial`; `so2` = `so2_relative` (2-D only); `so3` rejected |
| `ScatterCfg.J` | 1 to 14 | 1 to 11 (canvas limit 4096) |
| `ScatterCfg.L` | 1 to 16 | 1 in 1-D; even values 2 to 16 in 2-D |
| `ScatterCfg.Q` | 1 to 32 | 1 to 32 in 1-D; 1 in 2-D |
| `ScatterCfg.dim_shape` | positive integers | 1-D: `[]` (one signal of `signal_len`) or `[n]` (a batch); 2-D: `[rows, cols]` |
| pad policy | no field | server option `--pad` (default `zero_pad`), recorded in each manifest |
| `ReduceInput.method` | `mean`, `std`, `log_mean`, `ratio` | all four (VDS-1 sections 14.8, 14.9) |
| `CompareInput`, `DetectAnomalyInput` | `query_oid`, `k`, `tau` | as specified; the reference library is the session's earlier coefficient tensors with the same configuration (VDS-1 section 20.3) |

The contract file defines input schemas, not tool names. The tools are
named `compute_scattering`, `reduce`, `compare`, `detect_anomaly`; the
public agent's recipes call the reduce step `reduce_scattering`, which the
server accepts as an alias. `reduce`, `compare` and `detect_anomaly` accept
OIDs produced by the same server process (they need the configuration and
shape the server recorded); `IngestG4Input`, `WellSliceInput` and
`FeaturizeWellInput` belong to the ingest side and are not part of this
data plane.

## Fix carried in C3a

A scattering conformance case whose group is `so2_relative` failed when run
on its own (`vikshep-conformance run --only ...`): the C1 runner took its
input from the cached coefficients of the trivial-group case. The input is
now drawn from the trivial variant's stream by rule (VDS-1 section 11.2);
the full report and every vector are unchanged, and a test runs a pooled
case in isolation. The suite library moved to `vikshep-conformance-core`
(the `vikshep-conformance` command is unchanged) so that the C ABI and the
Python module can run the self-test.

## Decisions taken in C3a

| Decision | Outcome |
|---|---|
| Manifest hash | SHA3-256 of the canonical manifest without `execution` and `manifest_hash`: identical for the same computation on every backend, platform and executor |
| Wall clock | `started_unix_ms` / `finished_unix_ms` integers inside `execution` (not hashed); `null` when not recorded |
| Tensor store | segment named by OID holding the raw bytes; shape and dtype in messages; readers verify the hash over the implied length (covers macOS page rounding) |
| Windows store | one file per OID, written to a temporary name and renamed |
| Segment lifetime | the MCP server removes the segments it created at end of input unless `--keep-segments` |
| Reference library | session coefficient tensors with the same configuration, per event, in order of first computation, excluding the query |
| C API config | `VkspScatterConfig` with `struct_size`; helpers for 1-D and 2-D defaults |
| Python build | PyO3 behind the crate feature `python` (enabled by maturin) so workspace tests never link Python; wheels stripped |
| Embedded vectors | `expected.json` and `expected.bin` compiled into the C library and the wheel (about 1.5 MB) |

## Decisions taken in C2

| Decision | Outcome |
|---|---|
| Versioning | `numerics_version` stays 1; new `tier2_version = 1` for sections 15 to 19 |
| Sums | pairwise `psum` (block size 1, split at `floor(n/2)`) over events; sequential dot products over features and parameters |
| Weights | `v_i = w_i / psum(w)`; dCorr2 defined as 0 when the denominator is below `1e-12` |
| Chunked dCorr2 | `c = min(4096, floor(n/2))`, Philox permutation (stream `0x20001`), partial chunk dropped; exact up to n = 10000 |
| Gradient | exact analytic dCorr2 gradient by default, `sign(0) = 0` at ties; Pearson proxy only on request |
| JSD binning | fixed range from the pre-cut background, 20 bins, last bin closed |
| Significance | Asimov `Z_A` from one cut at 50% signal efficiency; `null` when `b <= 0` |
| Win condition | evaluated at a declared `lambda_star` against the unique lambda = 0 row; FALSE if either difference is undefined |
| Training | Adam `0.9 / 0.999 / 1e-8`; batch weights normalized per batch; DisCo term on the batch's background events |
| Calibration | standardize, `X^T X + 1e-4 I`, Cholesky in fixed loop order |
| SW1 | fingerprint distribution = per-position log-coefficient vectors; 32 Philox directions by default; exact 1-D W1 for unequal sizes |
| HNSW | simple neighbour selection, `(distance, id)` order, `M = 8`, `ef_construction = 64`, `ef_search = 32`, level cap 16 |
| Layout | Fruchterman-Reingold, 100 iterations, unit square |
| JSON numbers | ECMAScript `Number::toString` (shortest round trip, ties to even); seeds as decimal strings; Markdown fixed six decimals |
| Philox streams | registry in VDS-1 section 15.4 |

Still open: D-FTZ (GPU measurements), D-SQRT, D-STEER (VDS-1 section 13).

## Recorded measurements

detmath accuracy (VDS-1 section 4.3):

| Function | f64 max ulp (not correctly rounded / n) | f32 max ulp (not correctly rounded / n) |
|---|---|---|
| `exp` | 1 (143 / 2010) | 0 (0 / 2010) |
| `ln` | 1 (4 / 2006) | 0 (0 / 2006) |
| `sin` | 1 (41 / 2009) | 1 (1 / 2009) |
| `cos` | 1 (39 / 2009) | 0 (0 / 2009) |

C3a local checks (Linux x86_64 container unless stated):

| Check | Result |
|---|---|
| Wheel `vikshep_compute-0.1.0-cp310-abi3` (manylinux, x86_64, stripped) | builds; 1.9 MB |
| pytest (`crates/vikshep-py/tests`) | 21 passed on Python 3.10 and 3.11 |
| Source install `pip install ./crates/vikshep-py` | builds and imports |
| C smoke test, C example, C++ example (`-Werror -pedantic`) | build and run; the forwarding backend gives the CPU reference's OID |
| Quick self-test (`vksp_conformance_selftest`, 185 cases) | 0 failures; two runs in 0.15 s including process start |
| MCP integration test, POSIX shared memory and file store | 3 / 3 passed; no segment left in `/dev/shm` |
| MCP integration test, Windows build (`x86_64-pc-windows-gnu`) under Wine | 3 / 3 passed with the default file store; same pinned manifest hash |
| Python `SharedMemory(name=oid)` producer with `vikshep-mcp` | output OID equals `vikshep_compute.scatter` |

Cross-platform conformance (suite v1, 331 cases), reports byte-identical:

| Platform | How | Result |
|---|---|---|
| x86_64 Linux | native; backends `cpu`, `cpu-serial`, `capi-cpu` | 331 / 331 pass, identical |
| AArch64 Linux | cross-compiled, qemu-user | 331 / 331 pass, identical |
| x86_64 Windows (`x86_64-pc-windows-gnu`) | MinGW build under Wine | 331 / 331 pass, identical |

Local report SHA-256: `046ed735aa4faf29d8f36425f82c8f94abaec864e465cd333d661fb14a7f1e9f` (equal on all three platforms).

Python parity (VDS-1 section 12.4):

| Quantity | Stated tolerance | Worst measured |
|---|---|---|
| dCorr2 (11 cases, absolute) | 1e-12 | 1.1e-16 |
| Pearson-proxy gradient (relative to its largest entry) | 1e-12 | 5.0e-16 |
| Calibration coefficients, standardization, bias, r2, residual std (relative) | 1e-12 | 1.1e-13 |

Gradient and model checks (VDS-1 section 12.5):

| Check | Stated tolerance | Measured |
|---|---|---|
| Exact dCorr2 gradient vs central differences (step 1e-6), relative to the largest component | 1e-6 | below 1e-9 |
| MLP logit gradient vs central differences | 1e-8 | passes |
| HNSW recall@10 vs brute force on SW1 (400 clouds) | at least 0.95 | 1.000 |
| Injected outlier in a scattering pipeline | flagged | flagged; k-th neighbour distance more than 10 times every other event's |

Canonical JSON numbers: identical to Node.js `String(x)` on the committed
fixture of 6170 values, the RFC 8785 Appendix B samples and 256356 further
values checked during development.

Benchmark reports in the suite (synthetic sample, 1200 training and 1200
evaluation events, 4 features, 5 epochs; numbers as printed in `report.md`).
These are conformance fixtures, not performance claims:

| Case | lambda_star | Delta-sigma | Delta-JSD | Result |
|---|---|---|---|---|
| `tier2/report/logistic/lambdas0_1_4/star1/n1200` | 1 | -6.666948 | -0.038645 | FALSE |
| `tier2/report/mlp16/lambdas0_0.5_2/star0.5/n1200` | 0.5 | 0.020611 | 0.008354 | FALSE |

In the logistic case the DisCo term removes most of the background
sculpting (JSD 0.040 to 0.002) at a large cost in `Z_A`; in the MLP case
`Z_A` rises slightly but the JSD does not fall. Both evaluate the rule as
specified.

Throughput (criterion, `scatter_2d`, 2-D 64x64, J = 3, L = 8, order 2,
batch of 1,000 events, CPU reference):

| Machine | Date | Events per second |
|---|---|---|
| cloud container, 4 vCPU Intel Xeon @ 2.10 GHz, Linux x86_64 | 2026-10-02 | 48.6 (criterion interval 47.9 to 49.3) |

## Next

* **C3b** in `samvardhan03/Vikshep`: per the C3b prompt.
* **B1**, **B2**, **D1** in the private engine repository: per their prompts.
