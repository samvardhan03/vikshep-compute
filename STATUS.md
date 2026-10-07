# Status

Current milestone: **C1.1** (VDS-1.1 flush-to-zero in the CPU reference,
`numerics_version = 2`, conformance suite v2). Next: **B2-r** in
`vikshep-compute-pro`, **B1**, **C3b**, **V1-a**.

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

## C3a (complete)

Gate met: PR [#3](https://github.com/samvardhan03/vikshep-compute/pull/3)
merged (commit 63cfe6f). CI run
[37143077708](https://github.com/samvardhan03/vikshep-compute/actions/runs/37143077708)
on the PR head passed every job: wheels for Linux x86_64 and aarch64, macOS
arm64 and x86_64 and Windows x86_64 with pytest on every native target, the C
and C++ examples on Linux and macOS, the MCP integration test on all four OS
legs (POSIX shared memory on Linux and macOS, the file store on Windows),
conformance and `hash-diff`. The first run failed only in `c-abi`: the
captured native library list carried an ANSI colour escape
(`-lc<ESC>[0m`); fixed by capturing it with `--color never`.

C3a delivered the Python package `vikshep-compute`, the stable C API
(`include/vikshep.h`) with external backend registration, the MCP data plane
`vikshep-mcp`, and `spec/provenance.schema.json`.

## C1.1 checklist (VDS-1.1 flush-to-zero)

### Spec
- [x] VDS-1 section 8.1 states the VDS-1.1 rules precisely: the flush definition; host flushing of every Tier-1 input tensor and every host-built table (filters, twiddles) before any kernel; flushing of the operands and the rounded result of every `+`, `-`, `*`, `sqrt` in every kernel; tininess judged after rounding, with the note that a before-rounding backend differs only for results rounding to exactly FLT_MIN and that `round_to_flt_min` cases catch it; sign of zero preserved; Tier 2 (binary64, CPU only) unchanged; `numerics_version = 2`, vectors v2
- [x] Sections 2.4, 4.4, 5.6, 6.2, 6.5, 11.2, 11.3, 12.2, 14.3.4, 14.6.1 and 20.3 updated; permitted omission of provably idle flushes documented with the proof of the CPU fast path

### Implementation
- [x] `vikshep_numerics::flush::ftz` (explicit, portable software flush; no MXCSR or FPCR flags), bit-identical to the spec's reference definition over a sweep of binary32 patterns
- [x] FFT (`vikshep_numerics::fft`): input and twiddles flushed on load, every butterfly sum, difference and product and the `2^-m` scaling flushed; binary64 host transforms unchanged
- [x] CPU kernels: `mul_real_filter` and `modulus` (both squares, the sum and the root) flush operands and results; `subsample` copies; `capabilities().preserves_subnormals` is now false
- [x] Host: the scattering driver flushes input samples at canvas embedding; filters are flushed at table construction; the conformance runner flushes every kernel input and filter before calling the backend
- [x] `numerics_version = 2`; `conformance/vectors/v1/` kept for history, `conformance/vectors/v2/` generated with the CPU reference

### Suite v2
- [x] 402 cases: the 313 Tier-1 and sweep cases of v1, 71 VDS-1.1 adversarial cases, 18 Tier-2 cases. New input kinds (VDS-1 section 11.2), each documented in `conformance/cases.toml` and the spec: `subnormal` (FFT, modulus, filter, 1-D and 2-D scattering), `flt_min_band` (sums and differences below, at and above FLT_MIN; FFT, modulus, scattering), `sqrt_flt_min_band` (squares below, at and above FLT_MIN; modulus), `round_to_flt_min` (products rounding up to exactly FLT_MIN; filter), `long_tail` (filter decaying to `2^-149`), `signed_zero` (filter products flushing to both signed zeros)
- [x] Against v1, only the `near_subnormal` cases and the JSON outputs that embed `numerics_version` changed; every other v1 output is byte-identical in v2; no stored kernel output contains a subnormal (242,828 values checked); the `round_to_flt_min` outputs hold exact FLT_MIN values and the `signed_zero` outputs both zero signs

### Bindings, tests, docs
- [x] Python byte-equality tests read `vectors/v2` (24 tests, three new flush-to-zero scattering cases); the C ABI and Python self-tests embed v2 (quick subset 256 cases); the MCP integration test checks `numerics_version` 2 and its pinned manifest hash was updated (`6ff71ab4...`)
- [x] Kymatio oracle within the stated tolerance and the exact power-of-two homogeneity test pass unchanged
- [x] `docs/external_backends.md`: what a backend must do under VDS-1.1 (flush after every operation unless proven identical by the suite; pass 100% of suite v2)
- [x] CI compares the sweep reports with `vectors/v2/hashes.json`
- [x] Flush cost on the CI runners measured and recorded under Recorded measurements (2026-10-06): 1.38 on Linux x86_64, 1.23 on Linux AArch64, 2.10 on Windows, 2.62 on macOS (noisy). Higher than the local 1.08 and not yet profiled.
- [x] `crates/vikshep-capi/c/abi_smoke.c` expects `numerics_version` 2. The C1.1 commit left it at 1, so the `c-abi` job failed on run 37547330000.

### Gate C1.1
- [x] numerics_version 2 implemented with software flush-to-zero
- [x] Locally: suite v2 (402 cases) passes with byte-identical reports on x86_64 Linux (`cpu`, `cpu-serial`, `capi-cpu`), AArch64 Linux (qemu-user) and x86_64 Windows (MinGW build under Wine); all workspace tests, pytest, the C/C++ examples and the MCP integration test pass
- [x] Suite v2 identical in CI on Linux x86_64, Linux arm64, macOS arm64 and Windows x86_64: `conformance` passed on all four and `hash-diff` passed in dispatch run 37547330000 (04a1edf)
- [ ] All bindings and the MCP test green in CI. Run 37547330000 passed the wheels with pytest on four targets and the MCP test on all four test runners. `c-abi` failed on the stale `abi_smoke.c` assert, which is fixed in the next commit; to be confirmed on the PR run
- [ ] PR open

## Decisions taken in C1.1

| Decision | Outcome |
|---|---|
| Flush primitive | exponent-field test with a branch-free mask; equals `if |x| < FLT_MIN { copysign(0, x) } else { x }` for every binary32 value |
| Where flushes happen | host: inputs and tables once; kernels: operands when loaded and every rounded result |
| Tininess | after rounding (the only observable value); `round_to_flt_min` cases pin it |
| Tier 2 | unchanged (binary64, no flush), including pooling and r2, which round once from binary64 on the host |
| CPU fast path | an FFT stage skips its result flushes when every nonzero operand is at least `2^-69` and every nonzero twiddle component at least `2^-10` (proof in VDS-1 section 8.1); bit-identical to flushing everything, tested, and suite v2 was generated with the unoptimized form |
| Vectors | `v2/` new; `v1/` kept unchanged for history |

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

Open decisions (VDS-1 section 13): D-FTZ is closed as flush-to-zero
(VDS-1.1, implemented in C1.1: `numerics_version` 2, suite v2). D-SQRT is
closed for Metal and open for CUDA; D-STEER is open.

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

Cross-platform conformance (suite v2, 402 cases, `numerics_version` 2),
reports byte-identical:

| Platform | How | Result |
|---|---|---|
| x86_64 Linux | native; backends `cpu`, `cpu-serial`, `capi-cpu` | 402 / 402 pass, identical |
| AArch64 Linux | cross-compiled, qemu-user | 402 / 402 pass, identical |
| x86_64 Windows (`x86_64-pc-windows-gnu`) | MinGW build under Wine | 402 / 402 pass, identical |

Local report SHA-256: `b142e88f450a5999d2ace979e65720bf0a435e4ecfbce7de95cea096f7d74de3`
(equal on all three platforms). Suite v1 (331 cases, `numerics_version` 1)
results are in the history of this file.

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

Cost of the VDS-1.1 software flush (same benchmark, same container type,
2026-10-06, criterion, mean seconds per batch of 1,000 events with the
interval; measured one after the other on an otherwise idle machine):

| Build | Seconds per 1,000 events | Events per second | Relative |
|---|---|---|---|
| `main` before C1.1 (`numerics_version` 1, no flush) | 19.62 (19.23 to 20.03) | 51.0 | 1.00 |
| flush after every operation, no fast path | 50.79 (50.31 to 51.24) | 19.7 | 2.59 |
| C1.1 as committed (flush with the proven FFT fast path) | 21.14 (20.90 to 21.38) | 47.3 | 1.08 |

The fast path removes the result flushes of an FFT stage only when they
are provably idle; the element-wise kernels always flush.

Flush cost on the GitHub-hosted runners (same benchmark, `bench` job,
2026-10-06, two `workflow_dispatch` runs started together: run 37547333227
on `main` at 63cfe6f, before C1.1, and run 37547330000 on this branch at
04a1edf, C1.1 as committed; criterion mean seconds per batch of 1,000
events with the interval, 10 samples each):

| Runner (vCPUs) | `main` before C1.1 | C1.1 | Relative |
|---|---|---|---|
| `ubuntu-latest`, x86_64 (4) | 17.90 (17.85 to 17.96) | 24.75 (24.71 to 24.81) | 1.38 |
| `ubuntu-24.04-arm`, AArch64 (4) | 10.21 (10.19 to 10.23) | 12.57 (12.55 to 12.60) | 1.23 |
| `macos-14`, Apple Silicon (3) | 23.09 (16.76 to 30.89) | 60.50 (42.17 to 80.93) | 2.62 |
| `windows-latest`, x86_64 (4) | 14.37 (13.80 to 15.05) | 30.24 (29.86 to 30.77) | 2.10 |

On every runner the cost is higher than the 1.08 measured in the cloud
container above. Treat the macOS row as indicative only: both intervals
are wide on the shared 3-vCPU runner. The two AArch64 jobs ran on
different runner image versions (20261004.142.1 on `main`, 20260927.135.1
on the branch). The other three pairs used the same image. The gap between
Linux x86_64 and Windows x86_64 points at code generation (the element-wise
flushes and the per-stage fast-path check) rather than at the arithmetic.
The cause has not been profiled yet. Conformance does not depend on it:
the reports are identical on all four platforms.

## Next

* **B2-r** in `vikshep-compute-pro`: re-run the Metal backend against suite
  v2 (`numerics_version` 2).
* **B1**: the CUDA backend against suite v2.
* **C3b** in `samvardhan03/Vikshep`: per the C3b prompt.
* **V1-a**: per its prompt.
