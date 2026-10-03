# Status

Current milestone: **C2** (deterministic statistics, exact DisCo training,
calibration, anomaly search). Next: **C3a**.

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

## C2 checklist

### A. `vikshep-stats`
- [x] Weighted dCorr2, exact statistic (VDS-1 section 16.1.1): Szekely-Rizzo with every mean weighted, O(n^2) time and O(n) memory per row, rows in parallel and combined in index order
- [x] Chunked estimator (16.1.2), labelled an estimator in code and spec: chunks of `min(4096, floor(n/2))` events from a Philox permutation; `weighted_dcorr2` switches to it above n = 10000
- [x] Exact analytic gradient of dCorr2 with respect to the scores (16.2); derivation in `docs/disco_gradient.md`; verified against central finite differences
- [x] Pearson-proxy gradient kept as the explicitly requested fast mode `gradient = "pearson_proxy"` (16.2.1), labelled as such
- [x] JSD with fixed-range binning (reference min/max, 20 equal-width bins by default), `0 ln 0 = 0`, nats (16.3)
- [x] Cut at a target signal efficiency and the Asimov significance, labelled "Asimov proxy, not a Wilks fit" in code, spec, `report.json` and `report.md` (16.4, 16.5)
- [x] Lambda-frontier rows and the benchmark report: `report.json` (RFC 8785) and `report.md` with the win condition Delta-sigma > 0 AND Delta-JSD <= 0 versus the lambda = 0 baseline, printed TRUE or FALSE (16.6, 16.7)

### B. `vikshep-train`
- [x] Logistic head and a one-hidden-layer tanh MLP head of configurable width (17.2); initialization from Philox
- [x] Per-epoch Philox shuffle and consecutive minibatches (17.6); Adam with documented constants and no `powf` (17.5); sigmoid, tanh and softplus via detmath `exp`/`ln` (17.3)
- [x] DisCo loss `wBCE + lambda dCorr2_w(yhat, m | background)` with the exact gradient (17.4); training on frozen features (C1 r2 ratios via `features_from_r2`, or user columns)
- [x] Lambda sweep producing the frontier and the benchmark report (17.7); synthetic benchmark sample with a sculpting feature (17.8)
- [x] Calibration: closed-form ridge with a deterministic Cholesky (fixed loop order, binary64), residual tensor with OID and calibration constants as canonical JSON (section 18)

### C. `vikshep-anomaly`
- [x] Sliced Wasserstein-1 between fingerprint distributions (19.1, 19.2): per-position log-coefficient point clouds (their mean is the C1 log-mean fingerprint), Philox unit directions, exact 1-D W1 by merging sorted samples, pairwise mean over directions
- [x] Deterministic HNSW (19.3): single-threaded insertion in input order, levels from Philox, `(distance, id)` ordering everywhere, `M = 8`, `ef_construction = 64`, `ef_search = 32` by default, canonical graph bytes
- [x] Detection by k-th nearest reference distance above `tau`, leave-one-out on the reference set (19.4)
- [x] Fixed-iteration Fruchterman-Reingold layout seeded from Philox (19.5)
- [x] Output contract (19.6): `nodes [{id, x, y, flagged}]`, `edges [{i, j, sw1}]`, `tau`, `config_digest`, `numerics_version` (and `tier2_version`); no features or fingerprints leave the core

### D. Python parity
- [x] `oracles/python_disco/gen_fixtures.py` runs the public Vikshep `weighted_dcorr2`, `_pearson_dcorr2_grad` and `train_calibrate` on the inputs of `backend/ingest/tests/test_disco.py` (same seeds and constructions); fixtures committed as exact bit patterns
- [x] Every original Python assertion re-checked on the Rust values (independent variables near 0; `Y = X^2` with symmetric `X` large; weighted and unweighted diverge on skewed weights; and the remaining cases of the file); Rust matches Python within the stated tolerance (VDS-1 section 12.4)

### E. Conformance additions
- [x] 18 `[[tier2]]` cases (VDS-1 section 11.5): dCorr2 exact (weighted and unweighted) and chunked, gradient, Pearson proxy, JSD, four small training runs (logistic and MLP, exact and proxy), calibration, SW1 matrix, HNSW graph bytes plus flags and graph JSON, two benchmark reports (JSON and Markdown bytes)
- [x] Suite v1 now 331 cases; the 313 C1 cases are unchanged; all run in every CI OS leg and in `hash-diff`

### F. Spec and status
- [x] VDS-1 sections 15 (Tier-2 determinism rules), 16 (statistics), 17 (training), 18 (calibration), 19 (anomaly search); sections 9.3, 10.1, 11, 12.3 to 12.5 extended
- [x] `numerics_version` stays 1 (Tier-1 arithmetic untouched); `tier2_version = 1` versions sections 15 to 19 independently (section 10.1)
- [x] Canonical JSON (RFC 8785) moved to `vikshep_numerics::jcs` with ECMAScript number formatting, checked against Node.js; the C1 manifests use it unchanged
- [x] STATUS updated

### Gate C2
- [x] All Tier-2 outputs bit-identical locally: suite v1 (331 cases) passes with byte-identical reports on x86_64 Linux (native; `cpu`, `cpu-serial`, `capi-cpu`), AArch64 Linux (qemu-user) and x86_64 Windows (MinGW build under Wine)
- [ ] Bit-identical in CI on Linux x86_64, Linux arm64, macOS arm64, Windows x86_64 (`hash-diff` on the C2 PR)
- [x] Exact DisCo gradient verified by central finite differences
- [x] Python-parity tests pass
- [x] Benchmark report bytes identical across platforms for the same inputs and seed (conformance cases `tier2/report/*`, locally on the three platforms above)
- [ ] PR open

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

## Next: C3a

Per the C3a prompt.
