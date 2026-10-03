# Status

Current milestone: **C1** (reference scattering core and conformance suite
v1). Next: **C2**.

## C0 (complete)

Gate met: CI run [37069011872](https://github.com/samvardhan03/vikshep-compute/actions/runs/37069011872)
on `main` (commit 9fb357f) passed every job, including `test` and `hashes` on
`ubuntu-latest`, `ubuntu-24.04-arm`, `macos-14` and `windows-latest`, and the
`hash-diff` job. `ubuntu-24.04-arm` is available to this repository.

C0 delivered the workspace (Rust 1.97.0, edition 2024), `spec/VDS-1.md`
sections 1 to 13, `vikshep-detmath` (Option A: `libm =0.2.16`, no
features), SplitMix64 and Philox4x32-10 with cited known-answer tests, the
determinism lint, and the cross-platform sweep hashes.

## C1 checklist

### A. FFT (`vikshep-numerics::fft`)
- [x] Normative Stockham recursion (VDS-1 section 6) for binary32 (Tier 1) and binary64 (host), forward and inverse, batched 1-D, 2-D rows then columns
- [x] Twiddle tables per section 7 (octant plus exact symmetry), built with detmath, cached per N; binary64 host tables (section 7.1)
- [x] Tests: naive binary64 DFT for all N = 2..4096; round trip; exact table values; 2-D separability; bit-for-bit equality with the literal spec transcription
- [x] Spec pseudo-code needed no correction (verified in C0, re-verified against the production code)

### B. Filter banks (`vikshep-scatter::filters`)
- [x] 1-D Morlet bank (Fourier domain) and 2-D oriented Morlet bank (spatial domain, binary64 host FFT) with Gaussian low-pass, Kymatio 0.3.0 parameterization, binary64 with detmath, truncation below `2^-40` of the peak, one rounding to binary32
- [x] Stored Fourier filter is the real part; discarded imaginary part below `1e-5` of the peak (tested; measured at most `1.5e-7`)
- [x] Pad policies `Circular` and `ZeroPad` (halo `2^J` per side, canvas `next_pow2(n + 2^(J+1))`, crop offset 1 output sample)
- [x] Filter-bank fingerprint (SHA3-256 of the canonical filter bytes), recorded in the provenance manifest and as a conformance output
- [x] Deviations from Kymatio documented (VDS-1 section 14.3.6)

### C. Cascade (`vikshep-scatter::cascade`)
- [x] `ScatterConfig { dim, group, j, q, l, max_order, pad, shape, carrier_cutoff }`
- [x] `so2_relative` pooling (binary64 sum in increasing `l1`, divided by `L`, one rounding)
- [x] Execution model: host prepares filters and tables; the backend runs only `fft`, `ifft`, `mul_real_filter`, `modulus`, `subsample` in the order of VDS-1 section 14.6.2
- [x] Modulus `sqrt(re * re + im * im)`, no fusion
- [x] Canonical path order and tensor shape (section 14.7); r2 (14.8); log-mean fingerprint with pairwise binary64 sum (14.9); RFC 8785 provenance manifest (14.10)

### D. Backend API
- [x] `ScatterBackend` trait with the five kernels on batches, `name()`, `numerics_version()`, `capabilities()`
- [x] C ABI mirror `VkspBackendV1`, header `include/vikshep_backend.h` generated with cbindgen 0.29.4; memory ownership and alignment documented; `vksp_cpu_backend_v1()` exposes the CPU reference; C smoke test
- [x] `vikshep-cpu` reference backend; rayon only across independent canvases (parallel and serial runs give identical reports)

### E. Conformance suite v1
- [x] `conformance/cases.toml`: 1-D (N in {256, 1024}, J in {2, 4, 6}, Q in {1, 4, 8}), 2-D (32x32, 64x64, 128x128; J 1..4; L in {4, 8}; circular/circular and zero_pad/circular; both groups), zero padding, non-power-of-two length, order 1, L = 6, FFT N = 2..4096 forward and inverse, 2-D FFTs, modulus and filter cases, the C0 sweeps: 313 cases
- [x] Philox inputs per case (stream id = SHA3 of the case id); adversarial inputs: zeros, constants, impulses, near-subnormal, large, alternating
- [x] `conformance/vectors/v1/`: full bytes for outputs up to 16 KiB, SHA3-256 for the rest; generated with `cargo run --release -p vikshep-conformance -- generate`
- [x] `vikshep-conformance run --backend cpu` JSON report (per case PASS/FAIL; first differing element, expected/actual bits and ulp distance on failure; fault injection tested)
- [x] CI: every OS leg runs the full suite (CPU and through the C ABI); `hash-diff` requires all-pass and byte-identical reports plus the sweep hashes
- [x] First CI run of C1 ([run 37103963283](https://github.com/samvardhan03/vikshep-compute/actions/runs/37103963283), PR [#1](https://github.com/samvardhan03/vikshep-compute/pull/1)): the suite passed 313 / 313 on all four OS legs, CPU and C ABI, and the four reports were byte-identical (SHA-256 `5a527b3b1f39966f4946e61f3112b223e1be5e8482e3bc28150c9a48f8f01444`, equal to the local report). `hash-diff` failed only because its all-pass check grepped for `"fail": 0,` (the key is last in its object, so there is no comma); it now parses the report with `jq`

### F. Correctness oracles and properties
- [x] `oracles/kymatio_fixtures.py` (Kymatio 0.3.0, numpy 2.4.6, scipy 1.17.1, binary64): 20 configurations, fixtures committed with provenance
- [x] Within stated tolerance (VDS-1 section 12.1): full-resolution reference 2e-6 per order (worst 3.7e-7); Kymatio's own 2-D cascade 0.05 / 0.06 for orders 1 / 2 (worst 3.2e-2 / 3.8e-2, aliasing of its intermediate subsampling)
- [x] Littlewood-Paley bounds recorded (section 14.3.5); non-expansiveness (worst energy ratio 0.062); circular-shift covariance
- [x] EXACT property `r2(2^k x) == r2(x)` and `S(2^k x) == 2^k S(x)` bit for bit: passes on 1-D and 2-D, both groups, both pads, k in {-6, -3, -1, 1, 2, 5}

### G. Performance baseline (measurement, not a claim)
- [x] Criterion bench `cargo bench -p vikshep-scatter --bench scatter_2d` (2-D 64x64, J = 3, L = 8, order 2, batch of 1,000 events)
- [x] Local measurement recorded below
- [ ] Per-runner measurements: the `bench` CI job runs on manual dispatch (Actions, CI, Run workflow); record its results here

### H. Spec and status
- [x] VDS-1 section 14 complete (configuration, filter banks, padding, groups, execution plan, kernels, output layout, r2, log-mean, manifest); sections 6, 7, 8, 11, 12 extended; `numerics_version = 1`
- [x] D-FTZ CPU observations recorded (section 8.1)
- [x] STATUS updated

### Gate C1
- [x] Conformance v1 generated; identical reports on x86_64 Linux (native), AArch64 Linux (qemu-user) and x86_64 Windows (MinGW build under Wine)
- [x] Identical in CI on Linux x86_64, Linux arm64, macOS arm64, Windows x86_64 (run 37103963283; `hash-diff` green pending the re-run with the corrected check)
- [x] FFT correctness tests pass
- [x] Kymatio oracle within stated tolerance
- [x] Exact power-of-two homogeneity test passes
- [x] Spec section 14 complete
- [x] PR open ([#1](https://github.com/samvardhan03/vikshep-compute/pull/1))

## Decisions taken in C1

| Decision | Outcome |
|---|---|
| Filter parameterization | Kymatio 0.3.0, 1-D in the Fourier domain, 2-D in the spatial domain; Q2 = 1, T = 2^J |
| Orientation index | `theta_l = l pi / L` (Kymatio index mapped by `l = (L/2 - 1 - t) mod L`) |
| 2-D normalization | `2 pi sigma^2 / slant` (Kymatio uses 3.1415) |
| Resolution | full resolution throughout, subsample by `2^J` once at the end |
| Filter truncation | below `2^-40` of each filter's peak, for exact homogeneity |
| Zero padding | halo `2^J` per side, canvas `next_pow2(n + 2^(J+1))` |
| Pooling | mean over `l1`, binary64 sum in increasing `l1`, one rounding |
| r2 | `fl32(fl64(S2) / fl64(S1))`, `+0` where `S1 = 0`, default `carrier_cutoff = 1` |
| Log-mean | `eps = 2^-20`, `ln(eps + |c|)`, pairwise binary64 sum |
| Conformance storage | full bytes up to 16 KiB, otherwise SHA3-256 |

Still open: D-FTZ (GPU measurements), D-SQRT, D-STEER (VDS-1 section 13).

## Recorded measurements

detmath accuracy (VDS-1 section 4.3):

| Function | f64 max ulp (not correctly rounded / n) | f32 max ulp (not correctly rounded / n) |
|---|---|---|
| `exp` | 1 (143 / 2010) | 0 (0 / 2010) |
| `ln` | 1 (4 / 2006) | 0 (0 / 2006) |
| `sin` | 1 (41 / 2009) | 1 (1 / 2009) |
| `cos` | 1 (39 / 2009) | 0 (0 / 2009) |

Cross-platform conformance (suite v1, 313 cases), reports byte-identical:

| Platform | How | Result |
|---|---|---|
| x86_64 Linux | native; backends `cpu`, `cpu-serial`, `capi-cpu` | 313 / 313 pass, identical |
| AArch64 Linux | cross-compiled, qemu-user | 313 / 313 pass, identical |
| x86_64 Windows (`x86_64-pc-windows-gnu`) | MinGW build under Wine | 313 / 313 pass, identical |

Throughput (criterion, `scatter_2d`, 2-D 64x64, J = 3, L = 8, order 2,
batch of 1,000 events, CPU reference):

| Machine | Date | Events per second |
|---|---|---|
| cloud container, 4 vCPU Intel Xeon @ 2.10 GHz, Linux x86_64 | 2026-10-02 | 48.6 (criterion interval 47.9 to 49.3) |

## Next: C2

Anomaly search and calibration on top of the log-mean fingerprint and r2
(`vikshep-anomaly`, `vikshep-stats`), per the C2 prompt.
