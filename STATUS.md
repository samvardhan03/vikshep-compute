# Status

Current milestone: **C0** (foundation). Next: **C1** (FFT and scattering).

## C0 checklist

### 1. Repository skeleton
- [x] Rust workspace, edition 2024, resolver 3
- [x] `rust-toolchain.toml` pinned to Rust **1.97.0** (rustc 1.97.0, 2d8144b78, 2026-07-07)
- [x] Crates: `vikshep-detmath`, `vikshep-numerics`, `vikshep-scatter`, `vikshep-stats`, `vikshep-train`, `vikshep-anomaly`, `vikshep-backend-api`, `vikshep-cpu`, `vikshep-py`, `vikshep-capi`, `vikshep-mcp`, `vikshep-conformance` (binary)
- [x] `LICENSE` (AGPL-3.0), `LICENSES/CC-BY-4.0.txt`, `LICENSING.md`, `README.md`, `STATUS.md`, `CONTRIBUTING.md`, `SECURITY.md`, `.gitignore`, `.gitattributes`
- [x] Workspace lint `unsafe_code = "forbid"` everywhere except `vikshep-capi` and `vikshep-py` (FFI; they deny `unsafe_op_in_unsafe_fn` and `clippy::undocumented_unsafe_blocks` instead). `clippy::suboptimal_flops` not enabled.

### 2. Specification
- [x] `spec/VDS-1.md` sections 1-13 written; section 14 (scattering) headings reserved, "Specified in C1"
- [x] FFT pseudo-code verified as a correct DFT for N = 2..4096; no correction needed; result location recorded (always `x`; for odd m the final step copies from scratch)
- [x] Twiddle construction from the first octant specified and verified (exact values at p = 0, N/8, N/4, 3N/8)

### 3. `vikshep-numerics::rng`
- [x] SplitMix64, Philox4x32-10, `Stream` (`next_u32`, `next_u64`, `next_f32_unit`, `next_f64_unit`, `next_normal_f64`), own code
- [x] Philox4x32-10 known answers from Random123 `tests/kat_vectors` (tag v1.14.0) pass
- [x] SplitMix64 known answers (seeds 0 and 1234567) from Vigna's `splitmix64.c` and OpenJDK `SplittableRandom`, which agree, pass

### 4. `vikshep-detmath`
- [x] `exp`, `ln`, `sin`, `cos` for f64 and f32
- [x] mpmath fixture generator committed (`scripts/gen_detmath_fixtures.py`) with fixture; accuracy asserted
- [x] Determinism sweep (1,000,000 inputs per function), hashed with SHA3-256

### 5. Determinism lint
- [x] `scripts/determinism-lint.sh` with `scripts/determinism-lint.allow` (currently empty); type-aware backstop in `clippy.toml`

### 6. CI
- [x] Workflow `.github/workflows/ci.yml`: `fmt`, `clippy` (deny warnings, plus a check that `libm` has no features enabled), `determinism-lint`, `test` and `hashes` on `ubuntu-latest`, `ubuntu-24.04-arm`, `macos-14`, `windows-latest`, and the `hash-diff` job
- [ ] CI run on GitHub (pending: branch not yet pushed). `ubuntu-24.04-arm` is included because arm64 hosted runners are offered for public repositories; if the first run shows it unavailable, remove it from both matrices and set `expected` in `hash-diff` to 3.

### 7. Documentation
- [x] README, STATUS

### Gate C0
- [x] Workspace builds; all tests pass locally
- [x] `spec/VDS-1.md` complete except the scattering sections reserved for C1
- [x] Philox and SplitMix known-answer tests pass with cited sources
- [ ] CI green on Linux x86_64, macOS arm64, Windows x86_64, and the hash-diff job passes (pending first push)
- [ ] PR open (pending)

## Decisions taken

| Decision | Outcome |
|---|---|
| Toolchain | Rust 1.97.0, edition 2024 |
| D-MATH (detmath implementation) | Option A: `libm =0.2.16` with `default-features = false` (no `arch` paths) |
| D-DIV | No division in Tier 1 |
| RNG stream mapping | key = first SplitMix64 output of the master seed; counter = (block lo, block hi, stream_id lo, stream_id hi) |
| Normal variates | Box-Muller cosine branch only, 4 words per variate |
| Sweep master seed | `0x56445331` ("VDS1") |
| NaN serialization | canonical quiet NaN (VDS-1 section 2.6) |

Still open: D-FTZ, D-SQRT, D-STEER (see VDS-1 section 13).

## Recorded measurements

detmath accuracy against correctly rounded mpmath references (VDS-1 section 4.3):

| Function | f64 max ulp (not correctly rounded / n) | f32 max ulp (not correctly rounded / n) |
|---|---|---|
| `exp` | 1 (143 / 2010) | 0 (0 / 2010) |
| `ln` | 1 (4 / 2006) | 0 (0 / 2006) |
| `sin` | 1 (41 / 2009) | 1 (1 / 2009) |
| `cos` | 1 (39 / 2009) | 0 (0 / 2009) |

Cross-platform hash reports checked before the first CI run, all
byte-identical to `conformance/vectors/v1/hashes.json`:

| Platform | How | Result |
|---|---|---|
| x86_64 Linux (glibc), debug and release builds | native | identical |
| AArch64 Linux | cross-compiled, run under qemu-user | identical; full test suite also passes |
| x86_64 Windows (`x86_64-pc-windows-gnu`) | cross-compiled with MinGW, run under Wine | identical |

These emulated runs are supporting evidence only. The authoritative proof is
the CI `hash-diff` job on real macOS arm64, Windows (MSVC), and Linux
x86_64/arm64 runners.

## Next: C1

FFT (VDS-1 section 6) and twiddle tables (section 7) as production code in
`vikshep-numerics` / `vikshep-cpu`, the scattering definition (filters,
cascade, output layout; section 14), Kymatio and property-test oracles, and
FFT and scattering cases added to the conformance suite and the `hash-diff`
job.
