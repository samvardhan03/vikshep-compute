# VDS-1: Vikshep Determinism Specification, version 1

| | |
|---|---|
| Status | Draft. Complete for `numerics_version = 1` (sections 1 to 14) and `tier2_version = 1` (sections 15 to 19) |
| Numerics version | 1 |
| Tier-2 version | 1 |
| Reference implementation | this repository (`vikshep-compute`), Rust CPU code |
| Copyright | Samvardhan Singh |
| Licence | CC-BY-4.0 (see `LICENSING.md`) |

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT,
RECOMMENDED, MAY and OPTIONAL are to be interpreted as described in BCP 14
(RFC 2119, RFC 8174) when, and only when, they appear in all capitals.

Notation: `fl32(e)` and `fl64(e)` denote the IEEE-754 binary32 / binary64
rounding (round-to-nearest, ties-to-even) of the exact value `e`. `lo32(v)`
and `hi32(v)` are the low and high 32-bit halves of a 64-bit integer. Hex
values are written with a `0x` prefix. All integer arithmetic on fixed-width
unsigned integers is modulo 2^width unless stated otherwise.

---

## 1. Purpose and scope

Vikshep turns detector-physics data into fixed wavelet-scattering features.
Its central promise is determinism: the same input and configuration produce
the same bytes on every supported machine. This specification defines what
"the same bytes" means, which computations it covers, and how a backend
proves it.

### 1.1 Tiers

Every computation belongs to exactly one tier.

**Tier 1: bit-exact on every conforming backend** (CPU reference, CUDA,
Metal). Tier 1 consists of:

- the forward and inverse FFT (section 6);
- multiplication of a complex spectrum by a real-valued Fourier-domain
  filter;
- the complex modulus;
- subsampling;
- and therefore the scattering coefficients S0, S1, S2 (section 14).

A conforming backend MUST produce Tier-1 outputs whose canonical bytes
(section 9) are identical to the reference implementation's.

**Tier 2: bit-exact on every platform because it always runs in the Rust CPU
code of this repository.** Tier 2 consists of: filter and twiddle-table
construction (section 7), r2 ratios, reductions, statistics (weighted
distance correlation, Jensen-Shannon divergence), training, calibration,
anomaly search, and report generation. GPU backends MUST NOT re-implement
Tier-2 computations; they receive Tier-2 results (for example, filter and
twiddle tables) as bytes from the host.

**Tier 3: reproducible only on the same build and platform.** External
simulators (Geant4) and any other third-party code whose arithmetic this
specification does not control. Tier 3 is out of scope of conformance.
Implementations record Tier-3 provenance (tool, version, build, platform,
seeds) in the provenance manifest (section 9.3) and make no cross-platform
claim about it.

### 1.2 Supported platforms

The reference implementation is supported on x86_64 and AArch64 CPUs running
Linux, macOS or Windows. 32-bit x86 without SSE2 (x87 arithmetic) is not
supported: x87 extended precision violates section 2.

---

## 2. Arithmetic model

2.1. All Tier-1 tensors MUST be IEEE-754 binary32. Tier-2 code MAY use
binary64; where a Tier-2 result becomes a Tier-1 input (for example a filter
or twiddle table), it MUST be rounded to binary32 exactly once, with
round-to-nearest-even, at the point this specification defines.

2.2. The rounding mode is round-to-nearest, ties-to-even, everywhere. No code
may change the floating-point environment.

2.3. Each operation is rounded individually, and expressions are evaluated
exactly in the order and grouping this specification writes them (left
operand first). Implementations MUST NOT reassociate, distribute, contract,
or otherwise transform floating-point expressions.

2.4. Only the following operations may appear in Tier-1 kernels:

- addition, subtraction and multiplication (`+ - *`);
- square root, which MUST be correctly rounded (see open decision D-SQRT);
- comparisons;
- exact power-of-two scaling (multiplication by `2^k`, which is exact unless
  the result is subnormal or overflows; with subnormals preserved it is still
  correctly rounded and therefore deterministic, see section 8).

2.5. The following MUST NOT appear in Tier-1 kernels:

- division (see D-DIV);
- transcendental functions of any kind;
- fused multiply-add, whether written explicitly or produced by compiler
  contraction;
- fast-math, relaxed-precision or approximate-function compilation modes;
- vendor FFT or BLAS libraries (cuFFT, vDSP, MPS, cuBLAS, MKL and similar);
- floating-point atomics;
- any reduction whose order depends on thread count, block size, scheduling
  or hardware.

2.6. NaN values: Tier-1 and Tier-2 computations on valid inputs MUST NOT
produce NaN. Because NaN payload propagation differs between processors, any
NaN that would be serialized MUST first be replaced by the canonical quiet NaN
(`0x7FC00000` for binary32, `0x7FF8000000000000` for binary64).

---

## 3. Backend obligations (normative compile settings)

### 3.1 CPU reference (Rust)

- Default Rust float semantics. Rust never contracts `a * b + c` into a fused
  multiply-add and never reassociates, at any optimisation level; build
  profiles MUST NOT add flags that change this.
- Explicit `mul_add` is forbidden in Tier-1 code unless this specification
  says otherwise (it currently never does). The determinism lint
  (`scripts/determinism-lint.sh`) and the clippy configuration (`clippy.toml`)
  reject it, together with the platform-libm float methods (section 4).
- The lints `clippy::suboptimal_flops` and `clippy::manual_mul_add` MUST NOT
  be enabled, since they recommend `mul_add`.
- Float algebraic or fast intrinsics (`fadd_fast`, `algebraic_add` and
  similar) MUST NOT be used.

### 3.2 CUDA (future private backend)

- `nvcc` MUST be invoked with `--fmad=false --prec-div=true --prec-sqrt=true`.
- `--use_fast_math` and `-ftz=true` MUST NOT be used, except that the FTZ
  setting follows decision D-FTZ (section 8) once decided.
- No cuFFT, no cuBLAS, no floating-point `atomicAdd`.
- Intrinsics that select a rounding mode or fuse operations (`__fmaf_rn` and
  similar) MUST NOT be used; `__fadd_rn`/`__fmul_rn` MAY be used to block
  contraction.

### 3.3 Metal (private backend — settings verified 2026-10-03)

The following settings were verified on AMD Radeon Pro 555 (GCN 4th-gen,
Metal 2, macOS 13.7.8) and confirmed accepted without error.

- `MTLCompileOptions.fastMathEnabled = false` — disables fast-math
  optimisations; accepted on macOS 13 / Metal 2.  On macOS 14+,
  `MTLMathMode.safe` MUST be used instead (the deprecated flag is removed
  in Sonoma and later).
- `#pragma STDC FP_CONTRACT OFF` placed at the top of each `.metal` source
  file (after `#include <metal_stdlib>`) — prevents compiler-induced FMA
  contraction via Metal's LLVM backend; accepted on all tested platforms.
- `precise::sqrt(x)` — requests IEEE-754 correctly rounded sqrt; verified
  bit-exact vs. the CPU reference over 2^20 inputs (see D-SQRT, section 13).
- AOT compilation via `xcrun metal -fno-fast-math` is NOT required and MUST
  NOT be assumed; it requires full Xcode.  Runtime JIT via
  `MTLDevice.makeLibrary(source:options:)` is the normative path and
  accepts all required options.
- `MTLMathMode` (macOS 14+ / Sonoma): replace `fastMathEnabled = false` with
  `mathMode = .safe` on deployment targets of macOS 14 and later.

### 3.4 Host-built tables

Twiddle tables (section 7) and, from C1, filter banks are built on the host
by the reference Rust code and passed to every backend as canonical bytes.
Backends MUST NOT recompute them.

---

## 4. Portable math (`vikshep-detmath`)

### 4.1 Requirement

All transcendental functions used anywhere in Tier 2 (currently `exp`, `ln`,
`sin`, `cos`; any function added later) MUST come from the
`vikshep-detmath` crate and MUST NOT come from the Rust standard library,
whose float methods call the platform C library (glibc, Apple libm, the
Microsoft CRT), whose results differ between platforms. The determinism lint
enforces this for all Rust code outside `vikshep-detmath`.

### 4.2 Decision D-MATH: implementation (decided: Option A)

- **Option A (adopted):** wrap the pure-Rust `libm` crate, pinned to exactly
  `=0.2.16`, built with `default-features = false`. With the `arch` feature
  disabled, `libm` selects no hardware-specific path on any supported target
  (its only unconditional hardware paths are for 32-bit x86 without SSE2,
  which is unsupported). Every function is then a fixed sequence of IEEE-754
  binary64 operations and integer bit manipulation, which section 2 makes
  identical on every platform.
- **Option B (not adopted):** port the CORE-MATH correctly rounded functions.
  This becomes necessary only if Option A fails the cross-platform sweep
  (section 4.4); its licence must be verified before any port.

Binary32 variants (`exp_f32`, `ln_f32`, `sin_f32`, `cos_f32`) evaluate the
binary64 function on the exactly widened argument and round the result once
to binary32: `f32(x) = fl32(f64(x))`.

Because Cargo unifies features across a build, no crate in the workspace may
enable any `libm` feature. CI checks that `cargo tree -e features -i libm`
shows no enabled feature. Changing the pinned version requires re-running the
sweep; if any hash changes, `numerics_version` MUST be bumped (section 10).

### 4.3 Accuracy (recorded measurement)

Measured against correctly rounded references computed with mpmath 1.4.1 at
1300 bits (`scripts/gen_detmath_fixtures.py`, fixture
`crates/vikshep-detmath/tests/fixtures/mpmath_reference.txt`). The error is
the distance in units in the last place (ulp) from the correctly rounded
result. These figures are asserted exactly by
`crates/vikshep-detmath/tests/accuracy.rs`.

| Function | Precision | Inputs | Max error (ulp) | Not correctly rounded |
|---|---|---:|---:|---:|
| `exp` | binary64 | 2010 | 1 | 143 |
| `ln`  | binary64 | 2006 | 1 | 4 |
| `sin` | binary64 | 2009 | 1 | 41 |
| `cos` | binary64 | 2009 | 1 | 39 |
| `exp_f32` | binary32 | 2010 | 0 | 0 |
| `ln_f32`  | binary32 | 2006 | 0 | 0 |
| `sin_f32` | binary32 | 2009 | 1 | 1 |
| `cos_f32` | binary32 | 2009 | 0 | 0 |

Accuracy is a correctness property, not a determinism property: determinism
requires only that every platform computes the same value, which section 4.4
tests.

### 4.4 Determinism sweep (normative)

For each function, one million inputs are generated from a Philox stream
(section 5) with master seed `0x56445331` (ASCII "VDS1"), the outputs are
serialized little-endian in generation order, and the SHA3-256 of the bytes
is reported. Input `i` (0-based) is generated as follows; all arithmetic is
in the stated precision with one rounding per operation, in the order
written; `pow2(e)` is the exact power of two.

| Case | Stream id | Input |
|---|---:|---|
| `detmath.exp.f64` | 1 | `x = 1456 * u - 746`, `u = next_f64_unit()` |
| `detmath.ln.f64` | 2 | `x = from_bits(next_u64() mod 0x7FF0000000000000)` |
| `detmath.sin.f64` | 3 | trig rule below (binary64) |
| `detmath.cos.f64` | 4 | trig rule below (binary64) |
| `detmath.exp.f32` | 5 | `x = 194 * u - 105`, `u = next_f32_unit()` (binary32 arithmetic) |
| `detmath.ln.f32` | 6 | `x = from_bits(next_u32() mod 0x7F800000)` |
| `detmath.sin.f32` | 7 | trig rule below (binary32) |
| `detmath.cos.f32` | 8 | trig rule below (binary32) |

Trig rule, binary64: draw `a = next_u64()`, then `b = next_u64()`. If
`i mod 16 = 15`, `x = from_bits((a AND 0x8000000000000000) OR ((a AND
0x7FFFFFFFFFFFFFFF) mod 0x7FF0000000000000))` (an arbitrary finite value,
magnitudes up to about 1.8e308). Otherwise `u = (a >> 11) * pow2(-53)`,
`e = (b mod 65) - 32`, `x = (2 * u - 1) * pow2(e)`.

Trig rule, binary32: as binary64 with `a = next_u32()`, `b = next_u32()`,
sign mask `0x80000000`, magnitude mask `0x7FFFFFFF`, modulus `0x7F800000`,
`u = (a >> 8) * pow2(-24)` and `e = (b mod 33) - 16`.

The domains include overflow of `exp` to infinity, underflow to subnormals
and zero, `ln` of +0 and of subnormals, and argument reduction of the largest
finite arguments. No sweep output is NaN (checked).

The expected hashes are in `conformance/vectors/v1/hashes.json`.
`cargo run -p vikshep-conformance -- hashes` prints the report; CI runs it on
every platform of section 1.2 and fails if any report differs from the
committed vectors.

---

## 5. Random numbers

All randomness in Vikshep (initialisation, sampling, bootstrap, test-signal
generation) MUST come from the generators of this section. Implementation:
`crates/vikshep-numerics/src/rng.rs` (own code, no external RNG crate).

### 5.1 SplitMix64 (seed expansion)

State: one 64-bit word `x`, initialised to the seed. Each output:

```
x = x + 0x9E3779B97F4A7C15
z = x
z = (z XOR (z >> 30)) * 0xBF58476D1CE4E5B9
z = (z XOR (z >> 27)) * 0x94D049BB133111EB
output z XOR (z >> 31)
```

This is the fixed-increment SplitMix64 of Vigna (2015), identical to
`java.util.SplittableRandom` (Steele, Lea and Flood, 2014) seeded with the
same value.

### 5.2 Philox4x32-10

Counter-based generator of Salmon, Moraes, Dror and Shaw ("Parallel random
numbers: as easy as 1, 2, 3", SC 2011). Key: 2 x 32-bit words. Counter: 4 x
32-bit words. Output: 4 x 32-bit words.

Constants: multipliers `M0 = 0xD2511F53`, `M1 = 0xCD9E8D57`; key increments
`W0 = 0x9E3779B9`, `W1 = 0xBB67AE85`. `mulhilo(a, b)` returns the high and
low 32-bit halves of the exact 64-bit product.

```
philox_u32x4(key[2], ctr[4]):
  k = key; c = ctr
  for r in 0..10:
    if r > 0:
      k[0] = k[0] + W0
      k[1] = k[1] + W1
    (hi0, lo0) = mulhilo(M0, c[0])
    (hi1, lo1) = mulhilo(M1, c[2])
    c = [hi1 XOR c[1] XOR k[0], lo1, hi0 XOR c[3] XOR k[1], lo0]
  return c
```

(Ten rounds, nine key bumps: the first round uses the unmodified key.)

### 5.3 Stream mapping (normative)

A stream is identified by `(master_seed: u64, stream_id: u64)`. Its output is
a sequence of 32-bit words indexed by `index` in `[0, 2^66)`:

```
k        = first SplitMix64 output with state initialised to master_seed
key      = [lo32(k), hi32(k)]
block    = index >> 2
lane     = index AND 3
counter  = [lo32(block), hi32(block), lo32(stream_id), hi32(stream_id)]
word(master_seed, stream_id, index) = philox_u32x4(key, counter)[lane]
```

Distinct `stream_id`s use disjoint counters under the same key, so streams
never overlap. Because every word is a pure function of
`(master_seed, stream_id, index)`, outputs MAY be generated in any order or in
parallel and are still bit-identical. Code that consumes words sequentially
maintains a position; the position after a draw is the position before plus
the number of words consumed.

### 5.4 Derived values (normative)

| Method | Words consumed | Definition |
|---|---:|---|
| `next_u32` | 1 | `word(index)` |
| `next_u64` | 2 | `w0 OR (w1 << 32)` where `w0` is drawn first |
| `next_f32_unit` | 1 | `(next_u32 >> 8) * 2^-24` in binary32: exact, in `[0, 1)` |
| `next_f64_unit` | 2 | `(next_u64 >> 11) * 2^-53` in binary64: exact, in `[0, 1)` |
| `next_normal_f64` | 4 | Box-Muller, below |

`next_normal_f64` (binary64, each operation rounded once, in this order):

```
u1 = 1 - next_f64_unit()          # in (0, 1], so ln(u1) is finite
u2 = next_f64_unit()              # in [0, 1)
r  = sqrt(-2 * detmath.ln(u1))    # sqrt correctly rounded
z  = r * detmath.cos(TAU * u2)    # TAU = fl64(2 * pi) = 0x401921FB54442D18
return z
```

The sine branch is discarded so that each variate depends only on its own
four words.

### 5.5 Known-answer vectors

Philox4x32-10, from the Random123 distribution, file `tests/kat_vectors`, at
release tag v1.14.0 of https://github.com/DEShawResearch/random123 (counter
words, key words, expected output words, all hex):

| Counter | Key | Output |
|---|---|---|
| `00000000 00000000 00000000 00000000` | `00000000 00000000` | `6627e8d5 e169c58d bc57ac4c 9b00dbd8` |
| `ffffffff ffffffff ffffffff ffffffff` | `ffffffff ffffffff` | `408f276d 41c83b0e a20bc7c6 6d5451fd` |
| `243f6a88 85a308d3 13198a2e 03707344` | `a4093822 299f31d0` | `d16cfe09 94fdcceb 5001e420 24126ea1` |

SplitMix64, first five outputs. Produced by Vigna's public-domain reference
`splitmix64.c` (as reproduced in https://github.com/lemire/testingRNG,
`source/splitmix64.h`) and independently by OpenJDK 21
`new java.util.SplittableRandom(seed).nextLong()`; both agree:

| Seed | Outputs |
|---|---|
| 0 | `e220a8397b1dcdaf 6e789e6aa1b965f4 06c45d188009454f f88bb8a8724c81ec 1b39896a51a8749b` |
| 1234567 | `599ed017fb08fc85 2c73f08458540fa5 883ebce5a3f27c77 3fbef740e9177b3f e3b8346708cb5ecd` |

Both tables are tested in `crates/vikshep-numerics/src/rng.rs`.

### 5.6 Stream sweep cases

With master seed `0x56445331`, one million values each, serialized
little-endian and hashed with SHA3-256 (vectors in
`conformance/vectors/v1/hashes.json`):

| Case | Stream id | Values |
|---|---:|---|
| `rng.philox.u32` | 100 | `next_u32` |
| `rng.philox.f32_unit` | 101 | `next_f32_unit` |
| `rng.philox.f64_unit` | 102 | `next_f64_unit` |
| `rng.philox.normal_f64` | 103 | `next_normal_f64` |
| `rng.splitmix64.u64` | n/a | SplitMix64 outputs with seed `0x56445331` |

---

## 6. FFT algorithm (normative)

### 6.1 Sizes

Transform lengths are `N = 2^m` with `1 <= m <= 12` (`2 <= N <= 4096`).
Complex values are pairs `(re, im)` of binary32. `TW` is the forward twiddle
table `TW_N` of section 7.

### 6.2 Recursion

Radix-2 Stockham autosort, decimation in frequency. `x` holds the input,
`y` is scratch of the same length.

```
fft0(n, s, eo, x, y):          # n: current length, s: stride, eo: parity
  if n == 1:
    if eo: for q in 0..s: y[q] = x[q]
    return
  m = n / 2
  for p in 0..m:
    w = TW[p * (N / n)]        # forward: exp(-2*pi*i*p/n) from table TW
    for q in 0..s:
      a = x[q + s*(p + 0)]
      b = x[q + s*(p + m)]
      y[q + s*(2*p + 0)] = (a.re + b.re, a.im + b.im)
      d = (a.re - b.re, a.im - b.im)
      y[q + s*(2*p + 1)] = (d.re*w.re - d.im*w.im, d.re*w.im + d.im*w.re)
  fft0(n / 2, 2*s, not eo, y, x)

forward FFT of length N: fft0(N, 1, false, x, y); result is in x
```

### 6.3 Evaluation order

- `a + b` and `a - b` are componentwise, one rounding per component.
- The complex product `d * w` is evaluated as: real part
  `fl(fl(d.re * w.re) - fl(d.im * w.im))`; imaginary part
  `fl(fl(d.re * w.im) + fl(d.im * w.re))`. Two independent products, then
  one subtraction (real) or addition (imaginary), left operand first, no
  fusion.
- The butterflies within one level are independent; a backend MAY execute
  them in any order or in parallel, since each output element is written
  exactly once per level from values of the previous level.

### 6.4 Result location (verified)

The recursion has been verified as written (no correction was needed) by
`crates/vikshep-numerics/tests/vds1_fft_pseudocode.rs`, which transcribes it
literally and compares it with a naive binary64 DFT for every `m` in
`1..=12`. The result always ends in `x`:

- **m even:** the last butterfly level writes into `x`; the final `n == 1`
  call has `eo = false` and copies nothing. `y` holds the output of the
  second-to-last level.
- **m odd:** the last butterfly level writes into `y`; the final `n == 1`
  call has `eo = true`, `s = N`, and copies all `N` elements from `y` into
  `x`. Both arrays then hold the result.

The output is in natural (not bit-reversed) order: `X[k] = sum_j x[j] *
exp(-2*pi*i*j*k/N)`.

### 6.5 Inverse transform

The inverse transform runs the same recursion with the conjugate table
`TWC[p] = (TW[p].re, -TW[p].im)` and then multiplies every component of the
result by the exact constant `2^-m`. That multiplication is exact unless the
result is subnormal, in which case it is correctly rounded (section 8).

### 6.6 Two-dimensional transforms

A 2-D transform of an `R x C` row-major array (`R`, `C` powers of two within
6.1) applies the 1-D transform to every row first (length `C`), then to every
column (length `R`), each with its own table `TW_C` / `TW_R`. Rows are
independent of each other, as are columns, so they MAY be processed in any
order or in parallel. An axis of length 1 is not transformed, so a `1 x C`
array is a 1-D transform. For the inverse, every 1-D transform applies its
own scaling (rows by `2^-log2(C)`, then columns by `2^-log2(R)`).

### 6.7 Reference implementation

`vikshep_numerics::fft` (`crates/vikshep-numerics/src/fft.rs`) unrolls the
recursion into a loop over levels that alternates the two buffers and
performs the final copy when `m` is odd. Its output equals the literal
transcription of section 6.2 bit for bit for every `m` in `1..=12`, forward
and inverse (`production_fft_equals_literal_recursion_bitwise`), and is
within `4 * log2(N) * eps32 * ||x||_2` of a naive binary64 DFT on every
output element (`crates/vikshep-numerics/tests/fft.rs`). The 2-D transform
moves columns through a transposed buffer; this is data movement only.

---

## 7. Twiddle tables

`TW_N[p] = (cos(2*pi*p/N), -sin(2*pi*p/N))` for `p` in `0..N/2`. The table
is computed on the host in binary64 with `vikshep-detmath`, rounded once to
binary32 (round-to-nearest-even), and generated from the first octant,
extended by exact symmetry:

```
N = 2^m; Q4 = N / 4; Q8 = N / 8         # integer division
step = fl64(TAU * 2^-m)                 # exact: power-of-two scaling of TAU
s    = fl32(sqrt(1/2))                  # = 0x3F3504F3

# first quadrant: C[k] ~ cos(2*pi*k/N), S[k] ~ sin(2*pi*k/N), k in 0..=Q4
for k in 0..=Q8:
  theta = fl64(k * step)
  C[k] = fl32(detmath.cos(theta)); S[k] = fl32(detmath.sin(theta))
if N >= 8: C[Q8] = s; S[Q8] = s         # exact octant midpoint
for k in Q8+1..=Q4:
  C[k] = S[Q4 - k]; S[k] = C[Q4 - k]    # reflection about pi/4

# half circle
for p in 0..N/2:
  if p <= Q4: TW[p] = ( C[p],      -S[p])
  else:       TW[p] = (-S[p - Q4], -C[p - Q4])
```

Consequences (verified for every `N` in 6.1 by the test of section 6.4):
`TW[0] = (1, 0)`; for `N >= 4`, `TW[N/4] = (0, -1)`; for `N >= 8`,
`TW[N/8] = (s, -s)` and `TW[3N/8] = (-s, -s)`, all exactly. For `N = 2` the
table is `[(1, 0)]`; for `N = 4` it is `[(1, 0), (0, -1)]`.

Tables are built only by the reference Rust code (Tier 2). Backends receive
the bytes (`re, im` interleaved, little-endian binary32, `N/2` entries).

### 7.1 Binary64 host transform

Filter construction (section 14) uses the same recursion in binary64 on the
host (Tier 2). Its tables `TW64_N` follow the construction above with every
`fl32` replaced by the identity (values stay binary64) and
`s = fl64(sqrt(1/2))`. The binary64 transform is never a Tier-1 operation.

---

## 8. Subnormals

### 8.1 Decision D-FTZ: CLOSED — flush-to-zero

**Measurement (2026-10-03, `vikshep-compute-pro` B2 work, AMD Radeon Pro 555,
GCN 4th-gen, Metal 2, macOS 13.7.8):**

The Metal backend microtest (`crates/vikshep-backend-metal/tests/microtests.rs`
in `vikshep-compute-pro`) ran 1 048 576 (2^20) binary32 subnormal inputs through
`s + s` and `s * 2.0f` on the Radeon Pro 555.  At index 0 (bits = `0x00000001`,
the smallest positive subnormal), both operations returned `+0.0` instead of the
IEEE-754 correctly rounded result.  The GPU's MODE register on AMD GCN sets
`DENORM_FLZ_IN` and `DENORM_FLZ_OUT` to 1 by default for compute shaders; Metal
provides no MSL control to clear these bits.

**VDS-1.1 consequence (step 3 of section 8.2):** D-FTZ closes as "flush".
VDS-1.1 defines flush-to-zero semantics for all backends:

- Any Tier-1 binary32 input whose absolute value is strictly less than
  `f32::MIN_POSITIVE` (i.e. a subnormal) is treated as zero of the same sign
  before the operation, for `+`, `-`, `*`, and `sqrt`.
- Any Tier-1 binary32 result of `+`, `-`, `*`, and `sqrt` whose absolute value
  is strictly less than `f32::MIN_POSITIVE` is replaced by zero of the same sign.
- The CPU reference MUST emulate flush-to-zero in software.
  Canonical emulation helper:
  ```rust
  #[inline(always)]
  fn ftz(x: f32) -> f32 {
      if x.abs() < f32::MIN_POSITIVE { 0.0_f32.copysign(x) } else { x }
  }
  ```
  applied to all Tier-1 operands before each operation and to all Tier-1 results.
- `numerics_version` is incremented from 1 to 2; all conformance vectors MUST
  be regenerated.
- No Metal or CUDA backend may claim conformance under `numerics_version = 1`
  (subnormals-preserved vectors).  Conformance is re-run after regeneration.

**CPU observations (VDS-1.0):** IEEE subnormals were preserved in the CPU
reference.  Rust on x86_64 and AArch64 preserves subnormals by default.
`cpu_preserves_subnormals` (`crates/vikshep-scatter/tests/properties.rs`) and
the `near_subnormal` conformance cases (section 11.2) all passed bit-exactly on
every tested CPU platform.  These observations are superseded by VDS-1.1.

### 8.2 Decision procedure

1. The Metal backend work measures, on Apple GPU hardware, whether every
   Tier-1 operation (`+ - *`, `sqrt`, comparisons) preserves binary32
   subnormal inputs and outputs, using a dedicated test set of subnormal
   operands and results; the CUDA backend work does the same with
   `-ftz=false`.
2. If every backend preserves subnormals: D-FTZ closes as "preserve";
   nothing changes.
3. If any backend cannot preserve subnormals: VDS-1.1 defines flush-to-zero
   semantics for all backends (subnormal inputs of every Tier-1 operation are
   treated as zero of the same sign and subnormal results are replaced by
   zero of the same sign), the CPU reference emulates them in software,
   `numerics_version` is bumped, and all conformance vectors are regenerated.
4. The measurement, its data, and the outcome are recorded in this section.

---

## 9. Canonical serialization and hashing

### 9.1 Tensor bytes

A tensor is serialized as its elements in row-major (C) order, contiguous, no
padding, each element little-endian in its IEEE-754 encoding (binary32 for
Tier-1 tensors). Complex tensors interleave `re, im`.

### 9.2 Object identifier

`OID = lowercase hex of the first 14 bytes of SHA3-256(tensor bytes)`, which
is 28 hexadecimal characters. This matches the existing Vikshep contract and
MUST NOT change. Implementation: `crates/vikshep-numerics/src/oid.rs`.

### 9.3 Provenance manifest

Dtype, shape and layout are not part of the tensor bytes; they are recorded
in the provenance manifest. The manifest hash is SHA3-256 of the manifest
serialized with the JSON Canonicalization Scheme (RFC 8785, section 15.6).
Manifests never carry result floats: results are tensors referenced by their
OIDs.

---

## 10. Versioning

`numerics_version` is an integer, starting at 1 (`NUMERICS_VERSION` in
`vikshep-numerics`). Any change to arithmetic order, tables, algorithms,
constants, generator definitions, or the pinned `libm` version that changes
any output bit MUST bump it. Conformance vectors are keyed by
`numerics_version` (`conformance/vectors/v<version>/`). Within one
`numerics_version`, cases MAY be added, but the expected output of an existing
case MUST NOT change.

### 10.1 Tier-2 version

`tier2_version` (`TIER2_VERSION` in `vikshep-numerics`, starting at 1)
versions the Tier-2 algorithms of sections 15 to 19 independently of the
Tier-1 arithmetic. A change to a Tier-2 algorithm, order, constant or format
that changes an output bit MUST bump `tier2_version`; it does not bump
`numerics_version` as long as no Tier-1 output and no C0 sweep changes. The
expected outputs of the `tier2/*` conformance cases (section 11.5) are keyed
by `tier2_version`: a `tier2_version` bump regenerates exactly those cases.
Tier-2 reports carry both versions.

---

## 11. Conformance

### 11.1 Definition

A backend conforms to VDS-1 at a given `numerics_version` if and only if it
reproduces 100% of that version's conformance cases bit-exactly. Partial
conformance is non-conformance.

### 11.2 Suite v1

The suite is defined by `conformance/cases.toml` (CC-BY-4.0) as grids that
`vikshep-conformance` expands, outer to inner in the order the keys are
written, into cases with normative identifiers:

| Kind | Identifier | Outputs |
|---|---|---|
| 1-D FFT, N = 2..4096, forward and inverse | `fft1d/n{N}/{forward,inverse}/{input}` | `out` (complex) |
| 2-D FFT | `fft2d/{R}x{C}/{direction}/{input}` | `out` (complex) |
| modulus kernel | `modulus/n{n}/{input}` | `out` (complex) |
| filter multiplication kernel | `mul_real_filter/{R}x{C}/{input}` | `out` (complex) |
| scattering | `scatter/{dim}d/{shape}/J{J}-Q{Q}-L{L}/{pads}/{group}/o{max_order}/{input}` | `S`, `r2`, `log_mean`, `filters` |
| C0 sweeps (sections 4.4, 5.6) | `sweep/{name}` | `out` (hash only) |
| Tier-2 cases (section 11.5) | `tier2/{kind}/...` | per kind |

The grids cover 1-D scattering with N in {256, 1024}, J in {2, 4, 6}, Q in
{1, 4, 8}, plus zero padding, non-power-of-two length (192) and adversarial
inputs; 2-D scattering with N in {32x32, 64x64, 128x128}, J in {1, 2, 3, 4},
L in {4, 8}, pads circular/circular and zero_pad/circular, both groups, plus
adversarial inputs, a rectangular zero-padded case with L = 6 and order-1
cases. Version 1 has 331 cases: 313 Tier-1 and sweep cases (C0, C1) and 18
Tier-2 cases (C2).

**Stream id** of a case: the first 8 bytes, little-endian, of SHA3-256 of its
identifier. Inputs are drawn from `Stream(master_seed = 0x56445331,
stream_id)` (section 5).

**Inputs.** For a real signal of `R x C` samples (`R = 1` in 1-D), in
row-major order, with `u = next_f32_unit()` drawn once per sample and all
arithmetic in binary32:

| Input | Sample value |
|---|---|
| `uniform` | `2u - 1` |
| `near_subnormal` | `(2u - 1) * 2^-120` (straddles the subnormal range) |
| `large` | `(2u - 1) * 2^40` (no overflow anywhere in the cascade) |
| `zeros` | `+0` |
| `constant` | `0.75` |
| `impulse` | `1` at row `R/2`, column `C/2`, `+0` elsewhere |
| `alternating` | `+1` if `r + c` is even, else `-1` |

Complex inputs draw `re` then `im` for each sample for the three random
kinds, and use the real pattern with `im = +0` for the others. The filter of
a `mul_real_filter` case is `next_f32_unit()` per element, drawn after the
input. Scattering cases use `carrier_cutoff = 1`.

**Outputs.** `S` is the coefficient tensor (section 14.7), `r2` the ratio
tensor (14.8), `log_mean` the fingerprint (14.9, binary64) and `filters` the
canonical filter bytes (14.3.4).

### 11.3 Expected vectors and runner

`conformance/vectors/v1/expected.json` lists, per case and output, the
dtype, byte length and SHA3-256; outputs of at most 16384 bytes (except
`filters` and sweeps) are also stored in full in
`conformance/vectors/v1/expected.bin` at the recorded offset. The vectors
are produced by the CPU reference with

```
cargo run --release -p vikshep-conformance -- generate
```

and checked with

```
cargo run --release -p vikshep-conformance -- run --backend cpu [--report FILE]
```

which prints a JSON report: per case `PASS` or `FAIL`; for a failing output
stored in full, the index of the first differing element (complex values
count as two binary32 elements), the expected and actual bits and their
distance in ulps (for raw-byte outputs such as JSON, the first differing
byte); for a hash-only output, the hash mismatch. The report
contains no timing or platform data, so reports from different machines
compare byte for byte. The process exits non-zero if any case fails.
`--backend capi-cpu` runs the CPU reference through the C ABI of section
14.6.3.

### 11.4 Cross-platform verification

CI runs the full suite on Linux x86_64, Linux AArch64, macOS arm64 and
Windows x86_64; the `hash-diff` job fails unless every report is all-pass
and byte-identical. Before each first CI run, identical reports were also
obtained on AArch64 Linux under qemu-user and on x86_64 Windows (MinGW build)
under Wine.

### 11.5 Tier-2 cases

`[[tier2]]` entries of `conformance/cases.toml` are single cases. Inputs come
from `Stream(0x56445331, stream_id)` of the case (`z` a standard normal
`next_normal_f64`, `u` a `next_f64_unit`), per kind:

| Kind | Inputs | Outputs |
|---|---|---|
| `dcorr_exact`, `dcorr_chunked` (seed = stream id) | per event `x = z1`, `y = x x + 0.5 z2`, then `w = 0.1 + u` if weighted, else 1 | `value` |
| `dcorr_grad` | as above, scores `sigmoid(x)`, protected `y` | `value`, `grad` |
| `pearson_proxy` | as `dcorr_grad` | `grad` |
| `jsd` | synthetic sample (section 17.8, seed = master seed, split = stream id); background `m`, post-cut = background with `x0 > 0.5` | `pre`, `post`, `jsd` |
| `train` | synthetic sample of `n` events, `d = 4`; 4 epochs, batch 128, `lr = 0.02`, seed = stream id | `model` (section 17.6), `scores` |
| `calibration` | `x_ij = (1 + j) z + j`; `t = 1 + sum_j c_j x_ij + 0.1 z`, `c_j = (j + 1) (-1)^j / d` | `constants` (JSON), `residuals` |
| `sw1` | cloud `i`: `points x d` values `z + 0.1 i`; directions seeded with the stream id | `matrix` (all pairs) |
| `hnsw` | `n` clouds of 8 points in 4-D, values `z + 4` for every 50th cloud, else `z`; 8 directions, `ef_construction = 32`, `ef_search = 16`, 30 layout iterations, seed = stream id | `graph_bytes`, `kth_distance`, `flags`, `graph_json` |
| `report` | train and eval synthetic samples (splits = stream id and stream id + 1), 5 epochs, batch 128, `lr = 0.02` | `report_json`, `report_md` |

---

## 12. Correctness oracles

Determinism (identical bytes everywhere) and correctness (the bytes are
right) are tested separately.

### 12.1 Kymatio oracle

Kymatio 0.3.0 (BSD-3-Clause, numpy backend, binary64) is a test oracle only;
it is not a dependency of, and is not distributed with, any crate.
`oracles/kymatio_fixtures.py` writes 20 fixtures (`oracles/fixtures/`, 8
1-D and 12 2-D configurations including zero padding, a rectangular canvas
and both groups) holding the binary32 input and references in the canonical
path order, after aligning conventions (section 14.3.6). CI does not
install Kymatio; `crates/vikshep-scatter/tests/kymatio_oracle.rs` compares
against the committed fixtures. For each order `m` the error is

```
err_m = max over order-m paths and positions |ours - ref| / max |ref|
```

| Reference | Stated tolerance (orders 0, 1, 2) | Worst measured |
|---|---|---|
| `fullres`: Kymatio's filters, cascade at full resolution (1-D: Kymatio's own `scattering1d` core with oversampling >= J; 2-D: numpy around `kymatio.scattering2d.filter_bank`) | 2e-6, 2e-6, 2e-6 | 2.4e-7, 3.7e-7, 3.1e-7 |
| `kymatio_core` (2-D only): Kymatio's own `scattering2d` core, which subsamples U1 by `2^j1` and U2 by `2^j2` in the Fourier domain | 2e-6, 0.05, 0.06 | 2.4e-7, 3.2e-2, 3.8e-2 |

The `fullres` agreement is at binary32 precision. The `kymatio_core`
difference is the aliasing of Kymatio's intermediate subsampling, which
VDS-1 does not perform (section 14.3.6).

### 12.2 Property tests

`crates/vikshep-scatter/tests/`:

- Littlewood-Paley bounds of each bank, recorded in section 14.3.5.
- Discarded imaginary parts of 2-D Fourier filters below `1e-5` of the peak
  (section 14.3.3).
- Non-expansiveness on random pairs:
  `2^(J*dim) * ||Sx - Sy||^2 <= 1.05 * ||x - y||^2` (the factor corrects for
  the final subsampling; measured ratios are at most 0.07).
- Circular-shift covariance: shifting the input by `2^J` samples on circular
  axes shifts every path by one output sample, within `1e-5` of the peak.
- Exact power-of-two homogeneity of S and r2, bit for bit (section 14.8).
- Subnormal preservation on the CPU (section 8.1).

### 12.3 Low-level references

detmath against mpmath (section 4.3), the FFT against the literal recursion
and a naive DFT (sections 6.4, 6.7), the generators against published known
answers (section 5.5), canonical-JSON number formatting against ECMAScript
`String(x)` (section 15.6).

### 12.4 Parity with the public Python metric

`oracles/python_disco/gen_fixtures.py` runs the public Vikshep repository's
`weighted_dcorr2` (`backend/ingest/src/vikshep_ingest/disco.py`), its
Pearson-proxy gradient and its `train_calibrate`
(`backend/ingest/src/vikshep_ingest/cli/_train.py`, commit 7882dfc) on the
inputs of `backend/ingest/tests/test_disco.py` (same numpy seeds and
constructions) and records inputs and outputs as exact binary64 bit
patterns. Every original Python assertion is re-checked on the Rust values
(independent variables near 0, `Y = X^2` with symmetric `X` above 0.2,
weighted above unweighted on skewed weights, perfect linear and identical
arrays near 1, a single element exactly 0, weight-scale invariance, and the
length-mismatch error).

| Quantity | Stated tolerance | Worst measured |
|---|---|---|
| dCorr2 (11 cases, absolute) | 1e-12 | 1.1e-16 |
| Pearson-proxy gradient (relative to its largest entry) | 1e-12 | 5.0e-16 |
| Calibration coefficients, standardization, bias, r2, residual std (relative) | 1e-12 | 1.1e-13 |

The chunked estimator is not compared: the Python reference permutes with
numpy's generator and this one with Philox (section 16.1.2).

### 12.5 Gradient and model checks

The exact dCorr2 gradient matches central finite differences (step `1e-6`)
to within `1e-9` of its largest component on weighted and unweighted samples
(stated tolerance `1e-6`; `crates/vikshep-stats/tests/gradient.rs`). The MLP
logit gradient matches finite differences to `1e-8`. HNSW recall@10 against
brute force on SW1 (400 clouds) is 1.000 (stated minimum 0.95), and an
injected outlier event in a scattering pipeline is flagged with a k-th
neighbour distance more than ten times that of every other event
(`crates/vikshep-anomaly/tests/end_to_end.rs`).

---

## 13. Open decisions register

| Id | Question | Status | Default / resolution path |
|---|---|---|---|
| D-MATH | Source of portable transcendental functions | Decided: Option A (`libm =0.2.16`, no features) | Revisit only if the cross-platform sweep fails (section 4.2) |
| D-FTZ | Preserve subnormals or flush to zero on every backend | **Closed: flush** (Metal measurement 2026-10-03; VDS-1.1 defines FTZ semantics, section 8.1) | VDS-1.1: flush-to-zero for all backends; CPU emulation required; `numerics_version` bumped from 1 to 2; conformance vectors regenerated |
| D-SQRT | Is `sqrt` correctly rounded on each GPU backend? | Metal: **correctly rounded** (2026-10-03, AMD Radeon Pro 555, GCN 4th-gen; `precise::sqrt` verified bit-exact vs. CPU `f32::sqrt()` over 2^20 inputs — no Markstein fallback needed on this hardware). CUDA: open. | CPU: correctly rounded (IEEE-754 requires it; Rust lowers to the hardware square-root instruction). CUDA: `--prec-sqrt=true` expected to give a correctly rounded `sqrt`, to be verified. Fallback for any backend that fails: a Markstein-style square root using an exact fused multiply-add and a final correction step, proven correctly rounded; this is the only place an FMA could be admitted, and only by amendment of this specification |
| D-DIV | Division in Tier 1 | Decided: no division in Tier 1 | r2 ratios, pooling means and every other quotient are computed on the host (Tier 2) |
| D-STEER | Steerable-basis orientation synthesis | Open | Changes arithmetic; allowed only if the reference adopts it under a new `numerics_version` |

---

## 14. Scattering transform

Implementation: `crates/vikshep-scatter` (host, Tier 2), the backend
interface `crates/vikshep-backend-api`, the CPU reference
`crates/vikshep-cpu` (Tier 1) and its C mirror `crates/vikshep-capi`.

With `u (*) h = IDFT(DFT(u) . h^)` the circular convolution on the canvas
(section 14.4) by a real Fourier-domain filter `h^`, and `sub_J` the output
subsampling of section 14.4:

```
S0 x         = sub_J( x (*) phi_J )
S1[l1] x     = sub_J( |x (*) psi_l1| (*) phi_J )
S2[l1, l2] x = sub_J( ||x (*) psi_l1| (*) psi_l2| (*) phi_J ),   only for j(l2) > j(l1)
```

No intermediate layer is subsampled: every convolution runs on the full
canvas, and outputs are subsampled by `2^J` once, at the end. S0 is the real
part of `x (*) phi_J`.

### 14.1 Configuration

| Field | Meaning | Constraint |
|---|---|---|
| `dim` | 1 or 2 | |
| `group` | `trivial` or `so2_relative` (section 14.5) | `so2_relative` requires `dim = 2` |
| `J` | octaves; outputs subsampled by `2^J` | `1 <= J <= 11` |
| `Q` | first-order wavelets per octave | 1-D: `1 <= Q <= 32`; 2-D: `Q = 1` |
| `L` | orientations | 2-D: even, `2 <= L <= 32`; 1-D: `L = 1` |
| `max_order` | 1 or 2 | |
| `pad` | `circular` or `zero_pad`, one per axis (rows first) | section 14.4 |
| `shape` | signal length per axis (rows first) | multiple of `2^J`; circular axes a power of two; canvas `<= 4096` |
| `carrier_cutoff` | r2 keeps paths with `j1 >= carrier_cutoff` | default 1 |

The joint SE(2) convolution and 3-D SO(3) scattering are out of scope of
VDS-1.

### 14.2 One-dimensional Morlet filter bank

Source: Kymatio 0.3.0, `kymatio/scattering1d/filter_bank.py`
(`compute_params_filterbank`, `compute_sigma_psi`, `compute_xi_max`,
`get_max_dyadic_subsampling`, `adaptive_choice_P`, `morlet_1d`,
`gauss_1d`, `scattering_filter_factory` with `T = 2^J`). Filters are built
directly in the Fourier domain, in binary64, with `vikshep-detmath`.

Constants: `sigma0 = 0.1`, `alpha = 5`, `r_psi = fl64(sqrt(1/2))`,
`eps = 1e-7`, `P_max = 5`. `pow2(e)` is the exact power of two for integer
`e` and `detmath.exp(e * LN2)` otherwise, `LN2 = fl64(ln 2)`. Operations are
evaluated left to right with one rounding each.

**Parameters of a bank with quality factor `q`** (first order `q = Q`,
second order `q = 1`):

```
sigma_min  = sigma0 * 2^-J
xi_max     = max(1 / (1 + pow2(3 / q)), 0.35)
sigma_psi(xi) = xi * ((1 - f) / (1 + f)) * (1 / sqrt(2 * detmath.ln(1 / r_psi))),
                f = 1 / pow2(1 / q)
if sigma_psi(xi_max) <= sigma_min:
    filters = [];  elbow = sigma_psi(xi_max)
else:
    filters = [(xi_max, sigma_psi(xi_max))]
    while sigma_last > sigma_min * pow2(1 / q):
        append (xi_last / pow2(1 / q), sigma_last / pow2(1 / q))
    elbow = xi_last
for k in 1 .. q-1:
    append (elbow - (k / q) * elbow, sigma_min)
j(xi, sigma) = floor(-log2(min(xi + alpha * sigma, 0.5))) - 1
```

`j` is evaluated with exact comparisons: `floor(-log2(ub))` is the largest
integer `k` with `ub <= 2^-k`.

**Filter values** on a canvas of `N` bins, bin `k` at frequency `k / N`
cycles per sample:

```
P      = min(ceil(sqrt(-2 * (sigma * sigma) * detmath.ln(eps)) + 1), P_max)
G_c(k) = ( sum_{p = 0 .. 2P-2} detmath.exp( -(f_p - c)^2 / (2 * (sigma * sigma)) ) ) / (2P - 1),
         f_p = ((1 - P) * N + p * N + k) / N,   summed in increasing p
psi^   = G_xi - kappa * G_0,   kappa = G_xi(0) / G_0(0)
phi^   = G_0 with sigma = sigma_low = sigma0 * 2^-J
h^     = h^ / sum_n |h[n]|,    h = inverse binary64 DFT of h^ (section 7.1),
         |z| = sqrt(re * re + im * im), summed in increasing n
```

1-D filters are real by construction. The canonical order of a bank is the
generation order (decreasing `xi`, non-decreasing `j`).

### 14.3 Two-dimensional Morlet filter bank

Source: Kymatio 0.3.0, `kymatio/scattering2d/filter_bank.py` (`filter_bank`,
`morlet_2d`, `gabor_2d`). Filters are built in the spatial domain in binary64
and transformed with the binary64 host FFT. The same bank serves the first
and second order.

#### 14.3.1 Parameters

For `j` in `0..J` and orientation index `l` in `0..L` (canonical order: `j`
outer, `l` inner):

```
sigma = 0.8 * 2^j
theta = (l * pi) / L                 # angles 0, pi/L, ..., (L-1) pi / L
xi    = ((3/4) * pi) / 2^j           # radians per sample
slant = 4 / L
```

Low-pass: `sigma_phi = 0.8 * 2^(J-1)`, `theta = 0`, `xi = 0`, `slant = 1`.
`pi = fl64(pi)`.

#### 14.3.2 Spatial construction

On an `R x C` canvas (row `n`, column `m`):

```
c = detmath.cos(theta); s = detmath.sin(theta); s2 = slant * slant
den  = 2 * sigma * sigma
a    = (c * c + s2 * (s * s)) / den
b    = (2 * (c * s) * (1 - s2)) / den
d    = (s * s + s2 * (c * c)) / den
norm = 2 * pi * sigma * sigma / slant
for each (n, m):                     G = (0, 0); E = 0
  for ex in -2..=2: for ey in -2..=2:
    x = n + ex * R;  y = m + ey * C   (exact integers)
    q = -(a * (x * x) + b * (x * y) + d * (y * y))
    e = detmath.exp(q)
    t = x * xi * c + y * xi * s
    G = G + (e * detmath.cos(t), e * detmath.sin(t));  E = E + e
  g[n, m] = G / norm (componentwise);  env[n, m] = E / norm
```

Terms with `q < -746` contribute exactly `+0` (`exp` underflows to zero) and
MAY be skipped. Morlet:
`K = (sum g.re / sum env, sum g.im / sum env)` (sums in row-major order),
`psi = (g.re - K.re * env, g.im - K.im * env)`. Low-pass: `phi = env`.
Fourier filter: the forward binary64 2-D transform (rows, then columns).

#### 14.3.3 Real part

The spatial filters are Hermitian (`psi(-u) = conj(psi(u))`), so their
transforms are real up to rounding and the asymmetric tails of the 5 x 5
periodization. The stored filter is the real part. The discarded imaginary
part MUST NOT exceed `1e-5` of the largest real magnitude
(`discarded_imaginary_part_is_small`; measured at most `1.5e-7` on the tested
configurations).
Multiplying a complex spectrum by a real filter is two independent
products.

#### 14.3.4 Truncation, rounding and fingerprint

For every filter (1-D and 2-D): with `T = max_k |h^(k)|`, values with
`|h^(k)| < 2^-40 * T` become exactly `+0`; the rest are rounded once to
binary32. Truncation keeps every stored value far from the binary32
subnormal range, which keeps the exactness argument of section 14.8 valid.

The filter-bank fingerprint is the lowercase hex SHA3-256 of the canonical
filter bytes: `phi`, then the first-order bank, then the second-order bank
when it is distinct (1-D), each little-endian binary32 in canvas order. It
is recorded in every provenance manifest (section 14.10) and is the
`filters` output of every scattering conformance case.

#### 14.3.5 Littlewood-Paley bounds (recorded)

`A(w) = |phi^(w)|^2 + 1/2 sum_psi (|psi^(w)|^2 + |psi^(-w)|^2)` over all
canvas frequencies (the symmetrisation accounts for analytic wavelets
applied to real signals). The value 1 is attained at `w = 0`, where
`phi^ = 1` and every wavelet vanishes. Measured
(`littlewood_paley_bounds`):

| Bank | min A | max A |
|---|---:|---:|
| 1-D N=256, J=4, Q=1 (first and second order) | 0.165797 | 1.000000 |
| 1-D N=1024, J=6, Q=1 (first and second order) | 0.165824 | 1.000000 |
| 1-D N=1024, J=6, Q=8, first order | 0.000290 | 1.000000 |
| 1-D N=1024, J=6, second order (Q=1) | 0.165824 | 1.000000 |
| 2-D 64x64, J=3, L=4 | 0.132018 | 1.000000 |
| 2-D 64x64, J=3, L=8 | 0.122981 | 1.013572 |
| 2-D 128x128, J=4, L=8 | 0.122981 | 1.046507 |

Kymatio's normalizations (l1 in 1-D, Gaussian area in 2-D) are not tight
frames: the lower bounds are well below 1, and the 2-D upper bound slightly
exceeds 1 for L = 8.

#### 14.3.6 Deviations from Kymatio 0.3.0

1. **Orientation index.** Kymatio's index `t` has angle
   `(L/2 - 1 - t) pi / L`; VDS-1 index `l` has angle `l pi / L`. They
   correspond by `l = (L/2 - 1 - t) mod L`; where the angles differ by `pi`
   the wavelets are complex conjugates and the moduli of real signals agree.
   Reason: orientations in `[0, pi)` in increasing order.
2. **2-D normalization constant.** Kymatio divides by
   `2 * 3.1415 * sigma^2 / slant`; VDS-1 uses `pi`. Every 2-D filter is
   smaller by `3.1415 / pi` and order-`m` coefficients by `(3.1415/pi)^(m+1)`.
   Reason: the exact Gaussian normalization.
3. **Precision.** Kymatio builds 2-D filters with a binary32 rotation matrix
   and complex64 accumulation; VDS-1 uses binary64 throughout, and writes
   the off-diagonal curvature as one term `b` (section 14.3.2).
4. **Truncation** below `2^-40` of each filter's peak (section 14.3.4).
5. **Single resolution.** Kymatio 2-D (and 1-D with `oversampling < J`)
   subsamples intermediate layers by periodizing spectra; VDS-1 computes every
   layer at full resolution and subsamples once (section 14.4), which avoids
   aliasing and keeps the backend to five kernels.
6. **Boundary.** Kymatio pads by reflection; VDS-1 uses circular or zero
   padding (section 14.4).

The oracle fixtures (section 12.1) apply 1, 2 and 6; 3 and 4 fall within the
stated tolerance.

### 14.4 Canvas, padding and output grid

Per axis of length `n`, with `F = 2^J`:

| Pad | Canvas length | Signal offset | Output length | Output offset |
|---|---|---|---|---|
| `circular` | `n` (power of two) | 0 | `n / F` | 0 |
| `zero_pad` | `next_pow2(n + 2F)` | `F` | `n / F` | 1 |

The canvas holds `(x, +0)` at the signal positions and `(+0, +0)` elsewhere;
a 1-D canvas has one row. The output subsampling takes, for output position
`(a, c)`, the real part of canvas element `(F * (o_r + a), F * (o_c + c))`
with the output offsets `o` of the table (1-D: row 0).

### 14.5 Groups

`trivial`: every path of section 14.7. `so2_relative` (2-D only): pooled over
absolute orientation, keeping the relative orientation of second-order paths,
per output position:

```
S1'[j1]            = mean_l1 S1[(j1, l1)]
S2'[j1, j2, delta] = mean_l1 S2[(j1, l1), (j2, (l1 + delta) mod L)],   delta in 0..L
mean_l1 v(l1)      = fl32( (sum_{l1 = 0 .. L-1} fl64(v(l1))) / L )
```

The sum runs in increasing `l1` in binary64; the division is binary64; the
result is rounded once to binary32. Pooling runs on the host.

### 14.6 Execution plan and kernels

#### 14.6.1 Kernels

The host prepares filters and twiddle tables and moves data (copies and
canvas embedding are exact). A backend executes exactly five kernels, each
acting independently on every canvas of a batch of contiguous row-major
binary32 complex canvases:

| Kernel | Per element |
|---|---|
| `fft` | forward transform of sections 6.2 and 6.6 with host tables |
| `ifft` | inverse transform (conjugate tables, `2^-m` per axis) |
| `mul_real_filter` | `(re, im) -> (re * h, im * h)`, filter `index[b]` for canvas `b` |
| `modulus` | `(re, im) -> (sqrt(re * re + im * im), +0)`: two products, one addition, correctly rounded `sqrt` |
| `subsample` | `out[b][a][c] = in[b][F * (o_r + a)][F * (o_c + c)].re` |

#### 14.6.2 Ordered invocations

With `B` signals, the filter array `[phi, psi1_0, ..., psi2_0, ...]` (2-D:
the second-order bank is the first-order bank) and `n1` first-order filters:

1. host: embed the `B` signals in canvases `X`.
2. `fft(X)`.
3. `T = X`; `mul_real_filter(T, phi)`; `ifft(T)`; `subsample(T)` gives S0.
4. For each signal `b`:
   1. `U` = `n1` copies of `X[b]`; `mul_real_filter(U, psi1_0 .. psi1_{n1-1})`;
      `ifft(U)`; `modulus(U)`; `fft(U)`.
   2. `T = U`; `mul_real_filter(T, phi)`; `ifft(T)`; `subsample(T)` gives
      S1 of `b`.
   3. If `max_order = 2`, for each first-order filter `l1` with partners
      `l2` (`j(l2) > j(l1)`, in bank order): `Z` = one copy of `U[l1]` per
      partner; `mul_real_filter(Z, psi2_l2 ...)`; `ifft(Z)`; `modulus(Z)`;
      `fft(Z)`; `mul_real_filter(Z, phi)`; `ifft(Z)`; `subsample(Z)` gives
      S2 of `(b, l1, l2)`.
5. host: pooling (14.5), r2 (14.8), log-mean (14.9).

Because every kernel is element-wise per canvas, a backend MAY batch or
reorder independent invocations and MAY fuse kernels, provided the
arithmetic of every element is unchanged.

#### 14.6.3 Interfaces

Rust: the trait `vikshep_backend_api::ScatterBackend` (the five kernels on
batches, plus `name()`, `numerics_version()`, `capabilities()`). C: the
table `VkspBackendV1` in `include/vikshep_backend.h`, generated by cbindgen
from `crates/vikshep-capi`. The caller owns every buffer; a kernel must not
retain pointers after returning; `VkspComplex32` is `{ float re; float im; }`
(8 bytes, alignment 4); calls are synchronous. `vksp_cpu_backend_v1()`
exposes the CPU reference through the same table.

### 14.7 Output layout

`S` is a binary32 tensor `[B, P, out_rows, out_cols]` (1-D:
`[B, P, out_len]`), row-major. Paths, in canonical order:

- order 0;
- order 1 by first-order filter: 1-D in bank order; 2-D by `(j1, theta1)`;
- order 2 by `(first-order filter, second-order filter)`: 2-D by
  `(j1, theta1, j2, theta2)`, only `j2 > j1`;
- `so2_relative`: order 0; order 1 by `j1`; order 2 by `(j1, j2, delta)`.

Path counts in 2-D: `trivial` `1 + J L + L^2 J (J - 1) / 2`;
`so2_relative` `1 + J + L J (J - 1) / 2` (order 2 terms omitted when
`max_order = 1`).

### 14.8 Scale-free ratio r2 and exact homogeneity

For every order-2 path `(l1, l2)` with `j1 >= carrier_cutoff` (S0 and the
first-order carriers below the cutoff are dropped), per output position:

```
r2[l1, l2] = fl32( fl64(S2[l1, l2]) / fl64(S1[l1]) ),   r2 = +0 where S1[l1] = 0
```

computed on the host. r2 is a binary32 tensor `[B, P2, out...]` in the order
of the kept order-2 paths. For `so2_relative`, `S1'[j1]` divides
`S2'[j1, j2, delta]`.

**Exact property.** For any input `x` and integer `k` such that no
intermediate value overflows or is subnormal, `S(2^k x) = 2^k S(x)` and
`r2(2^k x) = r2(x)` bit for bit: every Tier-1 operation commutes with
scaling by a power of two (additions, subtractions and products by constants
scale exactly; `sqrt(2^(2k) v) = 2^k sqrt(v)`; multiplication by `2^-m` is
exact), the binary64 pooling mean commutes, and the ratio cancels the
factor. Truncation (14.3.4) keeps filter products away from underflow.
`crates/vikshep-scatter/tests/homogeneity.rs` checks this for
`k in {-6, -3, -1, 1, 2, 5}` on 1-D and 2-D configurations, both groups and
both pad policies.

### 14.9 Log-mean fingerprint

Per signal and path, in binary64 with `ln` from `vikshep-detmath`:

```
log_mean = psum( ln(2^-20 + |c_s|) for every output position s ) / count
psum([]) = +0;  psum([a]) = a;  psum(a[0..n]) = psum(a[0..h]) + psum(a[h..n]),  h = floor(n / 2)
```

The output is binary64 `[B, P]`. The absolute value covers S0, which can be
negative, and rounding below zero in S1 and S2.

### 14.10 Provenance manifest

Each scattering run is described by an RFC 8785 manifest
(`vikshep_scatter::manifest::scattering_manifest`) with the keys `backend`,
`config` (`J`, `L`, `Q`, `carrier_cutoff`, `dim`, `group`, `max_order`,
`pad`, `shape`), `filter_bank_sha3`, `input_oid`, `kind`
(`"vikshep.scattering"`), `numerics_version`, `output_dtype` and
`output_oid`.

---

## 15. Tier-2 determinism rules

Sections 15 to 19 specify the Tier-2 operations that run on top of the
scattering features: statistics (`vikshep-stats`), training and calibration
(`vikshep-train`) and anomaly search (`vikshep-anomaly`). They always run in
the Rust CPU code of this repository and MUST produce identical bytes on
every supported platform. They are versioned by `tier2_version`
(section 10.1).

### 15.1 Arithmetic

Binary64 throughout; every operation is rounded once and evaluated in the
order written (section 2.3). Transcendental functions come only from
`vikshep-detmath` (section 4); `sqrt` (correctly rounded), division,
`floor`, `ceil`, `min`, `max` and `abs` are allowed. No fused multiply-add.

### 15.2 Sums

Every sum over events (or other data-sized index sets) is the pairwise tree
`psum`, applied to the terms in their natural (event) order:

```
psum([])      = +0
psum([a])     = a
psum(a[0..n]) = psum(a[0..h]) + psum(a[h..n]),   h = floor(n / 2)
```

(`vikshep_numerics::sum::pairwise_sum`, block size 1). Dot products over
feature or parameter dimensions, and the sums of the Cholesky factorization,
are sequential from index 0 (from the bias where stated). Parallel execution
is allowed only across independent units (for example the rows of the dCorr2
double sums) whose results are written to fixed indices and then combined in
index order by `psum`; results never depend on the number of threads.

### 15.3 Means and divisions

A mean is `psum(values) / count`. Normalized weights are `v_i = w_i / psum(w)`.

### 15.4 Randomness

All randomness comes from Philox streams (section 5) with documented stream
ids under a master seed supplied by the caller:

| Use | Stream id |
|---|---|
| chunked dCorr2 permutation (16.1.2) | `0x20001` |
| head initialization (17.2) | `0x30000` |
| shuffle of epoch `e` (17.6) | `0x310000000 + e` |
| SW1 projection directions (19.2) | `0x40001` |
| HNSW levels (19.3) | `0x40002` |
| graph layout (19.5) | `0x40003` |
| synthetic sample, split `s` (17.8) | `0x50000 + s` (wrapping) |
| conformance case (11.5) | first 8 bytes of SHA3-256 of the case id |

Integer draws: `uniform_below(bound)` draws `r = next_u64()` until
`r <= 2^64 - 1 - ((2^64 - 1) mod bound + 1) mod bound` and returns
`r mod bound` (unbiased rejection). Permutations: Fisher-Yates from the
identity, `for i = n-1 down to 1: swap(p[i], p[uniform_below(i + 1)])`.
Unit and normal variates as in section 5.4.

### 15.5 Sorting

Sorts of floating-point keys use the IEEE totalOrder (`f64::total_cmp`)
with ties broken by ascending element index; descending sorts order keys
descending and ties by ascending index. Candidate sets in HNSW are ordered by
the pair `(distance, id)` in this total order.

### 15.6 Output formats

* **Tensors.** Floating-point results that leave the core are tensors
  (little-endian binary64, or binary32 for Tier-1 outputs, in a documented
  order) referenced by OIDs (section 9.2).
* **JSON.** Reports, manifests, calibration constants and graphs are RFC
  8785 canonical JSON (`vikshep_numerics::jcs`): members sorted by the
  UTF-16 code units of their keys, no whitespace, minimal string escapes,
  and numbers written as ECMAScript `Number::toString` writes binary64
  values: the shortest decimal digit string that round-trips (ties between
  equally short candidates resolved to the even digit), in fixed notation
  when the decimal exponent `n` satisfies `-6 < n <= 21` and in exponential
  notation (`1e+21`, `1.5e-7`) otherwise; `-0` is written `0`. NaN and
  infinities are not representable: an undefined statistic is `null`.
  Integers MUST stay within `2^53` in magnitude; 64-bit identifiers such as
  seeds are written as decimal strings. Files end with one newline. The
  formatter matches Node.js `String(x)` on a committed fixture of 6170 values
  (`crates/vikshep-numerics/tests/fixtures/ecmascript_numbers.txt`, generator
  `oracles/jcs_numbers/gen.sh`), on the RFC 8785 Appendix B samples, and on
  256356 further values checked during development, and JavaScript
  reproduces the stored reports byte for byte with
  `JSON.stringify(JSON.parse(x))`.
* **Markdown.** Numbers are written with exactly six digits after the
  decimal point, correctly rounded from the binary64 value with ties to even
  (Rust's exact decimal formatting); `-0.000000` is written `0.000000`, and
  `null` is written `n/a`.

---

## 16. Statistics

Implementation: `crates/vikshep-stats`.

### 16.1 Weighted distance correlation

Inputs: `x`, `y`, optional weights `w` (default all 1), equal lengths `n`,
finite values, weights finite, non-negative and not all zero (otherwise an
error). For `n < 2` the value is 0.

#### 16.1.1 Exact statistic (normative)

```
v_i     = w_i / psum(w)
r_i     = psum_j( v_j * |x_i - x_j| )                 # row means of a_ij
mu      = psum_i( v_i * r_i )
A_ij    = ((|x_i - x_j| - r_i) - r_j) + mu            # B_ij likewise for y
S_i^xy  = psum_j( (v_j * A_ij) * B_ij )               # S^xx, S^yy likewise
dCov2   = psum_i( v_i * S_i )                         # for xy, xx, yy
dCorr2  = dCov2_xy / sqrt(dCov2_xx * dCov2_yy)
```

`dCorr2` is defined as `0` when `dCov2_xx * dCov2_yy <= 0` or its square root
is below `1e-12`. Cost O(n^2) time and O(n) memory per row. The weighted
form is the Szekely-Rizzo statistic with every mean weighted by `w`: row and
column means normalized by `sum(w)` and the grand mean by `sum(w)^2`.

#### 16.1.2 Chunked estimator

`dcorr2_chunked` is an estimator of dCorr2, not the statistic: with
`c = min(4096, floor(n / 2))` (exact statistic if `c < 4`), permute the
events with `Stream(seed, 0x20001)`, cut the permutation into consecutive
chunks of `c` events (a final partial chunk is dropped), compute the exact
statistic of each chunk with its own weights, and return
`psum(chunk values) / chunks`. `weighted_dcorr2` uses the exact statistic for
`n <= 10000` and the chunked estimator with seed 0 above (the threshold of
the Python reference; its permutation differs, see section 12.4).

### 16.2 Exact gradient

For scores `s` and a fixed protected variable `m` (derivation:
`docs/disco_gradient.md`), with `D = dCov2(s, m)`, `Vs = dCov2(s, s)`,
`Vm = dCov2(m, m)`, `den = sqrt(Vs * Vm)` and `c = D / Vs`:

```
g_k = ((2 * v_k) / den) * psum_j( (v_j * sign(s_k - s_j)) * (B_kj - c * A_kj) )
```

with `sign(0) = 0` (a subgradient at ties) and `g = 0` wherever dCorr2 is
defined as 0. The returned value is computed exactly as in 16.1.1.

#### 16.2.1 Pearson-proxy fast mode

`pearson_proxy_grad` is the gradient of the squared weighted Pearson
correlation, as in the public Vikshep CLI: with `v_i = w_i / (psum(w) +
1e-12)`, weighted means `mu_s`, `mu_m`, `ds = s - mu_s`, `dm = m - mu_m`,
`cov = psum(ds dm v)`, `var_s = psum(ds^2 v) + 1e-12`,
`var_m = psum(dm^2 v) + 1e-12`, `sd = sqrt(var_s var_m)`, `r = cov / sd`:
`g_i = 2 r v_i (dm_i / sd - r ds_i / var_s)`. It captures linear dependence
only and is labelled a fast mode wherever it is offered.

### 16.3 Jensen-Shannon divergence

Binning (fixed range): `lo` and `hi` are the minimum and maximum of the
reference sample (the pre-cut background protected variable); `n_bins`
equal-width bins; `bin(v) = 0` if `hi <= lo`, else with
`t = ((v - lo) / (hi - lo)) * n_bins`, `bin(v) = 0` if `t <= 0`, otherwise
`min(floor(t), n_bins - 1)`. A weighted histogram holds, per bin, the `psum`
of the weights of its events in event order. For histograms `p`, `q`:

```
P = p / psum(p);  Q = q / psum(q);  M_i = 0.5 * (P_i + Q_i)
KL(A || M) = psum_i( A_i == 0 ? 0 : A_i * ln(A_i / M_i) )
JSD = 0.5 * KL(P || M) + 0.5 * KL(Q || M)                 # nats, in [0, ln 2]
```

JSD is undefined (an error, reported as `null`) when either histogram is
empty.

### 16.4 Cut at a target signal efficiency

Sort the signal events by score descending (section 15.5); accumulate their
weights sequentially in that order until the running sum reaches
`target * psum(signal weights)`; the score reached is the threshold
`t_cut`. Every event with `score >= t_cut` passes. Passing signal `s` and
background `b` are `psum`s of passing weights in event order; efficiencies
are `s / psum(signal weights)` and `b / psum(background weights)`. Default
target: 0.5.

### 16.5 Asimov significance proxy

**Asimov proxy, not a Wilks fit**: a closed-form expected significance from
the weighted counts of one cut, not a likelihood-ratio test:

```
Z_A = sqrt(2 * max((s + b) * ln(1 + s / b) - s, 0))     for s >= 0, b > 0
```

and `null` (undefined) when `b <= 0`. Every report labels it as above.

### 16.6 Lambda-frontier rows

For a model with DisCo strength `lambda`, scored on an evaluation set: the
cut of 16.4, `Z_A` of 16.5, the JSD of 16.3 between the pre-cut and post-cut
background histograms of the protected variable (`n_bins` default 20; `null`
when no background passes), and `dcorr2_bkg`, the weighted dCorr2 of score
and protected variable on the background events (16.1). Fields:
`lambda, threshold, s, b, sig_eff, bkg_eff, z_asimov, jsd, dcorr2_bkg`.

### 16.7 Benchmark report

`report.json` (canonical JSON, section 15.6) has the members `config`,
`frontier` (rows in sweep order), `inputs` (dataset and model OIDs, event
counts), `kind = "vikshep.benchmark_report"`, `numerics_version`,
`significance_method = "Asimov proxy, not a Wilks fit"`, `tier2_version`
and `win_condition` (`rule`, `lambda_star`, `delta_sigma`, `delta_jsd`,
`result`). The **win condition** is evaluated at `lambda_star` (a lambda of
the sweep) against the unique `lambda = 0` row:

```
Delta-sigma = Z_A(lambda_star) - Z_A(0)
Delta-JSD   = JSD(lambda_star) - JSD(0)
result      = TRUE  if Delta-sigma > 0 AND Delta-JSD <= 0,  else FALSE
```

`result` is `FALSE` when either difference is undefined. `report.md`
renders the configuration and inputs as canonical JSON, the frontier table
(numbers as in section 15.6) and the verdict line `Result: **TRUE**` or
`Result: **FALSE**`. The report bytes are part of the conformance suite
(section 11.5) and identical across platforms.

---

## 17. Training

Implementation: `crates/vikshep-train`.

### 17.1 Data and standardization

A dataset holds `n` events with `d` frozen features (row-major binary64),
labels (1 signal, 0 background), non-negative weights and the protected
variable; its canonical bytes are `x`, `y` (one byte each), `w`, `m`. Features
are standardized as in the public CLI: `mu_j = psum_i(x_ij) / n`,
`std_j = sqrt(psum_i((x_ij - mu_j)^2) / n) + 1e-8`,
`x'_ij = (x_ij - mu_j) / std_j`. Features from C1 ratios: per event and r2
path, `psum` over output positions of the ratio divided by their count.

### 17.2 Heads and initialization

* Logistic: `z = b + dot(theta, x')`; parameters `[b, theta_0..theta_{d-1}]`.
* MLP with `H` hidden tanh units: `a_h = b1_h + dot(W1_h, x')`,
  `z = b2` then `z += W2_h * tanh(a_h)` for `h = 0..H-1`; parameters
  `[W1 (H x d, row-major), b1 (H), W2 (H), b2]`.

Initialization from `Stream(seed, 0x30000)`, drawing standard normals `z`
in parameter order: logistic `theta_j = 0.01 z`, `b = 0`; MLP
`W1 = sqrt(1/d) z`, `b1 = 0`, `W2 = sqrt(1/H) z`, `b2 = 0`.

### 17.3 Activations

`sigmoid(z) = 1 / (1 + exp(-z))` for `z >= 0` and `e / (1 + e)`,
`e = exp(z)`, for `z < 0`; `tanh(a) = sign(a) (1 - e) / (1 + e)`,
`e = exp(-2 |a|)`; `softplus(z) = max(z, 0) + ln(1 + exp(-|z|))`.

### 17.4 DisCo loss and gradient

For a minibatch `B` (events in batch order), `v_i = w_i / psum_B(w)` (a
batch with non-positive total weight contributes loss 0 and gradient 0):

```
L     = psum_B( v_i * (softplus(z_i) - y_i * z_i) ) + lambda * R
dz_i  = v_i * (yhat_i - y_i)                                yhat = sigmoid(z)
dz_k += ((lambda * g_k) * yhat_k) * (1 - yhat_k)            background k in B
grad_p = psum_B( dz_i * dz_i/dparam_p )
```

The dependence term uses the background events of the batch (at least two,
and `lambda != 0`): with exact gradients `(R, g)` = dCorr2 of their scores and
protected values with their weights and its exact gradient (16.2); in the
Pearson-proxy fast mode `g` is the proxy (16.2.1) and `R` is not added to the
logged loss.

### 17.5 Adam

Constants `beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8`, learning rate `lr`
(default 0.01). Step `t` (from 1), per parameter, with running products
`beta1^t`, `beta2^t` (no `powf`):

```
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * (g * g)
p = p - lr * (m / (1 - beta1^t)) / (sqrt(v / (1 - beta2^t)) + eps)
```

### 17.6 Schedule and outputs

Fit the standardization, initialize, then for epoch `e = 0..epochs-1`
permute the events with `Stream(seed, 0x310000000 + e)` and process
consecutive minibatches of `batch_size` events (the last may be smaller),
one Adam step each. The epoch loss is the mean of the batch losses
(section 15.3). Defaults: 20 epochs, batch 256, `lr = 0.01`, `lambda = 0`,
exact gradients. A trained model's canonical bytes are `mu`, `std`, the
parameters and the epoch losses (binary64 LE); predictions are
`sigmoid(z)` of standardized rows.

### 17.7 Lambda sweep and benchmark

For each lambda of the sweep (which MUST contain 0 exactly once), train with
the same configuration and seed on the training set, score the evaluation
set, and build its frontier row (16.6). The benchmark report (16.7) records
the configuration (`batch_size`, `epochs`, `gradient`, `head`,
`lambda_star`, `lambdas`, `lr`, `n_bins`, `seed`, `target_sig_eff`) and the
inputs (training and evaluation OIDs and sizes, feature count, model OIDs).

### 17.8 Synthetic benchmark sample

Drawn sequentially from `Stream(seed, 0x50000 + split)` (wrapping), with
`d >= 2` features; per event: `u = next_f64_unit()`, signal if `u < 0.3`;
protected variable `m = 120 + 8 z` (signal) or `50 + 150 u'` (background);
features `x0 = (signal ? 1 : 0) + z`, `x1 = 0.04 (m - 100) + z`,
`x2..x_{d-1} = z`; weight 1 (signal) or `0.5 + u''` (background). `x1` makes
an unconstrained classifier sculpt the background `m` distribution, which
the DisCo term counteracts.

---

## 18. Calibration

`vikshep_train::calibrate::ridge`: ridge regression in closed form, as in the
public CLI's `train_calibrate` (ridge `1e-4` by default):

1. Standardize the features (17.1).
2. `G_jk = psum_i(x'_ij x'_ik) + ridge [j = k]`, `c_j = psum_i(x'_ij t_i)`.
3. Cholesky `G = L L^T`: for `i` ascending and `j = 0..=i`,
   `s = G_ij - sum_{k<j} L_ik L_jk` (sequential), `L_ii = sqrt(s)` (an error
   unless `s > 0`), `L_ij = s / L_jj`.
4. Forward substitution `L y = c`, backward substitution `L^T w = y`
   (sequential inner sums).
5. `bias = psum(t) / n - psum_i(dot(x'_i, w)) / n`; residuals
   `t_i - (dot(x'_i, w) + bias)`; `r2 = 1 - ss_res / (ss_tot + 1e-12)`;
   `residual_std` = population standard deviation of the residuals.

Exports: the residual tensor (binary64 LE, with its OID) and the calibration
constants as canonical JSON (`bias`, `coef`, `kind = "vikshep.calibration"`,
`mu`, `r2`, `residual_std`, `residuals_oid`, `std`).

---

## 19. Anomaly search

Implementation: `crates/vikshep-anomaly`.

### 19.1 Fingerprint distributions

An event's fingerprint distribution is the equal-weight empirical
distribution of its per-position log-coefficient vectors: for each output
position `s` of the scattering output, the point
`f_s = (ln(2^-20 + |c_{p,s}|))_p` with one coordinate per path `p`. Its mean
is exactly the log-mean fingerprint of section 14.9. A single log-mean
vector (a one-point cloud) is also accepted.

The public Vikshep repository has no normative SW1 definition (its web
explorer uses an 8-angle illustration), so this section is the definition.

### 19.2 Sliced Wasserstein-1

Directions: `P` unit vectors in `R^D` from `Stream(seed, 0x40001)`: direction
`k` has components `g_kc = next_normal_f64()` (drawn for `k`, then `c`) divided
by `sqrt(psum_c(g_kc^2))` (the first basis vector if the norm is 0).
Projection: sequential dot product of each point with each direction; the
projections of a cloud are sorted ascending (section 15.5).

1-D W1 between equal-weight empirical distributions with sorted samples
`a` (n values) and `b` (m values) is `integral |F_a(z) - F_b(z)| dz`,
computed by merging the samples (on equal values `a` first) and summing, by
`psum`, the terms `(|i m - j n| / (n m)) * (z_next - z)` over consecutive
merged values, where `i`, `j` count the values of `a`, `b` already passed.

`SW1(A, B) = psum_k( W1(A_k, B_k) ) / P`. Default `P = 32`.

### 19.3 HNSW index

Malkov and Yashunin's hierarchical navigable small world graph (2018,
Algorithms 1 and 2) with the simple neighbour selection, made deterministic:

* insertion single-threaded, in input order;
* node level `floor(-ln(u) / ln(M))`, `u = 1 - next_f64_unit()` from
  `Stream(seed, 0x40002)`, drawn per inserted node, capped at 16;
* at insertion, greedy descent with `ef = 1` from the top level to the
  node's level + 1, then on each layer from `min(level, top)` down to 0 a
  search with `ef_construction`; the node links to the `M` nearest found;
  each new neighbour links back, and a neighbour list longer than its cap
  (`2M` on layer 0, `M` above) is cut to the cap nearest by
  `(distance, id)`;
* SEARCH-LAYER keeps candidates and results ordered by `(distance, id)`
  (section 15.5) and stops when the nearest candidate is worse than the
  worst result;
* `knn(q, k)`: greedy descent to layer 1, then a layer-0 search with
  `max(ef_search, k)`; the first `k` results.

Defaults: `M = 8`, `ef_construction = 64`, `ef_search = 32`. Canonical graph
bytes (`u32` little-endian): node count, entry point, top level, then per
node its level and, for each layer `0..=level`, the neighbour count followed
by the neighbour ids in stored order.

### 19.4 Detection

A query is flagged when its distance to the `k`-th nearest reference
exceeds `tau`. For the reference set itself (leave-one-out), each event
searches `k + 1` neighbours and ignores itself. With fewer than `k`
neighbours the distance is undefined (`NaN` in tensors, section 2.6) and the
event is not flagged.

### 19.5 Layout

Fruchterman-Reingold in the unit square, fixed iteration count `T`
(default 100): initial positions `(u, u)` per node from
`Stream(seed, 0x40003)`; `k = sqrt(1 / n)`; at iteration `t` the
temperature is `0.1 (1 - t / T)`; every ordered pair repels with force
`k^2 / d` (sums in increasing `j`); every edge attracts with `d^2 / k`
(edges in output order); each node moves along its displacement by at most
the temperature; coordinates are clamped to `[0, 1]`; distances below
`1e-9` are raised to `1e-9`.

### 19.6 Output contract

The graph output never contains features or fingerprints. Canonical JSON
(section 15.6):

```
{ "config_digest": "<SHA3-256 of the canonical configuration>",
  "edges": [ {"i": int, "j": int, "sw1": number}, ... ],
  "nodes": [ {"flagged": bool, "id": int, "x": number, "y": number}, ... ],
  "numerics_version": 1, "tau": number, "tier2_version": 1 }
```

Nodes are the reference events (leave-one-out flags) followed by optional
queries (ids after the references, flagged against the references). Edges
are the undirected layer-0 HNSW links `(i, j)`, `i < j`, sorted, followed by
each query's links to its `k` nearest references. The configuration object
hashed into `config_digest` has the members `directions`,
`distance = "sliced_wasserstein_1"`, `ef_construction`, `ef_search`,
`fingerprint = "log_coefficients_per_position"`, `k`, `layout_iterations`,
`m`, `seed` (decimal string) and `tau`.

---

## Appendix A: References

- IEEE 754-2019, Standard for Floating-Point Arithmetic.
- J. K. Salmon, M. A. Moraes, R. O. Dror, D. E. Shaw, "Parallel random
  numbers: as easy as 1, 2, 3", SC 2011. Random123 distribution:
  https://github.com/DEShawResearch/random123
- G. L. Steele Jr., D. Lea, C. H. Flood, "Fast splittable pseudorandom number
  generators", OOPSLA 2014.
- S. Vigna, `splitmix64.c` (public domain, 2015).
- NIST FIPS 202, SHA-3 Standard.
- RFC 8785, JSON Canonicalization Scheme.
- `libm` crate 0.2.16, https://github.com/rust-lang/compiler-builtins
  (MIT licence; derived from musl libc).
- mpmath, https://mpmath.org (BSD licence; used to generate test fixtures).
- Kymatio 0.3.0, https://www.kymat.io (BSD-3-Clause; filter parameterization
  source and test oracle only).
- J. Bruna, S. Mallat, "Invariant scattering convolution networks", IEEE
  TPAMI 35(8), 2013.
- RFC 2119, RFC 8174 (requirement key words).
- G. J. Szekely, M. L. Rizzo, N. K. Bakirov, "Measuring and testing
  dependence by correlation of distances", Annals of Statistics 35(6), 2007.
- G. Kasieczka, D. Shih, "Robust jet classifiers through distance
  correlation", Physical Review Letters 125, 122001 (2020).
- G. Cowan, K. Cranmer, E. Gross, O. Vitells, "Asymptotic formulae for
  likelihood-based tests of new physics", EPJ C 71, 1554 (2011).
- D. P. Kingma, J. Ba, "Adam: a method for stochastic optimization", ICLR 2015.
- N. Bonneel, J. Rabin, G. Peyre, H. Pfister, "Sliced and Radon Wasserstein
  barycenters of measures", J. Math. Imaging Vision 51, 2015.
- Y. A. Malkov, D. A. Yashunin, "Efficient and robust approximate nearest
  neighbor search using hierarchical navigable small world graphs", IEEE
  TPAMI 42(4), 2020 (arXiv 2016).
- T. M. J. Fruchterman, E. M. Reingold, "Graph drawing by force-directed
  placement", Software: Practice and Experience 21(11), 1991.
- Public Vikshep repository, https://github.com/samvardhan03/Vikshep
  (Python reference metric for section 12.4).
