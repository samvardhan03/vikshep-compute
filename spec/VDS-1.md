# VDS-1: Vikshep Determinism Specification, version 1

| | |
|---|---|
| Status | Draft. Complete for `numerics_version = 1` except section 14 (scattering), reserved for milestone C1 |
| Numerics version | 1 |
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

### 3.3 Metal (future private backend)

- Fast math MUST be disabled (safe math mode), floating-point contraction
  MUST be disabled, and only precise functions MAY be used.
- The exact compiler options and source-level mechanism (for example
  `-fno-fast-math` / `MTLCompileOptions` math mode settings, `precise::`
  functions, and a contraction pragma) are to be verified on hardware and
  recorded here by the Metal backend work. Until recorded, no Metal backend
  can claim conformance.

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
order or in parallel.

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

---

## 8. Subnormals

### 8.1 Decision D-FTZ: OPEN

Default for the CPU reference: IEEE subnormals are preserved (no
flush-to-zero of inputs or outputs). Rust on x86_64 and AArch64 preserves
subnormals by default and the reference never changes the floating-point
environment.

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
serialized with the JSON Canonicalization Scheme (RFC 8785). Manifests never
carry result floats: results are tensors referenced by their OIDs.

---

## 10. Versioning

`numerics_version` is an integer, starting at 1 (`NUMERICS_VERSION` in
`vikshep-numerics`). Any change to arithmetic order, tables, algorithms,
constants, generator definitions, or the pinned `libm` version that changes
any output bit MUST bump it. Conformance vectors are keyed by
`numerics_version` (`conformance/vectors/v<version>/`). Within one
`numerics_version`, cases MAY be added, but the expected output of an existing
case MUST NOT change.

---

## 11. Conformance

A backend conforms to VDS-1 at a given `numerics_version` if and only if it
reproduces 100% of that version's conformance cases bit-exactly. Partial
conformance is non-conformance. The conformance runner is the
`vikshep-conformance` binary; at numerics version 1 its cases are the sweeps
of sections 4.4 and 5.6, and milestone C1 adds FFT and scattering cases.

---

## 12. Correctness oracles

Determinism (identical bytes everywhere) and correctness (the bytes are
right) are tested separately.

- **Tolerance tests against an independent binary64 implementation.**
  Kymatio (BSD-3-Clause) is used as a test oracle only; it is not a
  dependency of, and is not distributed with, any shipped crate.
- **Property tests:** Parseval / Littlewood-Paley energy bounds, translation
  behaviour, non-expansiveness, homogeneity.
- **Low-level references:** detmath against mpmath (section 4.3), the FFT
  recursion against a naive DFT (section 6.4), the generators against
  published known answers (section 5.5).

---

## 13. Open decisions register

| Id | Question | Status | Default / resolution path |
|---|---|---|---|
| D-MATH | Source of portable transcendental functions | Decided: Option A (`libm =0.2.16`, no features) | Revisit only if the cross-platform sweep fails (section 4.2) |
| D-FTZ | Preserve subnormals or flush to zero on every backend | Open | Preserve; procedure in section 8.2 |
| D-SQRT | Is `sqrt` correctly rounded on each GPU backend? | Open | CPU: correctly rounded (IEEE-754 requires it; Rust lowers to the hardware square-root instruction). CUDA: `--prec-sqrt=true` expected to give a correctly rounded `sqrt`, to be verified. Metal: to be measured. Fallback for any backend that fails: a Markstein-style square root using an exact fused multiply-add and a final correction step, proven correctly rounded; this is the only place an FMA could be admitted, and only by amendment of this specification |
| D-DIV | Division in Tier 1 | Decided: no division in Tier 1 | r2 ratios and any other quotient are computed on the host (Tier 2) |
| D-STEER | Steerable-basis orientation synthesis | Open | Changes arithmetic; allowed only if the reference adopts it under a new `numerics_version` |

---

## 14. Scattering transform

### 14.1 Morlet filter bank

Specified in C1.

### 14.2 Gaussian low-pass filter

Specified in C1.

### 14.3 Cascade (orders 0, 1, 2)

Specified in C1.

### 14.4 Output layout

Specified in C1.

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
- Kymatio, https://www.kymat.io (BSD-3-Clause; test oracle only).
