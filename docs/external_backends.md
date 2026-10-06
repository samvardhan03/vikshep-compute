# External backends

A backend executes the five Tier-1 kernels of `spec/VDS-1.md` section 14.6
(`fft`, `ifft`, `mul_real_filter`, `modulus`, `subsample`); everything else
(filters, twiddle tables, pooling, r2, reductions, statistics) stays in the
host code of this repository. This repository contains only the CPU
reference backend. A proprietary backend (for example CUDA or Metal) lives
in its own repository and build, and plugs in through the C ABI in
`include/vikshep_backend.h` without any change here.

## The table

A backend fills a `VkspBackendV1` (`abi_version = VKSP_ABI_VERSION`):

| Field | Meaning |
|---|---|
| `ctx` | backend-owned context, passed to every call |
| `name(ctx)` | NUL-terminated name, valid as long as the table (not `cpu`, which is reserved) |
| `numerics_version(ctx)` | must equal the library's `numerics_version` (1) |
| `capabilities(ctx)` | largest transform length, parallelism, subnormal handling |
| `fft`, `ifft`, `mul_real_filter`, `modulus`, `subsample` | the kernels, with the exact arithmetic of VDS-1 sections 6 to 8 and 14.6 |
| `destroy(ctx)` | releases `ctx` (may be null) |

Buffer ownership, alignment and synchronicity are documented in the header
and in the crate documentation of `vikshep-capi`: the caller owns every
buffer, kernels must not keep pointers, and results are in host memory when a
kernel returns `VKSP_OK`. A device backend copies to and from the device
inside each kernel (or keeps its own staging buffers in `ctx`).

## Arithmetic under VDS-1.1 (flush-to-zero)

`numerics_version` is 2 (VDS-1.1, `spec/VDS-1.md` section 8.1). A backend
reporting `numerics_version() == 2` must reproduce these rules exactly:

1. Treat every binary32 subnormal operand of `+`, `-`, `*` and `sqrt` as the
   zero of the same sign, and replace every subnormal **rounded** result by
   the zero of the same sign: `ftz(ftz(a) op ftz(b))`, `ftz(sqrt(ftz(a)))`.
   This applies to every butterfly sum, difference and product of the FFT,
   the `2^-m` scaling of the inverse, the filter products, and both squares,
   the sum and the root of the modulus. `subsample` copies values.
2. Judge tininess after rounding: a result that rounds to exactly FLT_MIN
   (`2^-126`) is kept. Hardware that flushes based on the unrounded value
   differs here; the `round_to_flt_min` cases of the suite detect it.
3. Keep the sign of a flushed zero, and keep IEEE zero-sign rules in later
   operations.
4. Never contract into fused multiply-add (VDS-1 section 2), and use a
   correctly rounded `sqrt`.

The host flushes inputs and tables (filters, twiddles) before calling a
kernel, so a backend always receives flush-free buffers; the rules above are
about the results it computes. The simplest correct implementation flushes
in software after every operation, as the CPU reference does
(`vikshep_numerics::flush::ftz`, a test of the exponent field). A backend may
rely on its hardware's own flush-to-zero mode instead only where that mode
is proven identical by the suite: it must pass 100% of suite v2
(`vksp_conformance_selftest("full", ...)`), including the `subnormal`,
`flt_min_band`, `sqrt_flt_min_band`, `round_to_flt_min`, `long_tail` and
`signed_zero` cases. No backend can claim conformance against the version 1
vectors (`conformance/vectors/v1/`, subnormals preserved).

## Registration

```c
#include "vikshep.h"            /* includes vikshep_backend.h */

VkspBackendV1 table = my_cuda_backend_v1();   /* exported by the private library */
if (vksp_register_backend(&table) != VKSP_OK) { /* table still owned by the caller */ }

/* Every host function that takes a backend name can now use it: */
vksp_scatter(&cfg, "cuda", input, input_len, out, out_cap, &out_len, oid);
vksp_conformance_selftest("full", "cuda", &cases, &failed, NULL, 0, NULL);

vksp_unregister_backend("cuda");             /* destroy(ctx) runs once unused */
```

`vksp_register_backend` copies the table and, on success, owns it. It rejects
a table with another `abi_version`, a missing kernel or metadata function, a
different `numerics_version`, an empty, reserved (`cpu`) or already
registered name. Calls into one registered backend are serialized by the
library, so a backend does not need to be thread-safe.

A host typically loads the proprietary library at run time (`dlopen` /
`LoadLibrary`), looks up its table constructor, and registers the table.
Rust hosts can do the same through `vikshep_capi::vksp_register_backend` or
wrap a table directly with `vikshep_capi::ForeignBackend`, which implements
the `ScatterBackend` trait.

## Proving conformance

A backend conforms when it reproduces the CPU reference bit for bit on
conformance suite v1 (VDS-1 section 11). After registration:

```c
size_t cases, failed;
int32_t s = vksp_conformance_selftest("full", "cuda", &cases, &failed, NULL, 0, NULL);
/* s == VKSP_OK: every case passed; VKSP_CONFORMANCE_FAILED otherwise */
```

The expected vectors are compiled into the library, so the self-test needs
no files. With a report buffer, the JSON report names the first differing
element of every failing output, with its bits and ulp distance. The quick
subset (`"quick"`, 256 cases) runs in well under a second on a CPU; `"full"`
runs all 402 cases of suite v2.

`examples/c/vikshep_example.c` registers a backend that forwards every kernel
to the CPU table and counts the calls, runs a scattering transform and the
self-test through it, and checks that the result is identical to the
reference.
