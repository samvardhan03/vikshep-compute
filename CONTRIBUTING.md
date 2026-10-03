# Contributing

## Contributor License Agreement

Outside contributions require a signed Contributor License Agreement (CLA),
because the code is offered under both the AGPL and a commercial licence
(see [`LICENSING.md`](LICENSING.md)). No CLA has been published yet. Until one
is, pull requests from outside contributors will not be merged.

Bug reports, questions about the specification, and reports of
cross-platform mismatches are welcome as issues at any time.

## Development checks

Run these before opening a pull request. CI runs the same checks.

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy -p vikshep-py --features python --all-targets --locked -- -D warnings
cargo test --workspace --locked
scripts/determinism-lint.sh
cargo run --release -p vikshep-conformance -- check
cargo run --release -p vikshep-conformance -- run --backend cpu
```

If you change `crates/vikshep-capi/src`, regenerate both C headers with
cbindgen 0.29.4 (CI verifies them):

```sh
cbindgen --config crates/vikshep-capi/cbindgen.toml --crate vikshep-capi \
         --output include/vikshep_backend.h
cbindgen --config crates/vikshep-capi/cbindgen-vikshep.toml --crate vikshep-capi \
         --output include/vikshep.h
```

A new exported function belongs to one header: add it to the `exclude` list
of the other configuration.

Python package (needs a Rust toolchain and Python 3.10 or later):

```sh
python -m pip install ./crates/vikshep-py pytest jsonschema
python -m pytest crates/vikshep-py/tests
```

The C and C++ examples (CI builds and runs them on Linux and macOS):

```sh
cargo rustc --release -p vikshep-capi --lib -- --print native-static-libs
cc -std=c99 -Iinclude examples/c/vikshep_example.c target/release/libvikshep_capi.a <native libs> -o c_example
c++ -std=c++17 -Iinclude examples/cpp/vikshep_example.cpp target/release/libvikshep_capi.a <native libs> -o cpp_example
```

The MCP data plane: `cargo test -p vikshep-mcp` runs its integration test
(POSIX shared memory on Unix, the file store everywhere).

Oracle fixtures are regenerated only on purpose, never in CI:

```sh
# Kymatio correctness oracle
pip install kymatio==0.3.0 numpy scipy && python3 oracles/kymatio_fixtures.py
# Parity with the public Vikshep Python metric
git clone https://github.com/samvardhan03/Vikshep /path/to/Vikshep
PYTHONPATH=/path/to/Vikshep/backend/ingest/src python3 oracles/python_disco/gen_fixtures.py
# ECMAScript number formatting for canonical JSON (needs python3 and node)
bash oracles/jcs_numbers/gen.sh
```

## Numerics rules

These follow from [`spec/VDS-1.md`](spec/VDS-1.md):

- Never call `f32`/`f64` transcendental or power methods (`exp`, `ln`, `sin`,
  `cos`, `powf`, `powi`, `hypot`, ...) outside `vikshep-detmath`. Use
  `vikshep-detmath`. `sqrt` is allowed.
- Never use `mul_add` or fast/algebraic float intrinsics. Do not enable
  `clippy::suboptimal_flops`.
- Never change the floating-point environment.
- Reductions must have a fixed order that does not depend on thread count.
- Any change that alters an output bit (arithmetic order, tables, filters,
  algorithms, constants, generator definitions, the pinned `libm` version)
  requires a `numerics_version` bump and new conformance vectors
  (`vikshep-conformance generate`). A change confined to the Tier-2
  operations of VDS-1 sections 15 to 19 (statistics, training, calibration,
  anomaly search, report formats) bumps `tier2_version` instead. Within one
  version, cases may be added but existing expected outputs must not change.
- Tier-2 code sums over events with `vikshep_numerics::sum::pairwise_sum`,
  draws randomness only from Philox streams registered in VDS-1 section
  15.4, sorts with `f64::total_cmp` and an index tie-break
  (`vikshep_numerics::order`), and writes JSON only through
  `vikshep_numerics::jcs`.
- Backends execute only the five kernels of VDS-1 section 14.6; pooling,
  ratios and reductions stay on the host.
- Exceptions to the determinism lint go in `scripts/determinism-lint.allow`,
  each with a justification.
- Bindings add no arithmetic: they copy inputs into contiguous row-major
  buffers and call the reference code (VDS-1 section 20).
- No licence checks, telemetry or network calls in this repository.

## Style

No AI-attribution text in files, commits or pull requests. Stage explicit
paths only.
