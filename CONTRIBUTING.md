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
cargo test --workspace --locked
scripts/determinism-lint.sh
cargo run --release -p vikshep-conformance -- check
cargo run --release -p vikshep-conformance -- run --backend cpu
```

If you change `crates/vikshep-capi/src/lib.rs`, regenerate the C header with
cbindgen 0.29.4:

```sh
cbindgen --config crates/vikshep-capi/cbindgen.toml --crate vikshep-capi \
         --output include/vikshep_backend.h
```

The Kymatio oracle fixtures are regenerated only on purpose, never in CI:
`pip install kymatio==0.3.0 numpy scipy && python3 oracles/kymatio_fixtures.py`.

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
  (`vikshep-conformance generate`). Within one version, cases may be added
  but existing expected outputs must not change.
- Backends execute only the five kernels of VDS-1 section 14.6; pooling,
  ratios and reductions stay on the host.
- Exceptions to the determinism lint go in `scripts/determinism-lint.allow`,
  each with a justification.

## Style

No AI-attribution text in files, commits or pull requests. Stage explicit
paths only.
