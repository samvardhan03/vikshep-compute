# Releasing

A release publishes the open core in three forms, all from one version tag:

| Form | Where | Install |
|---|---|---|
| Python package `vikshep-compute` (abi3 wheels, Python 3.10+, five targets) | PyPI | `pip install vikshep-compute` |
| Rust crates (`vikshep` plus the nine crates it re-exports or the data plane needs) | crates.io | `cargo add vikshep` |
| C/C++ archives: headers, static and shared C API library, `vikshep-mcp`, `vikshep-conformance`, conformance vectors | GitHub release | download, then link `lib/` and include `include/` |

Everything is built by `.github/workflows/release.yml`. A pull request that
touches the release files, or a manual run (Actions, Release, Run workflow),
builds and tests all of it and publishes nothing. Only a `vX.Y.Z` tag
publishes, and only after every build succeeded.

## Not published to crates.io

`vikshep-conformance-core` and `vikshep-conformance` embed the conformance
vectors from `conformance/`, which is outside their crate directories, so
`cargo package` cannot include them; `vikshep-capi` depends on them, and
`vikshep-py` is the Python package. These keep `publish.workspace = true`
(the workspace default, `publish = false`). The C API and the conformance
runner ship in the C/C++ archives instead; certificate verification is also
in the Python package (`vikshep_compute.verify_certificate`). A crate added
later is unpublished until it drops `publish.workspace = true`.

No source distribution goes to PyPI: the Python package is built from the
workspace, which an sdist of `crates/vikshep-py` alone does not contain.
Platforms without a wheel build from a checkout (README, Quickstart: Python).

## One-time setup

1. **PyPI** (trusted publishing, no stored token): sign in at pypi.org,
   Account, Publishing, "Add a new pending publisher": project
   `vikshep-compute`, owner `samvardhan03`, repository `vikshep-compute`,
   workflow `release.yml`, environment `pypi`.
2. **GitHub environments**: Settings, Environments, create `pypi` and
   `crates-io`. Adding yourself as a required reviewer on both makes every
   publish wait for your approval.
3. **crates.io**: sign in with GitHub at crates.io, Account Settings, API
   Tokens, New Token, scopes `publish-new` and `publish-update`, crate
   pattern `vikshep*`, with an expiry. Store it as the secret
   `CARGO_REGISTRY_TOKEN` of the `crates-io` environment (not as a
   repository secret). Once the crates exist, crates.io trusted publishing
   can replace the token.

## Procedure

1. On a branch: set the new version in `[workspace.package]` of `Cargo.toml`
   and in every internal pin (`version = "=X.Y.Z"` in
   `[workspace.dependencies]`), run `cargo update --workspace`, and update
   STATUS.md. A version that changes output bytes also follows
   `spec/VDS-1.md` section 10. Merge after CI and the Release dry run pass.
2. Tag the merge commit and push the tag:
   `git tag -a vX.Y.Z -m "vikshep-compute X.Y.Z"` then
   `git push origin vX.Y.Z`.
3. Approve the `pypi` and `crates-io` deployments if reviewers are set.
4. Check from a clean environment:
   - `python -m venv /tmp/v && /tmp/v/bin/pip install vikshep-compute==X.Y.Z && /tmp/v/bin/python -c "import vikshep_compute as vc; print(vc.conformance_selftest('quick')['failed'])"` prints `0`;
   - in a new cargo project, `cargo add vikshep@=X.Y.Z && cargo build`;
   - unpack an archive and run `bin/vikshep-conformance run --vectors conformance/vectors/v3`.

PyPI and crates.io versions are permanent: a broken release is yanked and
fixed by a new version, never replaced.
