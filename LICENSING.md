# Licensing

Copyright (C) 2026 Samvardhan Singh.

This repository uses two licences, chosen by path.

| Path | Licence | Text |
|---|---|---|
| `spec/` | Creative Commons Attribution 4.0 International (CC-BY-4.0) | [`LICENSES/CC-BY-4.0.txt`](LICENSES/CC-BY-4.0.txt) |
| `conformance/vectors/` | Creative Commons Attribution 4.0 International (CC-BY-4.0) | [`LICENSES/CC-BY-4.0.txt`](LICENSES/CC-BY-4.0.txt) |
| `conformance/cases.toml` | Creative Commons Attribution 4.0 International (CC-BY-4.0) | [`LICENSES/CC-BY-4.0.txt`](LICENSES/CC-BY-4.0.txt) |
| Everything else (source code, build files, scripts, tests and test fixtures, CI configuration, documentation) | GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later) | [`LICENSE`](LICENSE) |

The specification and conformance vectors are under CC-BY-4.0 so that anyone
can implement and test against VDS-1 independently, with attribution.

## Commercial licence

The code is also available under a commercial licence from the copyright
holder, for use where the terms of the AGPL are not suitable. Contact:
shekhawatsamvardhan@gmail.com.

## Third-party components

Dependencies keep their own licences. Current direct dependencies:

| Crate | Licence | Use |
|---|---|---|
| `libm` 0.2.16 | MIT | portable transcendental functions inside `vikshep-detmath` |
| `sha3` 0.12.0 | MIT OR Apache-2.0 | SHA3-256 for OIDs and conformance hashes |
| `rayon` 1.12.0 | MIT OR Apache-2.0 | parallelism across independent canvases (CPU backend, filter construction) |
| `serde` 1.0.229, `serde_json` 1.0.151, `toml` 1.1.6 | MIT OR Apache-2.0 | conformance runner (case definitions, vectors, reports) |
| `criterion` 0.8.2 (benchmarks only) | MIT OR Apache-2.0 | throughput measurement |

Tools, not dependencies: cbindgen 0.29.4 (MPL-2.0) generates
`include/vikshep_backend.h` (the generated header is covered by this
repository's AGPL-3.0-or-later licence, like the Rust source it is generated
from).

The Morlet filter parameterization of `vikshep-scatter` follows Kymatio 0.3.0
(BSD-3-Clause, Copyright (c) 2018-, The Kymatio developers); the formulas
are re-implemented and documented in `spec/VDS-1.md` section 14, with the
deviations listed there.

Test-only references, not distributed with any crate: mpmath (BSD) generates
the `vikshep-detmath` accuracy fixture; Kymatio 0.3.0 with numpy and scipy
(BSD-3-Clause) generates the correctness-oracle fixtures in
`oracles/fixtures/` (`oracles/kymatio_fixtures.py`).

## Contributions

See [`CONTRIBUTING.md`](CONTRIBUTING.md): outside contributions require a
Contributor License Agreement.
