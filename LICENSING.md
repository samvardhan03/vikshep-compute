# Licensing

Copyright (C) 2026 Samvardhan Singh.

This repository uses two licences, chosen by path.

| Path | Licence | Text |
|---|---|---|
| `spec/` | Creative Commons Attribution 4.0 International (CC-BY-4.0) | [`LICENSES/CC-BY-4.0.txt`](LICENSES/CC-BY-4.0.txt) |
| `conformance/vectors/` | Creative Commons Attribution 4.0 International (CC-BY-4.0) | [`LICENSES/CC-BY-4.0.txt`](LICENSES/CC-BY-4.0.txt) |
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

Test-only references, not distributed with any crate: mpmath (BSD) generates
the `vikshep-detmath` accuracy fixture; Kymatio (BSD-3-Clause) will serve as a
correctness oracle from milestone C1.

## Contributions

See [`CONTRIBUTING.md`](CONTRIBUTING.md): outside contributions require a
Contributor License Agreement.
