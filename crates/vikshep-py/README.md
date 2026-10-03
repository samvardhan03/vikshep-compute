# vikshep-compute (Python)

Python bindings of the open, deterministic compute core of Vikshep:
wavelet scattering, r2 and fingerprints, weighted distance correlation,
DisCo training, calibration, the benchmark report, Sliced Wasserstein-1
anomaly search, provenance manifests and the conformance self-test. Results
are bit-identical on every supported platform (specification VDS-1).

Distribution name `vikshep-compute`, import name `vikshep_compute`.

```python
import numpy as np
import vikshep_compute as vc

x = np.random.default_rng(0).standard_normal((4, 256)).astype(np.float32)
cfg = {"J": 4, "Q": 1, "pad": "zero_pad"}
coeffs, coeffs_oid = vc.scatter(x, cfg)          # float32 (4, paths, 16)
cfg["shape"] = [256]
ratios, r2_oid = vc.r2(coeffs, cfg)
fp, fp_oid = vc.fingerprint(coeffs, cfg)         # float64 (4, paths)
manifest, manifest_hash = vc.provenance_manifest(x, cfg, coeffs_oid)
print(vc.conformance_selftest("quick")["failed"])  # 0
```

Inputs are converted with numpy and copied in row-major order into buffers
owned by the library before any computation, so strides, memory order and
byte order never change a result.

Licence: AGPL-3.0-or-later. Source, specification and conformance vectors:
https://github.com/samvardhan03/vikshep-compute
