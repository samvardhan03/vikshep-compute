"""Deterministic compute core of Vikshep (VDS-1).

Wavelet scattering, r2 and fingerprints, weighted distance correlation,
DisCo training, calibration, the benchmark report, Sliced Wasserstein-1
anomaly search, provenance manifests and the conformance self-test. Every
result is bit-identical on every supported platform; arrays come back as
numpy arrays together with their OIDs (first 14 bytes of SHA3-256 of the
little-endian bytes, 28 hex characters).

Inputs are converted with ``numpy.asarray(..., dtype=float32 or float64)``
and copied in row-major order into buffers owned by the library before any
computation, so strides, memory order and byte order of the caller's arrays
never change a result.
"""

from ._core import (
    NUMERICS_VERSION,
    PLATFORM,
    TIER2_VERSION,
    AnomalyIndex,
    TaggerModel,
    __version__,
    _conformance_scatter_case,
    anomaly_index,
    anomaly_query,
    bench_report,
    calibrate,
    conformance_selftest,
    dcorr2_w,
    fingerprint,
    fingerprint_clouds,
    oid,
    provenance_manifest,
    r2,
    scatter,
    scatter_paths,
    synthetic_dataset,
    train_tag,
)

__all__ = [
    "NUMERICS_VERSION",
    "PLATFORM",
    "TIER2_VERSION",
    "AnomalyIndex",
    "TaggerModel",
    "__version__",
    "anomaly_index",
    "anomaly_query",
    "bench_report",
    "calibrate",
    "conformance_selftest",
    "dcorr2_w",
    "fingerprint",
    "fingerprint_clouds",
    "oid",
    "provenance_manifest",
    "r2",
    "scatter",
    "scatter_paths",
    "synthetic_dataset",
    "train_tag",
]
