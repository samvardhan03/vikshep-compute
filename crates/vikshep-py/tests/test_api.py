"""Behaviour of the Python API: input layout independence, manifests,
statistics, calibration and anomaly search."""

import json
from pathlib import Path

import numpy as np
import pytest

import vikshep_compute as vc

ROOT = Path(__file__).resolve().parents[3]
CFG = {"J": 4, "Q": 2, "pad": "zero_pad"}


def signals(n=3, length=256, seed=7):
    return np.random.default_rng(seed).standard_normal((n, length)).astype(np.float32)


def test_layout_and_dtype_do_not_change_results():
    x = signals()
    _, ref = vc.scatter(x, CFG)
    variants = [
        np.asfortranarray(x),                       # Fortran order
        np.repeat(x, 2, axis=1)[:, ::2],            # non-contiguous view
        x.astype(">f4"),                            # big-endian
        x.astype(np.float64),                       # float64, exactly representable
        x.tolist(),                                 # plain Python lists
    ]
    for v in variants:
        assert vc.scatter(v, CFG)[1] == ref
    # a batch is the signals back to back; each signal alone gives its slice
    coeffs, _ = vc.scatter(x, CFG)
    one, _ = vc.scatter(x[1], CFG)
    assert coeffs.shape[0] == 3 and one.shape[0] == 1
    assert one[0].tobytes() == coeffs[1].tobytes()


def test_config_validation():
    with pytest.raises(ValueError):
        vc.scatter(signals(), {"J": 4, "bogus": 1})
    with pytest.raises(ValueError):
        vc.scatter(signals(), {"J": 0})
    with pytest.raises(ValueError):
        vc.scatter(signals(length=255), {"J": 4})
    paths = vc.scatter_paths({**CFG, "shape": [256]})
    assert paths[0] == {"order": 0, "j": [], "theta": []}
    assert [p["order"] for p in paths] == sorted(p["order"] for p in paths)


def test_provenance_manifest():
    x = signals()
    coeffs, oid = vc.scatter(x, CFG)
    local_json, local_hash = vc.provenance_manifest(x, CFG, oid)
    cloud_json, cloud_hash = vc.provenance_manifest(
        x, CFG, oid, backend="cuda", backend_version="9", executor="cloud",
        started_unix_ms=1, finished_unix_ms=2,
    )
    assert local_hash == cloud_hash and local_json != cloud_json
    m = json.loads(local_json)
    assert m["manifest_hash"] == local_hash and m["outputs"][0]["oid"] == oid
    assert m["inputs"][0] == {"dtype": "float32", "name": "signal", "oid": vc.oid(x.tobytes()), "shape": [3, 256]}
    assert m["outputs"][0]["shape"] == list(coeffs.shape)
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "spec" / "provenance.schema.json").read_text())
    jsonschema.validate(m, schema)
    jsonschema.validate(json.loads(cloud_json), schema)


def test_dcorr2_w():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(500)
    assert vc.dcorr2_w(a, a) == pytest.approx(1.0, abs=1e-12)
    assert vc.dcorr2_w(a, rng.standard_normal(500)) < 0.05
    w = rng.uniform(0.1, 1.0, 500)
    assert vc.dcorr2_w(a, a * a, w) > 0.05
    with pytest.raises(ValueError):
        vc.dcorr2_w(a, a[:10])


def test_train_and_calibrate():
    data = vc.synthetic_dataset(800, 4, seed=3)
    m0 = vc.train_tag(data["x"], data["y"], data["w"], data["m"], epochs=5, batch_size=128)
    m1 = vc.train_tag(data["x"], data["y"], data["w"], data["m"], epochs=5, batch_size=128)
    assert m0.oid == m1.oid and m0.epoch_loss[-1] < m0.epoch_loss[0]
    scores = m0.predict(data["x"])
    assert scores.shape == (800,) and np.all((scores >= 0) & (scores <= 1))
    with pytest.raises(ValueError):
        vc.train_tag(data["x"], data["y"], lam=1.0)
    rng = np.random.default_rng(4)
    x = rng.standard_normal((300, 3))
    t = 2 * x[:, 0] - x[:, 1] + 0.5 * x[:, 2] + 1
    c = vc.calibrate(x, t)
    assert c["r2"] > 0.999999 and c["residuals"].shape == (300,)
    assert json.loads(c["constants_json"])["residuals_oid"] == c["residuals_oid"]


def test_anomaly_index_flags_an_outlier():
    rng = np.random.default_rng(5)
    refs = rng.standard_normal((60, 8, 4))
    index = vc.anomaly_index(refs, tau=1.0, k=5, seed=11)
    assert len(index) == 60 and len(index.config_digest) == 64
    queries = [rng.standard_normal((8, 4)), rng.standard_normal((8, 4)) + 6.0]
    r = vc.anomaly_query(index, queries)
    assert r["kth_distance"][1] > 4 * r["kth_distance"][0]
    assert list(r["flagged"]) == [False, True]
    graph = json.loads(r["graph_json"])
    assert len(graph["nodes"]) == 62 and graph["numerics_version"] == 1
    assert set(graph["nodes"][0]) == {"flagged", "id", "x", "y"}
    again = vc.anomaly_query(vc.anomaly_index(refs, tau=1.0, k=5, seed=11), queries)
    assert again["graph_json"] == r["graph_json"]


def test_fingerprint_clouds_from_coefficients():
    x = signals(n=2)
    coeffs, _ = vc.scatter(x, CFG)
    cfg = {**CFG, "shape": [256]}
    clouds = vc.fingerprint_clouds(coeffs, cfg)
    fp, _ = vc.fingerprint(coeffs, cfg)
    assert clouds.shape == (2, 16, coeffs.shape[1])
    # the mean of each distribution is the log-mean fingerprint
    assert np.allclose(clouds.mean(axis=1), fp, rtol=0, atol=1e-12)
