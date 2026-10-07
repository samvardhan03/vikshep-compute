"""Python results are byte-equal to the committed conformance vectors
(conformance/vectors/v2, numerics_version 2) for a subset of cases: scattering (S, r2,
log_mean), training runs and benchmark reports."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

import vikshep_compute as vc

ROOT = Path(__file__).resolve().parents[3]
VECTORS = ROOT / "conformance" / "vectors" / f"v{vc.NUMERICS_VERSION}"


def _load():
    expected = json.loads((VECTORS / "expected.json").read_text())
    blob = (VECTORS / expected["bytes_file"]).read_bytes()
    cases = {c["id"]: {o["name"]: o for o in c["outputs"]} for c in expected["cases"]}
    return cases, blob


CASES, BLOB = _load()
MASTER_SEED = int(
    re.search(r"^master_seed\s*=\s*(\S+)", (ROOT / "conformance" / "cases.toml").read_text(), re.M).group(1), 0
)


def stream_id(case_id):
    """First 8 bytes of SHA3-256 of the case id, little-endian (VDS-1 11.2)."""
    return int.from_bytes(hashlib.sha3_256(case_id.encode()).digest()[:8], "little")


def assert_matches(case_id, name, data: bytes):
    out = CASES[case_id][name]
    assert len(data) == out["len_bytes"], (case_id, name)
    assert hashlib.sha3_256(data).hexdigest() == out["sha3_256"], (case_id, name)
    if out.get("offset") is not None:
        off = out["offset"]
        assert data == BLOB[off : off + out["len_bytes"]], (case_id, name)


SCATTER_CASES = [
    "scatter/1d/256/J2-Q1-L1/circular/trivial/o2/uniform",
    "scatter/1d/256/J4-Q1-L1/zero_pad/trivial/o2/uniform",
    "scatter/1d/1024/J6-Q8-L1/circular/trivial/o2/uniform",
    "scatter/2d/32x32/J1-Q1-L4/circular.circular/trivial/o2/uniform",
    "scatter/2d/32x32/J1-Q1-L4/circular.circular/so2_relative/o2/uniform",
    "scatter/2d/32x32/J1-Q1-L8/zero_pad.circular/so2_relative/o2/uniform",
    # VDS-1.1 flush-to-zero: subnormal and near-FLT_MIN inputs
    "scatter/1d/256/J4-Q2-L1/circular/trivial/o2/near_subnormal",
    "scatter/1d/256/J4-Q2-L1/zero_pad/trivial/o2/subnormal",
    "scatter/2d/32x32/J2-Q1-L4/circular.circular/so2_relative/o2/flt_min_band",
]


@pytest.mark.parametrize("case_id", SCATTER_CASES)
def test_scattering_matches_vectors(case_id):
    cfg, x = vc._conformance_scatter_case(case_id)
    coeffs, oid = vc.scatter(x, cfg)
    s_bytes = coeffs.astype("<f4").tobytes()
    assert_matches(case_id, "S", s_bytes)
    assert oid == CASES[case_id]["S"]["sha3_256"][:28] == vc.oid(s_bytes)
    ratios, r2_oid = vc.r2(coeffs, cfg)
    assert_matches(case_id, "r2", ratios.astype("<f4").tobytes())
    assert r2_oid == CASES[case_id]["r2"]["sha3_256"][:28]
    fp, fp_oid = vc.fingerprint(coeffs, cfg)
    assert_matches(case_id, "log_mean", fp.astype("<f8").tobytes())
    assert fp_oid == CASES[case_id]["log_mean"]["sha3_256"][:28]


def _head_gradient_lambda(case_id):
    _, _, head, gradient, lam, n = case_id.split("/")
    return head, gradient, float(lam[len("lambda") :]), int(n[1:])


@pytest.mark.parametrize("case_id", [c for c in CASES if c.startswith("tier2/train/")])
def test_training_matches_vectors(case_id):
    head, gradient, lam, n = _head_gradient_lambda(case_id)
    sid = stream_id(case_id)
    data = vc.synthetic_dataset(n, 4, MASTER_SEED, sid)
    model = vc.train_tag(
        data["x"], data["y"], data["w"], data["m"],
        head=head, epochs=4, batch_size=128, lr=0.02, lam=lam, gradient=gradient, seed=sid,
    )
    assert_matches(case_id, "model", model.canonical_bytes())
    assert model.oid == CASES[case_id]["model"]["sha3_256"][:28]
    assert_matches(case_id, "scores", model.predict(data["x"]).astype("<f8").tobytes())


@pytest.mark.parametrize("case_id", [c for c in CASES if c.startswith("tier2/report/")])
def test_benchmark_report_matches_vectors(case_id):
    _, _, head, lambdas, star, n = case_id.split("/")
    lambdas = [float(v) for v in lambdas[len("lambdas") :].split("_")]
    sid = stream_id(case_id)
    n = int(n[1:])
    train = vc.synthetic_dataset(n, 4, MASTER_SEED, sid)
    evaluation = vc.synthetic_dataset(n, 4, MASTER_SEED, (sid + 1) % 2**64)
    report = vc.bench_report(
        train, evaluation, lambdas=lambdas, lambda_star=float(star[len("star") :]),
        head=head, epochs=5, batch_size=128, lr=0.02, gradient="exact", seed=sid,
    )
    assert_matches(case_id, "report_json", report["json"].encode())
    assert_matches(case_id, "report_md", report["markdown"].encode())
    assert "Asimov proxy, not a Wilks fit" in report["markdown"]


def test_embedded_selftest():
    r = vc.conformance_selftest("quick")
    assert r["cases"] > 100 and r["failed"] == 0 and r["passed"] == r["cases"]
    assert json.loads(r["report"])["summary"]["fail"] == 0
    with pytest.raises(ValueError):
        vc.conformance_selftest("everything")


def test_versions():
    assert vc.NUMERICS_VERSION == 2 and vc.TIER2_VERSION == 1
    assert isinstance(vc.__version__, str) and vc.PLATFORM
