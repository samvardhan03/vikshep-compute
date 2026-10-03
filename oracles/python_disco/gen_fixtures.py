#!/usr/bin/env python3
"""Parity fixtures: the public Vikshep Python metric versus vikshep-stats.

Runs the reference implementation of the public repository
(https://github.com/samvardhan03/Vikshep):

* backend/ingest/src/vikshep_ingest/disco.py       weighted_dcorr2
* backend/ingest/src/vikshep_ingest/cli/_train.py  _pearson_dcorr2_grad, train_calibrate

on the inputs of backend/ingest/tests/test_disco.py (same numpy seeds and
constructions) plus a proxy-gradient and a calibration case, and writes the
inputs and outputs as exact binary64 bit patterns to
oracles/python_disco/fixtures.json. Developer-run; CI reads the fixture.

    git clone https://github.com/samvardhan03/Vikshep /path/to/Vikshep
    PYTHONPATH=/path/to/Vikshep/backend/ingest/src python3 oracles/python_disco/gen_fixtures.py
"""

import json
import os
import subprocess
import sys

import numpy as np

from vikshep_ingest.disco import weighted_dcorr2
from vikshep_ingest.cli._train import _pearson_dcorr2_grad, train_calibrate

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures.json")


def hx(a):
    return np.asarray(a, dtype="<f8").tobytes().hex()


def rng(seed):
    return np.random.default_rng(seed)


def case(name, x, y, w=None, assertion=None):
    value = weighted_dcorr2(x, y, w)
    return {
        "name": name,
        "kind": "dcorr2",
        "x": hx(x),
        "y": hx(y),
        "w": None if w is None else hx(w),
        "expected": hx([value]),
        "expected_decimal": repr(value),
        "assertion": assertion,
    }


def main():
    src_dir = os.path.dirname(sys.modules["vikshep_ingest.disco"].__file__)
    try:
        commit = subprocess.check_output(
            ["git", "-C", src_dir, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    cases = []

    r = rng(0)
    x = r.normal(0, 1, 500); y = r.normal(0, 1, 500)
    cases.append(case("independent_variables_approx_zero", x, y, assertion="< 0.1"))

    r = rng(1)
    x = r.uniform(0, 1, 1000); y = r.uniform(0, 1, 1000)
    cases.append(case("independent_uniform", x, y, assertion="< 0.1"))

    r = rng(2)
    x = r.normal(0, 1, 500); y = x ** 2
    cases.append(case("nonlinear_dependence_detected", x, y, assertion="> 0.2"))

    x = np.linspace(0, 1, 200); y = 2.0 * x + 0.5
    cases.append(case("perfect_linear_gives_one", x, y, assertion="> 0.95"))

    r = rng(3)
    n = 400
    x_ind = r.normal(0, 1, n); y_ind = r.normal(0, 1, n)
    cases.append(case("skewed_uniform_weights_independent", x_ind, y_ind, np.ones(n)))
    x_mixed = x_ind.copy(); y_mixed = y_ind.copy()
    x_mixed[:20] = np.linspace(-2, 2, 20); y_mixed[:20] = x_mixed[:20]
    w_skewed = np.ones(n) * 0.01; w_skewed[:20] = 5.0
    cases.append(case("skewed_weighted", x_mixed, y_mixed, w_skewed,
                      assertion="> skewed_unweighted + 0.05"))
    cases.append(case("skewed_unweighted", x_mixed, y_mixed, np.ones(n)))

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cases.append(case("identical_arrays_returns_one", x, x.copy(), assertion="> 0.99"))

    x = np.array([1.0])
    cases.append(case("single_element_returns_zero", x, x.copy(), assertion="== 0"))

    r = rng(5)
    x = r.normal(0, 1, 100); y = x + r.normal(0, 0.3, 100)
    cases.append(case("weight_normalization_w1", x, y, np.ones(100)))
    cases.append(case("weight_normalization_w7", x, y, np.ones(100) * 7.0,
                      assertion="abs(- weight_normalization_w1) < 1e-8"))

    # Pearson proxy gradient (fast mode of the public CLI).
    r = rng(11)
    s = 1.0 / (1.0 + np.exp(-r.normal(0, 1, 300)))
    m = 50.0 + 100.0 * r.uniform(0, 1, 300)
    w = 0.1 + r.uniform(0, 1, 300)
    g = _pearson_dcorr2_grad(s, m, w)
    cases.append({"name": "pearson_proxy_grad", "kind": "pearson_proxy_grad",
                  "x": hx(s), "y": hx(m), "w": hx(w), "expected": hx(g)})

    # Calibration (ridge, closed form).
    r = rng(12)
    X = r.normal(0, 1, (400, 5)) * np.array([1.0, 2.0, 0.5, 3.0, 1.5]) + 1.0
    t = X @ np.array([0.5, -1.0, 2.0, 0.0, 0.25]) + 3.0 + r.normal(0, 0.1, 400)
    res = train_calibrate(X, t)
    cases.append({"name": "train_calibrate", "kind": "calibrate", "rows": 400, "cols": 5,
                  "x": hx(X.ravel()), "y": hx(t),
                  "expected_r2": hx([res["r2_score"]]),
                  "expected_residual_std": hx([res["residual_std"]]),
                  "expected_coef": hx(res["w"]),
                  "expected_bias": hx([res["bias"]]),
                  "expected_mu": hx(res["mu"].ravel()),
                  "expected_std": hx(res["std"].ravel())})

    doc = {
        "provenance": {
            "source": "https://github.com/samvardhan03/Vikshep",
            "commit": commit,
            "files": ["backend/ingest/src/vikshep_ingest/disco.py",
                      "backend/ingest/src/vikshep_ingest/cli/_train.py",
                      "backend/ingest/tests/test_disco.py"],
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "cases": cases,
    }
    with open(OUT, "w", newline="\n") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {len(cases)} cases to {OUT}")


if __name__ == "__main__":
    main()
