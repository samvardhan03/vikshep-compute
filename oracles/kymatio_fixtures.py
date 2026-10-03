#!/usr/bin/env python3
"""Kymatio correctness-oracle fixtures for vikshep-scatter (VDS-1 section 12).

Kymatio (BSD-3-Clause) is used as an independent binary64 reference. This
script is run by a developer, never by CI; its output is committed under
oracles/fixtures/ and read by crates/vikshep-scatter/tests/kymatio_oracle.rs.

    pip install kymatio==0.3.0 numpy scipy
    python3 oracles/kymatio_fixtures.py

For each configuration it writes one JSON file containing the binary32 input
and up to two references, both in the canonical path order of VDS-1 section
14.7 and on the output grid of section 14.4:

* "fullres": the full-resolution cascade (no intermediate subsampling, final
  spatial subsampling by 2^J) with Kymatio's own filters. In 1-D this is
  Kymatio's own cascade code (kymatio.scattering1d.core.scattering1d with
  oversampling >= J, which disables every intermediate subsampling). In 2-D
  Kymatio's cascade always subsamples intermediate layers, so the cascade is
  written here in numpy (15 lines) around kymatio.scattering2d.filter_bank.
* "kymatio_core" (2-D only): Kymatio's own 2-D cascade
  (kymatio.scattering2d.core.scattering2d) with identity padding on the
  canvas. It subsamples U1 by 2^j1 and U2 by 2^j2 by periodizing spectra, so
  it differs from VDS-1 by aliasing; it is compared with a looser tolerance.

Alignment of conventions (documented in VDS-1 section 14.3):
* Kymatio's 2-D orientation index t has angle (L/2 - 1 - t) * pi / L; VDS-1
  index l has angle l * pi / L. Paths are re-indexed with
  l = (L/2 - 1 - t) mod L (an angle differing by pi gives the complex
  conjugate wavelet, so moduli of real signals are unchanged).
* Kymatio's 2-D filters divide by 2 * 3.1415 * sigma^2 / slant, VDS-1 by
  2 * pi * sigma^2 / slant. Each filter application therefore differs by
  c = pi / 3.1415, and order-m coefficients by c^(m + 1); references are
  divided by that factor.
* Boundary: Kymatio pads by reflection; here the canvas of VDS-1 (circular,
  or zero padding at offset 2^J) is built explicitly and Kymatio runs with
  no padding of its own.
"""

import json
import math
import os
import sys

import numpy as np
import scipy
import kymatio
from kymatio.scattering1d.filter_bank import scattering_filter_factory
from kymatio.scattering1d.core.scattering1d import scattering1d
from kymatio.scattering1d.backend.numpy_backend import backend as backend_1d
from kymatio.scattering2d.filter_bank import filter_bank as filter_bank_2d
from kymatio.scattering2d.core.scattering2d import scattering2d
from kymatio.scattering2d.backend.numpy_backend import backend as backend_2d

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
C_NORM = math.pi / 3.1415

# (dim, shape, J, Q, L, pads, group)
CONFIGS = [
    (1, [256], 2, 1, 1, ["circular"], "trivial"),
    (1, [256], 4, 1, 1, ["circular"], "trivial"),
    (1, [256], 4, 4, 1, ["circular"], "trivial"),
    (1, [256], 6, 8, 1, ["circular"], "trivial"),
    (1, [1024], 4, 8, 1, ["circular"], "trivial"),
    (1, [1024], 6, 1, 1, ["circular"], "trivial"),
    (1, [256], 3, 2, 1, ["zero_pad"], "trivial"),
    (1, [192], 4, 1, 1, ["zero_pad"], "trivial"),
    (2, [32, 32], 1, 1, 4, ["circular", "circular"], "trivial"),
    (2, [32, 32], 2, 1, 4, ["circular", "circular"], "trivial"),
    (2, [32, 32], 3, 1, 8, ["circular", "circular"], "trivial"),
    (2, [32, 32], 4, 1, 8, ["circular", "circular"], "trivial"),
    (2, [64, 64], 3, 1, 8, ["circular", "circular"], "trivial"),
    (2, [32, 64], 2, 1, 4, ["circular", "circular"], "trivial"),
    (2, [32, 32], 2, 1, 4, ["zero_pad", "circular"], "trivial"),
    (2, [32, 32], 3, 1, 4, ["zero_pad", "zero_pad"], "trivial"),
    (2, [32, 32], 3, 1, 8, ["circular", "circular"], "so2_relative"),
    (2, [64, 64], 2, 1, 4, ["zero_pad", "circular"], "so2_relative"),
    (2, [32, 32], 4, 1, 4, ["circular", "circular"], "so2_relative"),
    (2, [64, 64], 4, 1, 8, ["circular", "circular"], "trivial"),
]


def axis_geometry(n, pad, J):
    step = 2 ** J
    if pad == "circular":
        return n, 0
    return 1 << (n + 2 * step - 1).bit_length(), step


def f32_hex(a):
    return np.asarray(a, dtype="<f4").tobytes().hex()


def f64_hex(a):
    return np.asarray(a, dtype="<f8").tobytes().hex()


def run_1d(x, shape, J, Q, pad):
    n = shape[0]
    canvas, off = axis_geometry(n, pad, J)
    u = np.zeros((1, canvas))
    u[0, off:off + n] = x
    phi, psi1, psi2 = scattering_filter_factory(canvas, J, (Q, 1), 2 ** J)
    ind = [0] * 64
    end = [canvas] * 64
    out = scattering1d(u, backend_1d, psi1, psi2, phi, ind_start=ind, ind_end=end,
                       oversampling=64, max_order=2, average=True)
    step = 2 ** J
    rows, paths = [], []
    for o in out:
        coef = np.asarray(o["coef"])[0]
        assert coef.shape[-1] == canvas
        rows.append(coef[off::step][: n // step])
        paths.append([len(o["n"])] + [int(v) for v in o["j"]])
    return paths, np.stack(rows)


def kym_to_ours_theta(t, L):
    return (L // 2 - 1 - t) % L


def filters_2d(canvas_shape, J, L):
    f = filter_bank_2d(canvas_shape[0], canvas_shape[1], J, L)
    phi = f["phi"]["levels"][0]
    psi = [(p["j"], kym_to_ours_theta(p["theta"], L), p["levels"][0], p) for p in f["psi"]]
    return f, phi, psi


def run_2d_fullres(u, J, L):
    _, phi, psi = filters_2d(u.shape, J, L)
    step = 2 ** J
    X = np.fft.fft2(u)
    out = {(): np.real(np.fft.ifft2(X * phi))[::step, ::step] / C_NORM}
    U1 = {}
    for j1, l1, h, _ in psi:
        U1[(j1, l1)] = np.fft.fft2(np.abs(np.fft.ifft2(X * h)))
        out[(j1, l1)] = np.real(np.fft.ifft2(U1[(j1, l1)] * phi))[::step, ::step] / C_NORM ** 2
    for j1, l1, _, _ in psi:
        for j2, l2, h2, _ in psi:
            if j2 > j1:
                U2 = np.fft.fft2(np.abs(np.fft.ifft2(U1[(j1, l1)] * h2)))
                out[(j1, l1, j2, l2)] = np.real(np.fft.ifft2(U2 * phi))[::step, ::step] / C_NORM ** 3
    return out


def run_2d_kymatio_core(u, J, L):
    f, _, _ = filters_2d(u.shape, J, L)
    res = scattering2d(u[None, :, :], lambda v: v, lambda v: v, backend_2d, J, L,
                       f["phi"], f["psi"], 2, out_type="list")
    out = {}
    for r in res:
        coef = np.asarray(r["coef"])[0]
        if len(r["j"]) == 0:
            key, order = (), 0
        elif len(r["j"]) == 1:
            key, order = (r["j"][0], kym_to_ours_theta(r["theta"][0], L)), 1
        else:
            key = (r["j"][0], kym_to_ours_theta(r["theta"][0], L),
                   r["j"][1], kym_to_ours_theta(r["theta"][1], L))
            order = 2
        out[key] = coef / C_NORM ** (order + 1)
    return out


def canonical_2d(out, J, L, group, crop):
    (r0, nr), (c0, nc) = crop
    keys = [()]
    keys += [(j, l) for j in range(J) for l in range(L)]
    keys += [(j1, l1, j2, l2) for j1 in range(J) for l1 in range(L)
             for j2 in range(J) for l2 in range(L) if j2 > j1]
    crop_ = lambda a: a[r0:r0 + nr, c0:c0 + nc]
    if group == "trivial":
        paths = [[len(k) // 2] + list(k[0::2]) + list(k[1::2]) for k in keys]
        return paths, np.stack([crop_(out[k]).ravel() for k in keys])
    paths, rows = [[0]], [crop_(out[()]).ravel()]
    for j in range(J):
        paths.append([1, j])
        rows.append(np.mean([crop_(out[(j, l)]) for l in range(L)], axis=0).ravel())
    for j1 in range(J):
        for j2 in range(j1 + 1, J):
            for d in range(L):
                paths.append([2, j1, j2, d])
                rows.append(np.mean([crop_(out[(j1, l1, j2, (l1 + d) % L)])
                                     for l1 in range(L)], axis=0).ravel())
    return paths, np.stack(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(20261003)
    provenance = {
        "kymatio": kymatio.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": sys.version.split()[0],
        "script": "oracles/kymatio_fixtures.py",
    }
    for i, (dim, shape, J, Q, L, pads, group) in enumerate(CONFIGS):
        x = rng.uniform(-1.0, 1.0, size=int(np.prod(shape))).astype(np.float32)
        refs = {}
        if dim == 1:
            paths, vals = run_1d(x.astype(np.float64), shape, J, Q, pads[0])
            refs["fullres"] = {"paths": paths, "values_f64_le_hex": f64_hex(vals)}
        else:
            geo = [axis_geometry(n, p, J) for n, p in zip(shape, pads)]
            u = np.zeros((geo[0][0], geo[1][0]))
            u[geo[0][1]:geo[0][1] + shape[0], geo[1][1]:geo[1][1] + shape[1]] = \
                x.astype(np.float64).reshape(shape)
            step = 2 ** J
            crop = [(g[1] // step, n // step) for g, n in zip(geo, shape)]
            paths, vals = canonical_2d(run_2d_fullres(u, J, L), J, L, group, crop)
            refs["fullres"] = {"paths": paths, "values_f64_le_hex": f64_hex(vals)}
            paths_k, vals_k = canonical_2d(run_2d_kymatio_core(u, J, L), J, L, group, crop)
            assert paths_k == paths
            refs["kymatio_core"] = {"paths": paths_k, "values_f64_le_hex": f64_hex(vals_k)}
        doc = {
            "provenance": provenance,
            "config": {"dim": dim, "shape": shape, "J": J, "Q": Q, "L": L,
                       "pad": pads, "group": group, "max_order": 2},
            "input_f32_le_hex": f32_hex(x),
            "references": refs,
        }
        name = f"case{i:02d}.json"
        with open(os.path.join(OUT_DIR, name), "w", newline="\n") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print(name, dim, shape, J, Q, L, pads, group, "paths:", len(refs["fullres"]["paths"]))


if __name__ == "__main__":
    main()
