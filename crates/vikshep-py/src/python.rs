//! The `vikshep_compute._core` extension module (PyO3).
//!
//! Every array argument is converted with numpy (`numpy.asarray(obj,
//! dtype=...)`: float32 for signals and coefficients, float64 for
//! statistics) and then copied element by element in row-major (C) order
//! into a Rust-owned contiguous buffer before any computation. Strides,
//! memory order (C or Fortran), byte order and alignment of the caller's
//! array therefore never affect a result; canonical bytes and OIDs are
//! little-endian (VDS-1 section 9.1).

use numpy::{
    AllowTypeChange, PyArray1, PyArrayDyn, PyArrayLikeDyn, PyArrayMethods, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use vikshep_anomaly::graph::{AnomalyConfig, AnomalyIndex};
use vikshep_anomaly::hnsw::HnswConfig;
use vikshep_anomaly::sw1::{Cloud, fingerprint_clouds as clouds_of};
use vikshep_cpu::CpuBackend;
use vikshep_numerics::oid::oid as oid_of;
use vikshep_numerics::provenance::{Execution, Executor};
use vikshep_numerics::{NUMERICS_VERSION, TIER2_VERSION};
use vikshep_scatter::config::DEFAULT_CARRIER_CUTOFF;
use vikshep_scatter::manifest::scattering_manifest;
use vikshep_scatter::reduce::{log_mean, log_mean_bytes, r2 as r2_of};
use vikshep_scatter::{Group, PadPolicy, ScatterConfig, ScatterOutput, Scattering};
use vikshep_stats::report::EvalConfig;
use vikshep_train::calibrate::ridge;
use vikshep_train::data::{Dataset, synthetic};
use vikshep_train::model::HeadKind;
use vikshep_train::sweep::benchmark;
use vikshep_train::train::{GradientMode, TrainConfig, TrainedModel, train};

type F32In<'py> = PyArrayLikeDyn<'py, f32, AllowTypeChange>;
type F64In<'py> = PyArrayLikeDyn<'py, f64, AllowTypeChange>;

fn value_err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Rust-owned row-major copy of an array and its shape.
fn owned_f32(a: &F32In<'_>) -> (Vec<f32>, Vec<usize>) {
    (a.as_array().iter().copied().collect(), a.shape().to_vec())
}

fn owned_f64(a: &F64In<'_>) -> (Vec<f64>, Vec<usize>) {
    (a.as_array().iter().copied().collect(), a.shape().to_vec())
}

fn f32_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn array_f32<'py>(
    py: Python<'py>,
    v: Vec<f32>,
    shape: Vec<usize>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    PyArray1::from_vec(py, v).reshape(shape)
}

fn array_f64<'py>(
    py: Python<'py>,
    v: Vec<f64>,
    shape: Vec<usize>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    PyArray1::from_vec(py, v).reshape(shape)
}

const CONFIG_KEYS: &[&str] = &[
    "J",
    "L",
    "Q",
    "carrier_cutoff",
    "dim",
    "group",
    "max_order",
    "pad",
    "shape",
];

fn pad_of(s: &str) -> PyResult<PadPolicy> {
    match s {
        "circular" => Ok(PadPolicy::Circular),
        "zero_pad" => Ok(PadPolicy::ZeroPad),
        other => Err(value_err(format!(
            "unknown pad {other:?} (circular or zero_pad)"
        ))),
    }
}

/// A scattering configuration from a dict with the keys of the manifest
/// configuration (`J`, `L`, `Q`, `carrier_cutoff`, `dim`, `group`,
/// `max_order`, `pad`, `shape`). `shape` may be omitted when `data_shape`
/// is given: it is then the trailing `dim` axes of the data.
fn scatter_config(
    cfg: &Bound<'_, PyDict>,
    data_shape: Option<&[usize]>,
) -> PyResult<ScatterConfig> {
    for k in cfg.keys() {
        let k: String = k.extract()?;
        if !CONFIG_KEYS.contains(&k.as_str()) {
            return Err(value_err(format!("unknown configuration key {k:?}")));
        }
    }
    let get = |k: &str| cfg.get_item(k);
    let shape: Option<Vec<usize>> = get("shape")?.map(|v| v.extract()).transpose()?;
    let dim: usize = match get("dim")? {
        Some(v) => v.extract()?,
        None => shape.as_ref().map_or(1, Vec::len),
    };
    let shape = match shape {
        Some(s) => s,
        None => {
            let ds = data_shape.ok_or_else(|| value_err("configuration needs 'shape'"))?;
            if ds.len() < dim {
                return Err(value_err("data has fewer axes than dim"));
            }
            ds[ds.len() - dim..].to_vec()
        }
    };
    let j: u32 = get("J")?
        .ok_or_else(|| value_err("configuration needs 'J'"))?
        .extract()?;
    let q: u32 = get("Q")?.map(|v| v.extract()).transpose()?.unwrap_or(1);
    let l: u32 = get("L")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(if dim == 2 { 8 } else { 1 });
    let max_order: u32 = get("max_order")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(2);
    let carrier_cutoff: u32 = get("carrier_cutoff")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or(DEFAULT_CARRIER_CUTOFF);
    let group = match get("group")?
        .map(|v| v.extract::<String>())
        .transpose()?
        .as_deref()
    {
        None | Some("trivial") => Group::Trivial,
        Some("so2_relative" | "so2") => Group::So2Relative,
        Some(other) => return Err(value_err(format!("unknown group {other:?}"))),
    };
    let pad = match get("pad")? {
        None => vec![PadPolicy::ZeroPad; dim],
        Some(v) => {
            if let Ok(s) = v.extract::<String>() {
                vec![pad_of(&s)?; dim]
            } else {
                let names: Vec<String> = v.extract()?;
                names.iter().map(|s| pad_of(s)).collect::<PyResult<_>>()?
            }
        }
    };
    let c = ScatterConfig {
        dim,
        group,
        j,
        q,
        l,
        max_order,
        pad,
        shape,
        carrier_cutoff,
    };
    c.validate().map_err(|e| value_err(e.0))?;
    Ok(c)
}

fn config_dict<'py>(py: Python<'py>, c: &ScatterConfig) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("J", c.j)?;
    d.set_item("L", c.l)?;
    d.set_item("Q", c.q)?;
    d.set_item("carrier_cutoff", c.carrier_cutoff)?;
    d.set_item("dim", c.dim)?;
    d.set_item("group", c.group.name())?;
    d.set_item("max_order", c.max_order)?;
    d.set_item("pad", c.pad.iter().map(|p| p.name()).collect::<Vec<_>>())?;
    d.set_item("shape", c.shape.clone())?;
    Ok(d)
}

fn prepared(cfg: ScatterConfig) -> PyResult<Scattering> {
    Scattering::new(cfg).map_err(|e| value_err(e.0))
}

/// Coefficients `[batch][paths][out]` of `sc` from any array of the right
/// size.
fn coefficients(sc: &Scattering, coeffs: &F32In<'_>) -> PyResult<ScatterOutput> {
    let (v, _) = owned_f32(coeffs);
    let per = sc.paths().len() * sc.config().out_len();
    if v.is_empty() || !v.len().is_multiple_of(per) {
        return Err(value_err(
            "coefficient count is not a positive multiple of paths * out_len",
        ));
    }
    Ok(ScatterOutput {
        paths: sc.paths().to_vec(),
        out_shape: sc.config().out_shape(),
        batch: v.len() / per,
        coefficients: v,
    })
}

/// Scattering coefficients of a batch of signals.
///
/// ``batch`` holds one or more signals whose trailing axes are the signal
/// shape; ``config`` is a dict (``J`` required; ``Q`` = 1, ``L`` = 1 in 1-D
/// or 8 in 2-D, ``max_order`` = 2, ``group`` = ``"trivial"``, ``pad`` =
/// ``"zero_pad"``, ``carrier_cutoff`` = 1 by default; ``shape`` defaults to
/// the trailing axes of ``batch``). Returns ``(coeffs, oid)``: float32
/// ``(B, paths, *out_shape)`` and the OID of their little-endian bytes.
#[pyfunction]
fn scatter<'py>(
    py: Python<'py>,
    batch: F32In<'py>,
    config: &Bound<'py, PyDict>,
) -> PyResult<(Bound<'py, PyArrayDyn<f32>>, String)> {
    let (x, shape) = owned_f32(&batch);
    let sc = prepared(scatter_config(config, Some(&shape))?)?;
    let sig = sc.config().signal_len();
    if x.is_empty() || !x.len().is_multiple_of(sig) {
        return Err(value_err(
            "batch size is not a positive multiple of the signal length",
        ));
    }
    let out = py
        .detach(|| sc.run(&CpuBackend::new(), &x))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let oid = oid_of(&out.canonical_bytes());
    let mut dims = vec![out.batch, out.paths.len()];
    dims.extend(&out.out_shape);
    Ok((array_f32(py, out.coefficients, dims)?, oid))
}

/// Output paths of a configuration: a list of dicts ``{order, j, theta}``
/// in canonical order (VDS-1 section 14.7).
#[pyfunction]
fn scatter_paths<'py>(
    py: Python<'py>,
    config: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyList>> {
    let sc = prepared(scatter_config(config, None)?)?;
    let list = PyList::empty(py);
    for p in sc.paths() {
        let d = PyDict::new(py);
        d.set_item("order", p.order)?;
        d.set_item("j", p.j.clone())?;
        d.set_item("theta", p.theta.clone())?;
        list.append(d)?;
    }
    Ok(list)
}

/// r2 ratios (VDS-1 section 14.8) of coefficients computed with ``config``
/// (which must include ``shape``). Returns ``(ratios, oid)``: float32
/// ``(B, pairs, *out_shape)``.
#[pyfunction]
fn r2<'py>(
    py: Python<'py>,
    coeffs: F32In<'py>,
    config: &Bound<'py, PyDict>,
) -> PyResult<(Bound<'py, PyArrayDyn<f32>>, String)> {
    let sc = prepared(scatter_config(config, None)?)?;
    let s = coefficients(&sc, &coeffs)?;
    let rr = r2_of(&s, sc.config().carrier_cutoff);
    let oid = oid_of(&rr.canonical_bytes());
    let mut dims = vec![rr.batch, rr.pairs.len()];
    dims.extend(sc.config().out_shape());
    Ok((array_f32(py, rr.values, dims)?, oid))
}

/// Log-mean fingerprint (VDS-1 section 14.9): float64 ``(B, paths)`` and
/// its OID.
#[pyfunction]
fn fingerprint<'py>(
    py: Python<'py>,
    coeffs: F32In<'py>,
    config: &Bound<'py, PyDict>,
) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, String)> {
    let sc = prepared(scatter_config(config, None)?)?;
    let s = coefficients(&sc, &coeffs)?;
    let fp = log_mean(&s);
    let oid = oid_of(&log_mean_bytes(&fp));
    Ok((array_f64(py, fp, vec![s.batch, s.paths.len()])?, oid))
}

/// Fingerprint distributions (VDS-1 section 19.1): float64
/// ``(B, positions, paths)``, one point per output position.
#[pyfunction]
fn fingerprint_clouds<'py>(
    py: Python<'py>,
    coeffs: F32In<'py>,
    config: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let sc = prepared(scatter_config(config, None)?)?;
    let s = coefficients(&sc, &coeffs)?;
    let cl = clouds_of(&s).map_err(|e| value_err(e.0))?;
    let (b, pos, p) = (s.batch, s.out_len(), s.paths.len());
    let v: Vec<f64> = cl.into_iter().flat_map(|c| c.points).collect();
    array_f64(py, v, vec![b, pos, p])
}

/// Weighted squared distance correlation of ``x`` and ``y`` (VDS-1 section
/// 16.1): exact up to 10,000 events, the chunked estimator above. ``w``
/// defaults to equal weights.
#[pyfunction]
#[pyo3(signature = (x, y, w = None))]
fn dcorr2_w(py: Python<'_>, x: F64In<'_>, y: F64In<'_>, w: Option<F64In<'_>>) -> PyResult<f64> {
    let (x, _) = owned_f64(&x);
    let (y, _) = owned_f64(&y);
    let w = w.map(|w| owned_f64(&w).0);
    py.detach(|| vikshep_stats::dcorr::weighted_dcorr2(&x, &y, w.as_deref()))
        .map_err(|e| value_err(e.0))
}

fn head_of(s: &str) -> PyResult<HeadKind> {
    if s == "logistic" {
        return Ok(HeadKind::Logistic);
    }
    s.strip_prefix("mlp")
        .and_then(|h| h.parse().ok())
        .filter(|&h| h > 0)
        .map(|hidden| HeadKind::Mlp { hidden })
        .ok_or_else(|| {
            value_err(format!(
                "head must be 'logistic' or 'mlp<width>', got {s:?}"
            ))
        })
}

fn gradient_of(s: &str) -> PyResult<GradientMode> {
    match s {
        "exact" => Ok(GradientMode::Exact),
        "pearson_proxy" => Ok(GradientMode::PearsonProxy),
        other => Err(value_err(format!(
            "gradient must be 'exact' or 'pearson_proxy', got {other:?}"
        ))),
    }
}

/// A dataset from arrays: ``x`` (n, d), ``y`` labels 0/1, optional weights
/// ``w`` (default 1) and protected variable ``m`` (default 0).
fn dataset(
    x: &F64In<'_>,
    y: &F64In<'_>,
    w: Option<&F64In<'_>>,
    m: Option<&F64In<'_>>,
) -> PyResult<Dataset> {
    let (xv, xs) = owned_f64(x);
    let (yv, _) = owned_f64(y);
    let n = yv.len();
    let d = match xs.len() {
        2 => xs[1],
        1 if n > 0 => xv.len() / n,
        _ => return Err(value_err("x must be (n, d)")),
    };
    let labels: Vec<u8> = yv
        .iter()
        .map(|&v| match v {
            0.0 => Ok(0u8),
            1.0 => Ok(1u8),
            _ => Err(value_err("labels must be 0 or 1")),
        })
        .collect::<PyResult<_>>()?;
    let wv = w.map_or_else(|| vec![1.0; n], |w| owned_f64(w).0);
    let mv = m.map_or_else(|| vec![0.0; n], |m| owned_f64(m).0);
    Dataset::new(d, xv, labels, wv, mv).map_err(|e| value_err(e.0))
}

fn dataset_from_dict(d: &Bound<'_, PyDict>) -> PyResult<Dataset> {
    let item = |k: &str| -> PyResult<Option<F64In<'_>>> {
        d.get_item(k)?.map(|v| v.extract()).transpose()
    };
    let x = item("x")?.ok_or_else(|| value_err("dataset needs 'x'"))?;
    let y = item("y")?.ok_or_else(|| value_err("dataset needs 'y'"))?;
    let w = item("w")?;
    let m = item("m")?;
    dataset(&x, &y, w.as_ref(), m.as_ref())
}

fn dataset_dict<'py>(py: Python<'py>, d: &Dataset) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("x", array_f64(py, d.x.clone(), vec![d.n, d.d])?)?;
    out.set_item("y", PyArray1::from_vec(py, d.y.clone()))?;
    out.set_item("w", PyArray1::from_vec(py, d.w.clone()))?;
    out.set_item("m", PyArray1::from_vec(py, d.m.clone()))?;
    out.set_item("oid", d.oid())?;
    Ok(out)
}

/// The synthetic benchmark sample of VDS-1 section 17.8: a dict with ``x``
/// (n, d), ``y`` (uint8), ``w``, ``m`` and the dataset ``oid``.
#[pyfunction]
#[pyo3(signature = (n, d, seed, split = 0))]
fn synthetic_dataset<'py>(
    py: Python<'py>,
    n: usize,
    d: usize,
    seed: u64,
    split: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let data = synthetic(n, d, seed, split).map_err(|e| value_err(e.0))?;
    dataset_dict(py, &data)
}

/// A trained tagger head (VDS-1 section 17).
#[pyclass(module = "vikshep_compute", name = "TaggerModel", frozen)]
struct TaggerModel {
    inner: TrainedModel,
}

#[pymethods]
impl TaggerModel {
    /// Flattened parameters (VDS-1 section 17.2).
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.head.params.clone())
    }

    /// Feature means of the standardization.
    #[getter]
    fn mu<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.standardizer.mu.clone())
    }

    /// Feature standard deviations (with the 1e-8 floor).
    #[getter]
    fn std<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.standardizer.std.clone())
    }

    /// Mean minibatch loss per epoch.
    #[getter]
    fn epoch_loss<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.epoch_loss.clone())
    }

    /// Head name (``logistic`` or ``mlp<width>``).
    #[getter]
    fn head(&self) -> String {
        self.inner.config.head.name()
    }

    /// OID of the canonical model bytes.
    #[getter]
    fn oid(&self) -> String {
        self.inner.oid()
    }

    /// Canonical model bytes: ``mu``, ``std``, parameters, epoch losses
    /// (binary64 little-endian).
    fn canonical_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.canonical_bytes())
    }

    /// Scores ``sigmoid(z)`` for raw feature rows ``x`` (n, d).
    fn predict<'py>(&self, py: Python<'py>, x: F64In<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (v, _) = owned_f64(&x);
        if !v.len().is_multiple_of(self.inner.head.d) {
            return Err(value_err("x does not have d columns"));
        }
        Ok(PyArray1::from_vec(py, self.inner.predict(&v)))
    }

    fn __repr__(&self) -> String {
        format!(
            "TaggerModel(head={}, oid={})",
            self.inner.config.head.name(),
            self.inner.oid()
        )
    }
}

/// Train a tagger head with the DisCo loss ``wBCE + lam * dCorr2_w(score,
/// m | background)`` (VDS-1 section 17). ``head``: ``"logistic"`` or
/// ``"mlp<width>"``; ``gradient``: ``"exact"`` or the fast mode
/// ``"pearson_proxy"``. ``m`` is required when ``lam > 0``.
#[pyfunction]
#[pyo3(signature = (x, y, w = None, m = None, *, head = "logistic", epochs = 20, batch_size = 256, lr = 0.01, lam = 0.0, gradient = "exact", seed = 0))]
#[allow(clippy::too_many_arguments)]
fn train_tag(
    py: Python<'_>,
    x: F64In<'_>,
    y: F64In<'_>,
    w: Option<F64In<'_>>,
    m: Option<F64In<'_>>,
    head: &str,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    lam: f64,
    gradient: &str,
    seed: u64,
) -> PyResult<TaggerModel> {
    if lam != 0.0 && m.is_none() {
        return Err(value_err(
            "the protected variable m is required when lam > 0",
        ));
    }
    let data = dataset(&x, &y, w.as_ref(), m.as_ref())?;
    let cfg = TrainConfig {
        head: head_of(head)?,
        epochs,
        batch_size,
        lr,
        lambda: lam,
        gradient: gradient_of(gradient)?,
        seed,
    };
    let inner = py
        .detach(|| train(&data, cfg))
        .map_err(|e| value_err(e.0))?;
    Ok(TaggerModel { inner })
}

/// Ridge calibration (VDS-1 section 18) of targets ``t`` on features ``x``
/// (n, d). Returns a dict: ``coef``, ``bias``, ``mu``, ``std``, ``r2``,
/// ``residual_std``, ``residuals`` (float64), ``residuals_oid`` and the
/// canonical ``constants_json``.
#[pyfunction]
#[pyo3(signature = (x, t, ridge_strength = vikshep_train::calibrate::DEFAULT_RIDGE))]
fn calibrate<'py>(
    py: Python<'py>,
    x: F64In<'py>,
    t: F64In<'py>,
    ridge_strength: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let (xv, xs) = owned_f64(&x);
    let (tv, _) = owned_f64(&t);
    let n = tv.len();
    let d = if xs.len() == 2 {
        xs[1]
    } else {
        return Err(value_err("x must be (n, d)"));
    };
    let c = py
        .detach(|| ridge(&xv, n, d, &tv, ridge_strength))
        .map_err(|e| value_err(e.0))?;
    let out = PyDict::new(py);
    out.set_item("coef", PyArray1::from_vec(py, c.coef.clone()))?;
    out.set_item("bias", c.bias)?;
    out.set_item("mu", PyArray1::from_vec(py, c.standardizer.mu.clone()))?;
    out.set_item("std", PyArray1::from_vec(py, c.standardizer.std.clone()))?;
    out.set_item("r2", c.r2)?;
    out.set_item("residual_std", c.residual_std)?;
    out.set_item("residuals_oid", oid_of(&c.residual_bytes()))?;
    out.set_item(
        "constants_json",
        c.constants().canonical().map_err(|e| value_err(e.0))?,
    )?;
    out.set_item("residuals", PyArray1::from_vec(py, c.residuals))?;
    Ok(out)
}

/// Point clouds from a 2-D array (one-point clouds, e.g. log-mean
/// fingerprints), a 3-D array (n, points, dim) or a list of 2-D arrays.
fn clouds(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Cloud>> {
    let mk = |dim: usize, pts: Vec<f64>| Cloud::new(dim, pts).map_err(|e| value_err(e.0));
    if let Ok(list) = obj.cast::<PyList>() {
        return list
            .iter()
            .map(|item| {
                let a: F64In<'_> = item.extract()?;
                let (v, s) = owned_f64(&a);
                if s.len() != 2 {
                    return Err(value_err("each cloud must be (points, dim)"));
                }
                mk(s[1], v)
            })
            .collect();
    }
    let a: F64In<'_> = obj.extract()?;
    let (v, s) = owned_f64(&a);
    match s.len() {
        2 => v
            .chunks(s[1].max(1))
            .map(|c| mk(s[1], c.to_vec()))
            .collect(),
        3 => v
            .chunks((s[1] * s[2]).max(1))
            .map(|c| mk(s[2], c.to_vec()))
            .collect(),
        _ => Err(value_err(
            "references must be (n, dim), (n, points, dim) or a list of (points, dim)",
        )),
    }
}

/// A deterministic anomaly index over fingerprint distributions (VDS-1
/// section 19).
#[pyclass(module = "vikshep_compute", name = "AnomalyIndex", frozen)]
struct PyAnomalyIndex {
    inner: AnomalyIndex,
}

#[pymethods]
impl PyAnomalyIndex {
    /// Number of reference events.
    fn __len__(&self) -> usize {
        self.inner.hnsw.len()
    }

    /// Canonical HNSW graph bytes (VDS-1 section 19.3).
    fn graph_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.hnsw.canonical_bytes())
    }

    /// SHA3-256 of the canonical configuration.
    #[getter]
    fn config_digest(&self) -> PyResult<String> {
        self.inner
            .config
            .value()
            .digest()
            .map_err(|e| value_err(e.0))
    }
}

/// Build an anomaly index (Sliced Wasserstein-1 over HNSW, VDS-1 section
/// 19) over reference clouds: a 2-D array (one-point clouds such as
/// log-mean fingerprints), a 3-D array (n, points, dim) such as
/// ``fingerprint_clouds(...)``, or a list of (points, dim) arrays.
#[pyfunction]
#[pyo3(signature = (references, *, tau, k = 10, directions = 32, m = 8, ef_construction = 64, ef_search = 32, layout_iterations = 100, seed = 0))]
#[allow(clippy::too_many_arguments)]
fn anomaly_index(
    py: Python<'_>,
    references: &Bound<'_, PyAny>,
    tau: f64,
    k: usize,
    directions: usize,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    layout_iterations: usize,
    seed: u64,
) -> PyResult<PyAnomalyIndex> {
    let refs = clouds(references)?;
    let cfg = AnomalyConfig {
        k,
        tau,
        directions,
        hnsw: HnswConfig {
            m,
            ef_construction,
            ef_search,
            seed,
        },
        layout_iterations,
    };
    let inner = py
        .detach(|| AnomalyIndex::build(&refs, cfg))
        .map_err(|e| value_err(e.0))?;
    Ok(PyAnomalyIndex { inner })
}

/// Query an anomaly index. Returns a dict: ``kth_distance`` (float64, NaN
/// with fewer than k references), ``nearest`` (int64 reference index, -1 if
/// none), ``flagged`` (bool), and ``graph_json``, the output contract
/// (VDS-1 section 19.6) of the references plus these queries.
#[pyfunction]
fn anomaly_query<'py>(
    py: Python<'py>,
    index: &PyAnomalyIndex,
    queries: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let q = clouds(queries)?;
    let idx = &index.inner;
    let (det, graph) = py
        .detach(|| -> Result<_, String> {
            let det = idx.detect(&q).map_err(|e| e.0)?;
            let graph = idx.graph(&q).and_then(|g| g.to_json()).map_err(|e| e.0)?;
            Ok((det, graph))
        })
        .map_err(value_err)?;
    let out = PyDict::new(py);
    out.set_item(
        "kth_distance",
        PyArray1::from_vec(
            py,
            det.iter()
                .map(|d| d.kth_distance.unwrap_or(f64::NAN))
                .collect(),
        ),
    )?;
    out.set_item(
        "nearest",
        PyArray1::from_vec(
            py,
            det.iter()
                .map(|d| d.nearest.map_or(-1, |i| i as i64))
                .collect(),
        ),
    )?;
    out.set_item(
        "flagged",
        PyArray1::from_vec(py, det.iter().map(|d| d.flagged).collect()),
    )?;
    out.set_item("graph_json", graph)?;
    Ok(out)
}

/// Lambda sweep and benchmark report (VDS-1 sections 16.7 and 17.7).
/// ``train`` and ``eval`` are dicts with ``x``, ``y``, ``w``, ``m`` (as
/// returned by ``synthetic_dataset``). Returns a dict with ``json``
/// (canonical report.json) and ``markdown`` (report.md); the win condition
/// is Delta-sigma > 0 AND Delta-JSD <= 0 against lambda = 0, and the
/// significance is an Asimov proxy, not a Wilks fit.
#[pyfunction]
#[pyo3(signature = (train, eval, *, lambdas, lambda_star, head = "logistic", epochs = 20, batch_size = 256, lr = 0.01, gradient = "exact", seed = 0, n_bins = 20, target_sig_eff = 0.5))]
#[allow(clippy::too_many_arguments)]
fn bench_report<'py>(
    py: Python<'py>,
    train: &Bound<'py, PyDict>,
    eval: &Bound<'py, PyDict>,
    lambdas: Vec<f64>,
    lambda_star: f64,
    head: &str,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    gradient: &str,
    seed: u64,
    n_bins: usize,
    target_sig_eff: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let tr = dataset_from_dict(train)?;
    let ev = dataset_from_dict(eval)?;
    let base = TrainConfig {
        head: head_of(head)?,
        epochs,
        batch_size,
        lr,
        lambda: 0.0,
        gradient: gradient_of(gradient)?,
        seed,
    };
    let ec = EvalConfig {
        target_sig_eff,
        n_bins,
    };
    let (_, report) = py
        .detach(|| benchmark(&tr, &ev, base, &lambdas, lambda_star, ec))
        .map_err(|e| value_err(e.0))?;
    let out = PyDict::new(py);
    out.set_item("json", report.json)?;
    out.set_item("markdown", report.markdown)?;
    Ok(out)
}

/// Provenance manifest (``spec/provenance.schema.json``) of a scattering
/// run of ``batch`` with ``config`` that produced ``coeffs_oid``. Returns
/// ``(json, manifest_hash)``; backend, executor and wall-clock times are
/// recorded but never change the hash.
#[pyfunction]
#[pyo3(signature = (batch, config, coeffs_oid, *, backend = "cpu", backend_version = None, executor = "local", started_unix_ms = None, finished_unix_ms = None))]
#[allow(clippy::too_many_arguments)]
fn provenance_manifest(
    batch: F32In<'_>,
    config: &Bound<'_, PyDict>,
    coeffs_oid: &str,
    backend: &str,
    backend_version: Option<&str>,
    executor: &str,
    started_unix_ms: Option<u64>,
    finished_unix_ms: Option<u64>,
) -> PyResult<(String, String)> {
    let (x, shape) = owned_f32(&batch);
    let sc = prepared(scatter_config(config, Some(&shape))?)?;
    let sig = sc.config().signal_len();
    if x.is_empty() || !x.len().is_multiple_of(sig) {
        return Err(value_err(
            "batch size is not a positive multiple of the signal length",
        ));
    }
    if coeffs_oid.len() != 28
        || !coeffs_oid
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(value_err("coeffs_oid must be 28 lowercase hex characters"));
    }
    let mut ex = Execution::local(
        backend,
        backend_version.unwrap_or(env!("CARGO_PKG_VERSION")),
    );
    ex.executor = Executor::parse(executor)
        .ok_or_else(|| value_err("executor must be 'local' or 'cloud'"))?;
    ex.started_unix_ms = started_unix_ms;
    ex.finished_unix_ms = finished_unix_ms;
    let m = scattering_manifest(&sc, x.len() / sig, &oid_of(&f32_le(&x)), coeffs_oid, ex);
    let json = m.to_json().map_err(|e| value_err(e.0))?;
    let hash = m.manifest_hash().map_err(|e| value_err(e.0))?;
    Ok((json, hash))
}

/// Run the conformance self-test against the vectors compiled into this
/// module: ``subset`` is ``"quick"`` or ``"full"``. Returns a dict with
/// ``cases``, ``passed``, ``failed`` and the JSON ``report``.
#[pyfunction]
#[pyo3(signature = (subset = "quick"))]
fn conformance_selftest<'py>(py: Python<'py>, subset: &str) -> PyResult<Bound<'py, PyDict>> {
    let report = py
        .detach(|| vikshep_conformance_core::suite::selftest(&CpuBackend::new(), subset))
        .map_err(value_err)?;
    let out = PyDict::new(py);
    out.set_item("cases", report.summary.cases)?;
    out.set_item("passed", report.summary.pass)?;
    out.set_item("failed", report.summary.fail)?;
    out.set_item("report", report.to_json())?;
    Ok(out)
}

/// OID of raw bytes: hex of the first 14 bytes of SHA3-256.
#[pyfunction]
fn oid(data: &[u8]) -> String {
    oid_of(data)
}

/// Testing helper: configuration (dict) and input signal (float32, shape
/// ``(1, *shape)``) of a scattering conformance case.
#[pyfunction]
fn _conformance_scatter_case<'py>(
    py: Python<'py>,
    case_id: &str,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyArrayDyn<f32>>)> {
    let (cfg, x) = vikshep_conformance_core::suite::scatter_case(case_id)
        .ok_or_else(|| value_err(format!("no scattering case {case_id:?}")))?;
    let mut dims = vec![1];
    dims.extend(&cfg.shape);
    Ok((config_dict(py, &cfg)?, array_f32(py, x, dims)?))
}

/// The extension module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("NUMERICS_VERSION", NUMERICS_VERSION)?;
    m.add("TIER2_VERSION", TIER2_VERSION)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("PLATFORM", vikshep_numerics::provenance::PLATFORM)?;
    m.add_class::<TaggerModel>()?;
    m.add_class::<PyAnomalyIndex>()?;
    m.add_function(wrap_pyfunction!(scatter, m)?)?;
    m.add_function(wrap_pyfunction!(scatter_paths, m)?)?;
    m.add_function(wrap_pyfunction!(r2, m)?)?;
    m.add_function(wrap_pyfunction!(fingerprint, m)?)?;
    m.add_function(wrap_pyfunction!(fingerprint_clouds, m)?)?;
    m.add_function(wrap_pyfunction!(dcorr2_w, m)?)?;
    m.add_function(wrap_pyfunction!(synthetic_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(train_tag, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate, m)?)?;
    m.add_function(wrap_pyfunction!(anomaly_index, m)?)?;
    m.add_function(wrap_pyfunction!(anomaly_query, m)?)?;
    m.add_function(wrap_pyfunction!(bench_report, m)?)?;
    m.add_function(wrap_pyfunction!(provenance_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(conformance_selftest, m)?)?;
    m.add_function(wrap_pyfunction!(oid, m)?)?;
    m.add_function(wrap_pyfunction!(_conformance_scatter_case, m)?)?;
    Ok(())
}
