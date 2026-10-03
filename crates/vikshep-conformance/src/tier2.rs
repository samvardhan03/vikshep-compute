//! Tier-2 conformance cases (`spec/VDS-1.md` section 11.5): statistics,
//! gradient, JSD, training, calibration, SW1, HNSW and the benchmark report.
//! Inputs are drawn from the case's Philox stream as documented per kind.

use serde::Deserialize;
use vikshep_anomaly::graph::{AnomalyConfig, AnomalyIndex};
use vikshep_anomaly::hnsw::HnswConfig;
use vikshep_anomaly::sw1::{Cloud, Directions, sw1};
use vikshep_numerics::jcs::format_number;
use vikshep_numerics::rng::Stream;
use vikshep_stats::dcorr::{dcorr2_chunked, dcorr2_exact, dcorr2_grad, pearson_proxy_grad};
use vikshep_stats::divergence::{Binning, jsd};
use vikshep_stats::report::EvalConfig;
use vikshep_train::calibrate::{DEFAULT_RIDGE, ridge};
use vikshep_train::data::synthetic;
use vikshep_train::model::{HeadKind, sigmoid};
use vikshep_train::sweep::benchmark;
use vikshep_train::train::{GradientMode, TrainConfig, train};

use crate::suite::{Dtype, Output};

/// One Tier-2 case from `[[tier2]]` in `conformance/cases.toml`.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Tier2Spec {
    /// Case kind.
    pub kind: String,
    /// Events (or clouds).
    pub n: Option<usize>,
    /// Weighted events.
    pub weighted: Option<bool>,
    /// Histogram bins.
    pub bins: Option<usize>,
    /// Head (`logistic` or `mlp<width>`).
    pub head: Option<String>,
    /// Gradient mode (`exact` or `pearson_proxy`).
    pub gradient: Option<String>,
    /// DisCo strength.
    pub lambda: Option<f64>,
    /// Sweep.
    pub lambdas: Option<Vec<f64>>,
    /// Win-condition lambda.
    pub lambda_star: Option<f64>,
    /// Features / dimension.
    pub d: Option<usize>,
    /// Points per cloud.
    pub points: Option<usize>,
    /// SW1 directions.
    pub directions: Option<usize>,
    /// HNSW M.
    pub m: Option<usize>,
    /// Neighbour rank.
    pub k: Option<usize>,
    /// Threshold.
    pub tau: Option<f64>,
}

fn num(x: f64) -> String {
    format_number(x).expect("finite parameter")
}

fn need<T: Copy>(v: Option<T>, what: &str, kind: &str) -> T {
    v.unwrap_or_else(|| panic!("cases.toml: tier2 {kind} needs {what}"))
}

impl Tier2Spec {
    /// Normative case identifier.
    #[must_use]
    pub fn id(&self) -> String {
        let k = self.kind.as_str();
        let n = || need(self.n, "n", k);
        let w = || {
            if need(self.weighted, "weighted", k) {
                "weighted"
            } else {
                "unweighted"
            }
        };
        match k {
            "dcorr_exact" | "dcorr_chunked" | "dcorr_grad" | "pearson_proxy" => {
                format!("tier2/{k}/n{}/{}", n(), w())
            }
            "jsd" => format!("tier2/jsd/n{}/bins{}", n(), need(self.bins, "bins", k)),
            "train" => format!(
                "tier2/train/{}/{}/lambda{}/n{}",
                self.head.as_deref().expect("head"),
                self.gradient.as_deref().expect("gradient"),
                num(need(self.lambda, "lambda", k)),
                n()
            ),
            "calibration" => format!("tier2/calibration/n{}/d{}", n(), need(self.d, "d", k)),
            "sw1" => format!(
                "tier2/sw1/clouds{}/points{}/d{}/dirs{}",
                n(),
                need(self.points, "points", k),
                need(self.d, "d", k),
                need(self.directions, "directions", k)
            ),
            "hnsw" => format!(
                "tier2/hnsw/n{}/m{}/k{}/tau{}",
                n(),
                need(self.m, "m", k),
                need(self.k, "k", k),
                num(need(self.tau, "tau", k))
            ),
            "report" => {
                let ls: Vec<String> = self
                    .lambdas
                    .as_ref()
                    .expect("lambdas")
                    .iter()
                    .map(|&l| num(l))
                    .collect();
                format!(
                    "tier2/report/{}/lambdas{}/star{}/n{}",
                    self.head.as_deref().expect("head"),
                    ls.join("_"),
                    num(need(self.lambda_star, "lambda_star", k)),
                    n()
                )
            }
            other => panic!("cases.toml: unknown tier2 kind {other:?}"),
        }
    }
}

fn f64_bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn out(name: &'static str, dtype: Dtype, bytes: Vec<u8>) -> Output {
    Output {
        name,
        dtype,
        bytes,
        storable: true,
    }
}

fn head_of(s: &str) -> HeadKind {
    match s {
        "logistic" => HeadKind::Logistic,
        _ => HeadKind::Mlp {
            hidden: s
                .strip_prefix("mlp")
                .and_then(|h| h.parse().ok())
                .expect("head mlp<width>"),
        },
    }
}

fn gradient_of(s: &str) -> GradientMode {
    match s {
        "exact" => GradientMode::Exact,
        "pearson_proxy" => GradientMode::PearsonProxy,
        other => panic!("cases.toml: unknown gradient {other:?}"),
    }
}

/// `(x, y, w)` for the dCorr kinds: per event `x = z1`, `y = x*x + 0.5 z2`
/// (z standard normal), then `w = 0.1 + u` if weighted (else 1).
fn pair_sample(n: usize, weighted: bool, s: &mut Stream) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut x, mut y, mut w) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..n {
        let a = s.next_normal_f64();
        let e = s.next_normal_f64();
        x.push(a);
        y.push(a * a + 0.5 * e);
        w.push(if weighted {
            0.1 + s.next_f64_unit()
        } else {
            1.0
        });
    }
    (x, y, w)
}

/// Clouds for SW1 / HNSW: cloud `i` has `points x d` values
/// `z + offset_i`, with `offset_i = 0.1 i` (sw1) or `4` for every 50th cloud
/// and 0 otherwise (hnsw).
fn clouds(n: usize, points: usize, d: usize, s: &mut Stream, outliers: bool) -> Vec<Cloud> {
    (0..n)
        .map(|i| {
            let off = if outliers {
                if i % 50 == 49 { 4.0 } else { 0.0 }
            } else {
                0.1 * i as f64
            };
            Cloud::new(
                d,
                (0..points * d).map(|_| s.next_normal_f64() + off).collect(),
            )
            .expect("valid cloud")
        })
        .collect()
}

/// Run one Tier-2 case.
pub fn run(spec: &Tier2Spec, master_seed: u64, stream_id: u64) -> Result<Vec<Output>, String> {
    let mut s = Stream::new(master_seed, stream_id);
    let k = spec.kind.as_str();
    let e = |x: String| x;
    Ok(match k {
        "dcorr_exact" | "dcorr_chunked" => {
            let (x, y, w) = pair_sample(
                need(spec.n, "n", k),
                need(spec.weighted, "weighted", k),
                &mut s,
            );
            let v = if k == "dcorr_exact" {
                dcorr2_exact(&x, &y, Some(&w))
            } else {
                dcorr2_chunked(&x, &y, Some(&w), stream_id)
            }
            .map_err(|er| e(er.0))?;
            vec![out("value", Dtype::F64, f64_bytes(&[v]))]
        }
        "dcorr_grad" => {
            let (x, y, w) = pair_sample(
                need(spec.n, "n", k),
                need(spec.weighted, "weighted", k),
                &mut s,
            );
            let scores: Vec<f64> = x.iter().map(|&v| sigmoid(v)).collect();
            let (v, g) = dcorr2_grad(&scores, &y, Some(&w)).map_err(|er| er.0)?;
            vec![
                out("value", Dtype::F64, f64_bytes(&[v])),
                out("grad", Dtype::F64, f64_bytes(&g)),
            ]
        }
        "pearson_proxy" => {
            let (x, y, w) = pair_sample(
                need(spec.n, "n", k),
                need(spec.weighted, "weighted", k),
                &mut s,
            );
            let scores: Vec<f64> = x.iter().map(|&v| sigmoid(v)).collect();
            let g = pearson_proxy_grad(&scores, &y, &w).map_err(|er| er.0)?;
            vec![out("grad", Dtype::F64, f64_bytes(&g))]
        }
        "jsd" => {
            // background of the synthetic sample; post-cut: x0 > 0.5
            let data =
                synthetic(need(spec.n, "n", k), 4, master_seed, stream_id).map_err(|er| er.0)?;
            let bkg: Vec<usize> = (0..data.n).filter(|&i| data.y[i] == 0).collect();
            let m: Vec<f64> = bkg.iter().map(|&i| data.m[i]).collect();
            let w: Vec<f64> = bkg.iter().map(|&i| data.w[i]).collect();
            let pass: Vec<usize> = (0..bkg.len())
                .filter(|&t| data.row(bkg[t])[0] > 0.5)
                .collect();
            let binning = Binning::from_range(&m, need(spec.bins, "bins", k)).map_err(|er| er.0)?;
            let pre = binning.histogram(&m, &w).map_err(|er| er.0)?;
            let post_m: Vec<f64> = pass.iter().map(|&t| m[t]).collect();
            let post_w: Vec<f64> = pass.iter().map(|&t| w[t]).collect();
            let post = binning.histogram(&post_m, &post_w).map_err(|er| er.0)?;
            let d = jsd(&pre, &post).map_err(|er| er.0)?;
            vec![
                out("pre", Dtype::F64, f64_bytes(&pre)),
                out("post", Dtype::F64, f64_bytes(&post)),
                out("jsd", Dtype::F64, f64_bytes(&[d])),
            ]
        }
        "train" => {
            let data =
                synthetic(need(spec.n, "n", k), 4, master_seed, stream_id).map_err(|er| er.0)?;
            let cfg = TrainConfig {
                head: head_of(spec.head.as_deref().expect("head")),
                epochs: 4,
                batch_size: 128,
                lr: 0.02,
                lambda: need(spec.lambda, "lambda", k),
                gradient: gradient_of(spec.gradient.as_deref().expect("gradient")),
                seed: stream_id,
            };
            let model = train(&data, cfg).map_err(|er| er.0)?;
            vec![
                out("model", Dtype::F64, model.canonical_bytes()),
                out("scores", Dtype::F64, f64_bytes(&model.predict(&data.x))),
            ]
        }
        "calibration" => {
            // x_ij = (1 + j) z + j; t = sum_j c_j x_ij + 1 + 0.1 z, c_j = (j + 1) (-1)^j / d
            let (n, d) = (need(spec.n, "n", k), need(spec.d, "d", k));
            let coef: Vec<f64> = (0..d)
                .map(|j| (j as f64 + 1.0) * if j % 2 == 0 { 1.0 } else { -1.0 } / d as f64)
                .collect();
            let mut x = Vec::with_capacity(n * d);
            let mut t = Vec::with_capacity(n);
            for _ in 0..n {
                let row: Vec<f64> = (0..d)
                    .map(|j| (1.0 + j as f64) * s.next_normal_f64() + j as f64)
                    .collect();
                let target = row.iter().zip(&coef).fold(1.0, |acc, (a, b)| acc + a * b)
                    + 0.1 * s.next_normal_f64();
                x.extend(row);
                t.push(target);
            }
            let c = ridge(&x, n, d, &t, DEFAULT_RIDGE).map_err(|er| er.0)?;
            let mut constants = c.constants().canonical().map_err(|er| er.0)?;
            constants.push('\n');
            vec![
                out("constants", Dtype::Bytes, constants.into_bytes()),
                out("residuals", Dtype::F64, c.residual_bytes()),
            ]
        }
        "sw1" => {
            let (n, p, d) = (
                need(spec.n, "n", k),
                need(spec.points, "points", k),
                need(spec.d, "d", k),
            );
            let cl = clouds(n, p, d, &mut s, false);
            let dirs = Directions::new(d, need(spec.directions, "directions", k), stream_id)
                .map_err(|er| er.0)?;
            let proj: Vec<_> = cl
                .iter()
                .map(|c| dirs.project(c))
                .collect::<Result<_, _>>()
                .map_err(|er| er.0)?;
            let mut mat = Vec::with_capacity(n * n);
            for a in &proj {
                for b in &proj {
                    mat.push(sw1(a, b));
                }
            }
            vec![out("matrix", Dtype::F64, f64_bytes(&mat))]
        }
        "hnsw" => {
            let n = need(spec.n, "n", k);
            let cl = clouds(n, 8, 4, &mut s, true);
            let cfg = AnomalyConfig {
                k: need(spec.k, "k", k),
                tau: need(spec.tau, "tau", k),
                directions: 8,
                hnsw: HnswConfig {
                    m: need(spec.m, "m", k),
                    ef_construction: 32,
                    ef_search: 16,
                    seed: stream_id,
                },
                layout_iterations: 30,
            };
            let idx = AnomalyIndex::build(&cl, cfg).map_err(|er| er.0)?;
            let det = idx.detect_self();
            let kth: Vec<f64> = det
                .iter()
                .map(|d| d.kth_distance.unwrap_or(f64::NAN))
                .collect();
            let flags: Vec<u8> = det.iter().map(|d| u8::from(d.flagged)).collect();
            let graph = idx
                .graph(&[])
                .map_err(|er| er.0)?
                .to_json()
                .map_err(|er| er.0)?;
            vec![
                out("graph_bytes", Dtype::Bytes, idx.hnsw.canonical_bytes()),
                out("kth_distance", Dtype::F64, f64_bytes(&kth)),
                out("flags", Dtype::Bytes, flags),
                out("graph_json", Dtype::Bytes, graph.into_bytes()),
            ]
        }
        "report" => {
            let n = need(spec.n, "n", k);
            let tr = synthetic(n, 4, master_seed, stream_id).map_err(|er| er.0)?;
            let ev = synthetic(n, 4, master_seed, stream_id.wrapping_add(1)).map_err(|er| er.0)?;
            let base = TrainConfig {
                head: head_of(spec.head.as_deref().expect("head")),
                epochs: 5,
                batch_size: 128,
                lr: 0.02,
                lambda: 0.0,
                gradient: GradientMode::Exact,
                seed: stream_id,
            };
            let lambdas = spec.lambdas.clone().expect("lambdas");
            let (_, report) = benchmark(
                &tr,
                &ev,
                base,
                &lambdas,
                need(spec.lambda_star, "lambda_star", k),
                EvalConfig::default(),
            )
            .map_err(|er| er.0)?;
            vec![
                out("report_json", Dtype::Bytes, report.json.into_bytes()),
                out("report_md", Dtype::Bytes, report.markdown.into_bytes()),
            ]
        }
        other => return Err(format!("unknown tier2 kind {other:?}")),
    })
}
