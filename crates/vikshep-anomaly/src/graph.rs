//! Anomaly detection and the graph output contract (`spec/VDS-1.md`
//! sections 19.4 to 19.6).
//!
//! The output never contains fingerprints or features: only node ids,
//! layout coordinates, flags, edge distances, the threshold, a digest of
//! the configuration and the versions.

use vikshep_numerics::jcs::{Value, object};
use vikshep_numerics::rng::Stream;
use vikshep_numerics::{NUMERICS_VERSION, TIER2_VERSION};

use crate::AnomalyError;
use crate::hnsw::{Hnsw, HnswConfig};
use crate::sw1::{Cloud, Directions, Projected, sw1};

/// Philox stream of the layout initialization.
pub const STREAM_LAYOUT: u64 = 0x4_0003;

/// Anomaly-search configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnomalyConfig {
    /// Neighbour rank used for flagging.
    pub k: usize,
    /// Flag when the distance to the `k`-th nearest reference exceeds this.
    pub tau: f64,
    /// Number of SW1 projection directions.
    pub directions: usize,
    /// Index parameters (its seed also seeds directions and layout).
    pub hnsw: HnswConfig,
    /// Spring-layout iterations.
    pub layout_iterations: usize,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            k: 5,
            tau: 1.0,
            directions: 32,
            hnsw: HnswConfig::default(),
            layout_iterations: 100,
        }
    }
}

impl AnomalyConfig {
    /// The configuration as canonical JSON (hashed into `config_digest`).
    #[must_use]
    pub fn value(&self) -> Value {
        object([
            ("directions", Value::Int(self.directions as i64)),
            ("distance", Value::Str("sliced_wasserstein_1".into())),
            (
                "ef_construction",
                Value::Int(self.hnsw.ef_construction as i64),
            ),
            ("ef_search", Value::Int(self.hnsw.ef_search as i64)),
            (
                "fingerprint",
                Value::Str("log_coefficients_per_position".into()),
            ),
            ("k", Value::Int(self.k as i64)),
            (
                "layout_iterations",
                Value::Int(self.layout_iterations as i64),
            ),
            ("m", Value::Int(self.hnsw.m as i64)),
            ("seed", Value::Str(self.hnsw.seed.to_string())),
            ("tau", Value::Num(self.tau)),
        ])
    }
}

/// Per-event detection result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    /// Distance to the `k`-th nearest reference (`None` if fewer than `k`).
    pub kth_distance: Option<f64>,
    /// Nearest reference id.
    pub nearest: Option<usize>,
    /// `kth_distance > tau`.
    pub flagged: bool,
}

fn detection(nn: &[(usize, f64)], k: usize, tau: f64) -> Detection {
    let kth = nn.get(k.wrapping_sub(1)).map(|x| x.1);
    Detection {
        kth_distance: kth,
        nearest: nn.first().map(|x| x.0),
        flagged: kth.is_some_and(|d| d > tau),
    }
}

/// A reference index over fingerprint distributions.
#[derive(Clone, Debug)]
pub struct AnomalyIndex {
    /// Configuration.
    pub config: AnomalyConfig,
    directions: Directions,
    reference: Vec<Projected>,
    /// The HNSW graph over the reference events.
    pub hnsw: Hnsw,
}

impl AnomalyIndex {
    /// Project the reference clouds and build the index (insertion in input
    /// order).
    pub fn build(reference: &[Cloud], config: AnomalyConfig) -> Result<Self, AnomalyError> {
        if reference.is_empty() || config.k == 0 || !config.tau.is_finite() {
            return Err(AnomalyError::new(
                "need references, k >= 1 and a finite tau",
            ));
        }
        let directions = Directions::new(reference[0].dim, config.directions, config.hnsw.seed)?;
        let proj: Vec<Projected> = reference
            .iter()
            .map(|c| directions.project(c))
            .collect::<Result<_, _>>()?;
        let hnsw = Hnsw::build(config.hnsw, proj.len(), &|a, b| sw1(&proj[a], &proj[b]))?;
        Ok(Self {
            config,
            directions,
            reference: proj,
            hnsw,
        })
    }

    /// SW1 between reference events `a` and `b`.
    #[must_use]
    pub fn distance(&self, a: usize, b: usize) -> f64 {
        sw1(&self.reference[a], &self.reference[b])
    }

    /// The `k` nearest references of `query` as `(reference id, SW1)`,
    /// ordered by `(distance, id)` (HNSW search, VDS-1 section 19.3).
    pub fn knn(&self, query: &Cloud, k: usize) -> Result<Vec<(usize, f64)>, AnomalyError> {
        let pq = self.directions.project(query)?;
        Ok(self.hnsw.knn(&|j| sw1(&pq, &self.reference[j]), k))
    }

    /// Flag each query by its `k`-th nearest reference.
    pub fn detect(&self, queries: &[Cloud]) -> Result<Vec<Detection>, AnomalyError> {
        queries
            .iter()
            .map(|q| {
                let nn = self.knn(q, self.config.k)?;
                Ok(detection(&nn, self.config.k, self.config.tau))
            })
            .collect()
    }

    /// Leave-one-out detection of every reference event: its `k` nearest
    /// other references (searched with `k + 1`, excluding itself).
    #[must_use]
    pub fn detect_self(&self) -> Vec<Detection> {
        (0..self.reference.len())
            .map(|i| {
                let nn: Vec<(usize, f64)> = self
                    .hnsw
                    .knn(&|j| self.distance(i, j), self.config.k + 1)
                    .into_iter()
                    .filter(|&(j, _)| j != i)
                    .take(self.config.k)
                    .collect();
                detection(&nn, self.config.k, self.config.tau)
            })
            .collect()
    }

    /// The output contract for the reference set (leave-one-out flags,
    /// layer-0 HNSW edges) plus optional queries (ids after the references,
    /// flagged against the references, linked to their `k` nearest).
    pub fn graph(&self, queries: &[Cloud]) -> Result<Graph, AnomalyError> {
        let n_ref = self.reference.len();
        let mut flagged: Vec<bool> = self.detect_self().iter().map(|d| d.flagged).collect();
        let mut edges: Vec<(usize, usize, f64)> = self
            .hnsw
            .layer0_edges()
            .into_iter()
            .map(|(i, j)| (i, j, self.distance(i, j)))
            .collect();
        for (qi, q) in queries.iter().enumerate() {
            let pq = self.directions.project(q)?;
            let nn = self
                .hnsw
                .knn(&|j| sw1(&pq, &self.reference[j]), self.config.k);
            flagged.push(detection(&nn, self.config.k, self.config.tau).flagged);
            for (j, d) in nn {
                edges.push((j, n_ref + qi, d));
            }
        }
        let n = flagged.len();
        let pairs: Vec<(usize, usize)> = edges.iter().map(|e| (e.0, e.1)).collect();
        let xy = spring_layout(
            n,
            &pairs,
            self.config.layout_iterations,
            self.config.hnsw.seed,
        );
        Ok(Graph {
            nodes: (0..n)
                .map(|i| Node {
                    id: i,
                    x: xy[i].0,
                    y: xy[i].1,
                    flagged: flagged[i],
                })
                .collect(),
            edges,
            tau: self.config.tau,
            config_digest: self
                .config
                .value()
                .digest()
                .map_err(|e| AnomalyError::new(e.0))?,
        })
    }
}

/// Fruchterman-Reingold layout in the unit square (VDS-1 section 19.5):
/// initial positions `(u, u)` per node from `Stream(seed, STREAM_LAYOUT)`;
/// `k = sqrt(1 / n)`; for iteration `t` the temperature is
/// `0.1 (1 - t / T)`; repulsion `k^2 / d` between every ordered pair (sums in
/// increasing `j`), attraction `d^2 / k` along each edge (edges in the given
/// order); each displacement is capped by the temperature and positions are
/// clamped to `[0, 1]`; distances below `1e-9` are raised to `1e-9`.
#[must_use]
pub fn spring_layout(
    n: usize,
    edges: &[(usize, usize)],
    iterations: usize,
    seed: u64,
) -> Vec<(f64, f64)> {
    let mut s = Stream::new(seed, STREAM_LAYOUT);
    let mut pos: Vec<(f64, f64)> = (0..n)
        .map(|_| (s.next_f64_unit(), s.next_f64_unit()))
        .collect();
    if n < 2 {
        return pos;
    }
    let k = (1.0 / n as f64).sqrt();
    let k2 = k * k;
    for t in 0..iterations {
        let temp = 0.1 * (1.0 - t as f64 / iterations as f64);
        let mut disp = vec![(0.0f64, 0.0f64); n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dx = pos[i].0 - pos[j].0;
                let dy = pos[i].1 - pos[j].1;
                let d = (dx * dx + dy * dy).sqrt().max(1e-9);
                let f = k2 / d;
                disp[i].0 += dx / d * f;
                disp[i].1 += dy / d * f;
            }
        }
        for &(i, j) in edges {
            let dx = pos[i].0 - pos[j].0;
            let dy = pos[i].1 - pos[j].1;
            let d = (dx * dx + dy * dy).sqrt().max(1e-9);
            let f = d * d / k;
            disp[i].0 -= dx / d * f;
            disp[i].1 -= dy / d * f;
            disp[j].0 += dx / d * f;
            disp[j].1 += dy / d * f;
        }
        for (p, dp) in pos.iter_mut().zip(&disp) {
            let len = (dp.0 * dp.0 + dp.1 * dp.1).sqrt().max(1e-9);
            let step = len.min(temp);
            p.0 = (p.0 + dp.0 / len * step).clamp(0.0, 1.0);
            p.1 = (p.1 + dp.1 / len * step).clamp(0.0, 1.0);
        }
    }
    pos
}

/// A node of the output graph.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Node {
    /// Event id.
    pub id: usize,
    /// Layout x in `[0, 1]`.
    pub x: f64,
    /// Layout y in `[0, 1]`.
    pub y: f64,
    /// Anomaly flag.
    pub flagged: bool,
}

/// The output contract.
#[derive(Clone, Debug, PartialEq)]
pub struct Graph {
    /// Nodes in id order.
    pub nodes: Vec<Node>,
    /// Edges `(i, j, sw1)`.
    pub edges: Vec<(usize, usize, f64)>,
    /// Threshold.
    pub tau: f64,
    /// SHA3-256 of the canonical configuration.
    pub config_digest: String,
}

impl Graph {
    /// Canonical JSON: `{config_digest, edges: [{i, j, sw1}], nodes: [{flagged,
    /// id, x, y}], numerics_version, tau, tier2_version}` with a trailing
    /// newline.
    pub fn to_json(&self) -> Result<String, AnomalyError> {
        let v = object([
            ("config_digest", Value::Str(self.config_digest.clone())),
            (
                "edges",
                Value::Array(
                    self.edges
                        .iter()
                        .map(|&(i, j, d)| {
                            object([
                                ("i", Value::Int(i as i64)),
                                ("j", Value::Int(j as i64)),
                                ("sw1", Value::Num(d)),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "nodes",
                Value::Array(
                    self.nodes
                        .iter()
                        .map(|n| {
                            object([
                                ("flagged", Value::Bool(n.flagged)),
                                ("id", Value::Int(n.id as i64)),
                                ("x", Value::Num(n.x)),
                                ("y", Value::Num(n.y)),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("numerics_version", Value::Int(i64::from(NUMERICS_VERSION))),
            ("tau", Value::Num(self.tau)),
            ("tier2_version", Value::Int(i64::from(TIER2_VERSION))),
        ]);
        let mut s = v.canonical().map_err(|e| AnomalyError::new(e.0))?;
        s.push('\n');
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clouds(n: usize, shift_last: bool) -> Vec<Cloud> {
        let mut s = Stream::new(8, 8);
        (0..n)
            .map(|i| {
                let off = if shift_last && i == n - 1 { 25.0 } else { 0.0 };
                Cloud::new(3, (0..12).map(|_| s.next_normal_f64() + off).collect()).unwrap()
            })
            .collect()
    }

    #[test]
    fn detects_an_outlier_and_output_is_deterministic() {
        let refs = clouds(60, true);
        let cfg = AnomalyConfig {
            k: 3,
            tau: 3.0,
            layout_iterations: 20,
            ..AnomalyConfig::default()
        };
        let idx = AnomalyIndex::build(&refs, cfg).unwrap();
        let flags = idx.detect_self();
        assert!(flags[59].flagged);
        assert!(flags[..59].iter().all(|d| !d.flagged));
        let q = clouds(2, true);
        let det = idx.detect(&q).unwrap();
        assert!(!det[0].flagged && det[1].flagged);
        let g = idx.graph(&q).unwrap();
        assert_eq!(g.nodes.len(), 62);
        assert!(
            g.nodes
                .iter()
                .all(|n| (0.0..=1.0).contains(&n.x) && (0.0..=1.0).contains(&n.y))
        );
        let json = g.to_json().unwrap();
        assert_eq!(
            json,
            AnomalyIndex::build(&refs, cfg)
                .unwrap()
                .graph(&q)
                .unwrap()
                .to_json()
                .unwrap()
        );
        assert!(json.starts_with("{\"config_digest\":\""));
    }
}
