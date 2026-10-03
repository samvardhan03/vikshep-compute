//! Deterministic HNSW index (`spec/VDS-1.md` section 19.3), after Malkov
//! and Yashunin (2018), Algorithms 1 and 2 with the simple neighbour
//! selection:
//!
//! * single-threaded insertion in input order;
//! * node levels `floor(-ln(u) / ln(M))`, `u = 1 - next_f64_unit()` from
//!   `Stream(seed, STREAM_HNSW_LEVELS)`, drawn in insertion order, capped at
//!   [`MAX_LEVEL`];
//! * every candidate set is ordered by `(distance, id)` in total order, so
//!   ties break by id;
//! * neighbour lists keep the `M` (layer 0: `2M`) nearest by `(distance, id)`.

use std::collections::{BTreeSet, HashSet};

use vikshep_numerics::rng::Stream;

use crate::AnomalyError;

/// Philox stream of the level assignment.
pub const STREAM_HNSW_LEVELS: u64 = 0x4_0002;
/// Highest level a node can receive.
pub const MAX_LEVEL: usize = 16;

/// Index parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HnswConfig {
    /// Neighbours per node above layer 0 (`2M` on layer 0); at least 2.
    pub m: usize,
    /// Candidate list size during construction.
    pub ef_construction: usize,
    /// Candidate list size during queries (raised to `k` if smaller).
    pub ef_search: usize,
    /// Master seed of the level assignment.
    pub seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 8,
            ef_construction: 64,
            ef_search: 32,
            seed: 0,
        }
    }
}

/// `(distance, id)` in total order.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Key(f64, usize);

impl Eq for Key {}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Key {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

/// The index: per node, its level and its neighbour list on each layer.
#[derive(Clone, Debug)]
pub struct Hnsw {
    /// Parameters.
    pub config: HnswConfig,
    /// Level of each node.
    pub levels: Vec<usize>,
    /// `links[node][layer]` neighbour ids.
    pub links: Vec<Vec<Vec<usize>>>,
    /// Entry point.
    pub entry: Option<usize>,
    /// Level of the entry point.
    pub top: usize,
    levels_stream: Stream,
}

impl PartialEq for Hnsw {
    /// Equal graphs (configuration, levels, links, entry point); the level
    /// stream's position follows from the node count.
    fn eq(&self, other: &Self) -> bool {
        self.config == other.config
            && self.levels == other.levels
            && self.links == other.links
            && self.entry == other.entry
            && self.top == other.top
    }
}

impl Hnsw {
    /// An empty index.
    pub fn new(config: HnswConfig) -> Result<Self, AnomalyError> {
        if config.m < 2 || config.ef_construction == 0 || config.ef_search == 0 {
            return Err(AnomalyError::new(
                "HNSW needs M >= 2 and positive ef values",
            ));
        }
        Ok(Self {
            config,
            levels: Vec::new(),
            links: Vec::new(),
            entry: None,
            top: 0,
            levels_stream: Stream::new(config.seed, STREAM_HNSW_LEVELS),
        })
    }

    /// Build by inserting items `0..n` in order; `dist(a, b)` between items.
    pub fn build(
        config: HnswConfig,
        n: usize,
        dist: &dyn Fn(usize, usize) -> f64,
    ) -> Result<Self, AnomalyError> {
        let mut h = Self::new(config)?;
        for i in 0..n {
            h.insert(dist);
            debug_assert_eq!(h.levels.len(), i + 1);
        }
        Ok(h)
    }

    /// Number of nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.levels.len()
    }

    /// True if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    fn max_links(&self, layer: usize) -> usize {
        if layer == 0 {
            2 * self.config.m
        } else {
            self.config.m
        }
    }

    fn draw_level(&mut self) -> usize {
        let u = 1.0 - self.levels_stream.next_f64_unit();
        let ml = 1.0 / vikshep_detmath::ln(self.config.m as f64);
        let l = (-vikshep_detmath::ln(u) * ml).floor();
        (l as usize).min(MAX_LEVEL)
    }

    /// Algorithm 2 (SEARCH-LAYER): the `ef` nearest nodes to the query found
    /// from `entry` on `layer`, ascending by `(distance, id)`.
    fn search_layer(
        &self,
        dq: &dyn Fn(usize) -> f64,
        entry: &[Key],
        ef: usize,
        layer: usize,
    ) -> Vec<Key> {
        let mut visited: HashSet<usize> = entry.iter().map(|k| k.1).collect();
        let mut cand: BTreeSet<Key> = entry.iter().copied().collect();
        let mut found: BTreeSet<Key> = entry.iter().copied().collect();
        while let Some(c) = cand.pop_first() {
            let worst = *found.last().expect("non-empty");
            if c > worst {
                break;
            }
            for &e in &self.links[c.1][layer] {
                if visited.insert(e) {
                    let k = Key(dq(e), e);
                    let worst = *found.last().expect("non-empty");
                    if found.len() < ef || k < worst {
                        cand.insert(k);
                        found.insert(k);
                        if found.len() > ef {
                            found.pop_last();
                        }
                    }
                }
            }
        }
        found.into_iter().collect()
    }

    /// Greedy descent from the entry point to `layer + 1`, then the `ef`
    /// nearest on `layer` (used by queries and insertions).
    fn descend(&self, dq: &dyn Fn(usize) -> f64, down_to: usize) -> Vec<Key> {
        let e = self.entry.expect("non-empty index");
        let mut ep = vec![Key(dq(e), e)];
        let mut layer = self.top;
        while layer > down_to {
            ep = vec![self.search_layer(dq, &ep, 1, layer)[0]];
            layer -= 1;
        }
        ep
    }

    /// Algorithm 1 (INSERT) of the next item id.
    fn insert(&mut self, dist: &dyn Fn(usize, usize) -> f64) {
        let q = self.levels.len();
        let level = self.draw_level();
        self.levels.push(level);
        self.links.push(vec![Vec::new(); level + 1]);
        if self.entry.is_none() {
            self.entry = Some(q);
            self.top = level;
            return;
        }
        let dq = |j: usize| dist(q, j);
        let mut ep = self.descend(&dq, level.min(self.top));
        for layer in (0..=level.min(self.top)).rev() {
            let w = self.search_layer(&dq, &ep, self.config.ef_construction, layer);
            let chosen: Vec<usize> = w.iter().take(self.config.m).map(|k| k.1).collect();
            self.links[q][layer].clone_from(&chosen);
            let cap = self.max_links(layer);
            for &e in &chosen {
                self.links[e][layer].push(q);
                if self.links[e][layer].len() > cap {
                    let mut keyed: Vec<Key> = self.links[e][layer]
                        .iter()
                        .map(|&x| Key(dist(e, x), x))
                        .collect();
                    keyed.sort();
                    keyed.truncate(cap);
                    self.links[e][layer] = keyed.into_iter().map(|k| k.1).collect();
                }
            }
            ep = w;
        }
        if level > self.top {
            self.entry = Some(q);
            self.top = level;
        }
    }

    /// The `k` nearest items to a query (`dq(j)` = distance to item `j`),
    /// ascending by `(distance, id)`, searched with `max(ef_search, k)`.
    #[must_use]
    pub fn knn(&self, dq: &dyn Fn(usize) -> f64, k: usize) -> Vec<(usize, f64)> {
        if self.is_empty() || k == 0 {
            return Vec::new();
        }
        let ep = self.descend(dq, 0);
        self.search_layer(dq, &ep, self.config.ef_search.max(k), 0)
            .into_iter()
            .take(k)
            .map(|key| (key.1, key.0))
            .collect()
    }

    /// Undirected layer-0 edges `(i, j)`, `i < j`, sorted, without duplicates.
    #[must_use]
    pub fn layer0_edges(&self) -> Vec<(usize, usize)> {
        let mut e: BTreeSet<(usize, usize)> = BTreeSet::new();
        for (i, l) in self.links.iter().enumerate() {
            for &j in &l[0] {
                e.insert((i.min(j), i.max(j)));
            }
        }
        e.into_iter().collect()
    }

    /// Canonical graph bytes (all `u32` little-endian): node count, entry
    /// point, top level; then per node in id order its level and, for each
    /// layer `0..=level`, the neighbour count followed by the neighbour ids
    /// in stored order.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        let mut put = |v: usize| b.extend((v as u32).to_le_bytes());
        put(self.len());
        put(self.entry.unwrap_or(0));
        put(self.top);
        for (lvl, layers) in self.levels.iter().zip(&self.links) {
            put(*lvl);
            for l in layers {
                put(l.len());
                for &x in l {
                    put(x);
                }
            }
        }
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn points(n: usize) -> Vec<f64> {
        let mut s = Stream::new(3, 3);
        (0..n).map(|_| s.next_f64_unit()).collect()
    }

    #[test]
    fn exact_on_small_sets_and_deterministic() {
        let p = points(300);
        let dist = |a: usize, b: usize| (p[a] - p[b]).abs();
        let cfg = HnswConfig::default();
        let h = Hnsw::build(cfg, p.len(), &dist).unwrap();
        assert_eq!(h, Hnsw::build(cfg, p.len(), &dist).unwrap());
        let q = 0.4321;
        let got = h.knn(&|j| (p[j] - q).abs(), 5);
        let mut all: Vec<(usize, f64)> = (0..p.len()).map(|j| (j, (p[j] - q).abs())).collect();
        all.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        assert_eq!(got, all[..5].to_vec());
        assert!(h.links.iter().all(|l| l[0].len() <= 2 * cfg.m));
        assert_eq!(
            h.canonical_bytes(),
            Hnsw::build(cfg, p.len(), &dist).unwrap().canonical_bytes()
        );
    }
}
