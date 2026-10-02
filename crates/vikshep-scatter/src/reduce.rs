//! Host (Tier-2) reductions of scattering outputs: the scale-free ratio r2
//! and the log-mean fingerprint (VDS-1 sections 14.8 and 14.9).

use vikshep_numerics::sum::pairwise_sum;

use crate::cascade::ScatterOutput;

/// Additive constant of the log-mean fingerprint: exactly `2^-20`.
pub const LOG_MEAN_EPS: f64 = 1.0 / 1_048_576.0;

/// r2 ratios of a batch.
#[derive(Clone, Debug, PartialEq)]
pub struct R2Output {
    /// For each r2 path: (second-order path index, first-order path index)
    /// in the source output.
    pub pairs: Vec<(usize, usize)>,
    /// Number of signals.
    pub batch: usize,
    /// Output samples per path.
    pub out_len: usize,
    /// `[batch][r2 path][spatial]`, binary32.
    pub values: Vec<f32>,
}

impl R2Output {
    /// Canonical bytes (little-endian binary32).
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        self.values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

/// `r2[l1, l2] = S2[l1, l2] / S1[l1]` per spatial position, for every
/// second-order path with `j1 >= carrier_cutoff` (S0 and the dropped
/// first-order carriers do not appear). Evaluated as
/// `fl32(fl64(S2) / fl64(S1))`; a zero denominator gives `+0`.
#[must_use]
pub fn r2(out: &ScatterOutput, carrier_cutoff: u32) -> R2Output {
    let mut first = std::collections::HashMap::new();
    for (i, p) in out.paths.iter().enumerate() {
        if p.order == 1 {
            // trivial: keyed by filter index; pooled: keyed by j1
            let key = p.n.first().map_or(p.j[0] as usize, |&n| n);
            first.insert(key, i);
        }
    }
    let pairs: Vec<(usize, usize)> = out
        .paths
        .iter()
        .enumerate()
        .filter(|(_, p)| p.order == 2 && p.j[0] >= carrier_cutoff)
        .map(|(i, p)| {
            let key = p.n.first().map_or(p.j[0] as usize, |&n| n);
            (i, first[&key])
        })
        .collect();
    let n = out.out_len();
    let mut values = Vec::with_capacity(out.batch * pairs.len() * n);
    for b in 0..out.batch {
        for &(p2, p1) in &pairs {
            let num = out.path(b, p2);
            let den = out.path(b, p1);
            values.extend(num.iter().zip(den).map(|(&a, &d)| {
                if d == 0.0 {
                    0.0
                } else {
                    (f64::from(a) / f64::from(d)) as f32
                }
            }));
        }
    }
    R2Output {
        pairs,
        batch: out.batch,
        out_len: n,
        values,
    }
}

/// Log-mean fingerprint: for each signal and path,
/// `pairwise_sum_s(ln(LOG_MEAN_EPS + |c_s|)) / count` in binary64, with `ln`
/// from `vikshep-detmath`. Returns `[batch][path]`.
#[must_use]
pub fn log_mean(out: &ScatterOutput) -> Vec<f64> {
    let n = out.out_len();
    let mut res = Vec::with_capacity(out.batch * out.paths.len());
    let mut logs = vec![0.0f64; n];
    for b in 0..out.batch {
        for p in 0..out.paths.len() {
            for (dst, &c) in logs.iter_mut().zip(out.path(b, p)) {
                *dst = vikshep_detmath::ln(LOG_MEAN_EPS + f64::from(c).abs());
            }
            res.push(pairwise_sum(&logs) / n as f64);
        }
    }
    res
}

/// Canonical bytes of a log-mean fingerprint (little-endian binary64).
#[must_use]
pub fn log_mean_bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
