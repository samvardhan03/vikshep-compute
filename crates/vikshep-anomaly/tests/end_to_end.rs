//! End to end: CPU scattering -> fingerprint distributions -> SW1 -> HNSW ->
//! flags, plus HNSW recall against brute force on SW1.

use vikshep_anomaly::graph::{AnomalyConfig, AnomalyIndex};
use vikshep_anomaly::hnsw::HnswConfig;
use vikshep_anomaly::sw1::{Directions, fingerprint_clouds, sw1};
use vikshep_cpu::CpuBackend;
use vikshep_numerics::rng::Stream;
use vikshep_scatter::reduce::log_mean;
use vikshep_scatter::{Group, PadPolicy, ScatterConfig, Scattering};

fn images(n: usize, anomalous: usize) -> Vec<f32> {
    let mut s = Stream::new(21, 0);
    let mut x = Vec::with_capacity(n * 32 * 32);
    for e in 0..n {
        for _row in 0..32 {
            for c in 0..32 {
                let base = 0.3 * (s.next_f32_unit() - 0.5);
                // the anomalous event carries a strong vertical stripe pattern
                let stripe = if e == anomalous && c % 4 < 2 {
                    2.0
                } else {
                    0.0
                };
                x.push(base + stripe);
            }
        }
    }
    x
}

#[test]
fn scattering_fingerprints_flag_the_odd_event() {
    let cfg = ScatterConfig::two_d(32, 32, 2, 4, [PadPolicy::Circular; 2], Group::Trivial);
    let sc = Scattering::new(cfg).unwrap();
    let n = 40;
    let out = sc.run(&CpuBackend::new(), &images(n, 17)).unwrap();
    let clouds = fingerprint_clouds(&out).unwrap();
    assert_eq!(clouds.len(), n);
    assert_eq!(clouds[0].dim, out.paths.len());
    // the mean of a cloud is the C1 log-mean fingerprint
    let lm = log_mean(&out);
    let p = out.paths.len();
    for k in 0..p {
        let mean: f64 =
            clouds[3].points.chunks(p).map(|pt| pt[k]).sum::<f64>() / clouds[3].len() as f64;
        assert!((mean - lm[3 * p + k]).abs() < 1e-12);
    }
    let idx = AnomalyIndex::build(
        &clouds,
        AnomalyConfig {
            k: 3,
            tau: 0.5,
            layout_iterations: 10,
            ..AnomalyConfig::default()
        },
    )
    .unwrap();
    let flags = idx.detect_self();
    let d17 = flags[17].kth_distance.unwrap();
    let others = flags
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 17)
        .map(|(_, f)| f.kth_distance.unwrap())
        .fold(0.0f64, f64::max);
    println!("k-th neighbour SW1: anomalous {d17:.4}, largest other {others:.4}");
    assert!(d17 > 3.0 * others);
}

#[test]
fn hnsw_recall_on_sw1() {
    let mut s = Stream::new(31, 1);
    let clouds: Vec<_> = (0..400)
        .map(|_| {
            vikshep_anomaly::sw1::Cloud::new(
                4,
                (0..32)
                    .map(|_| s.next_normal_f64() * (1.0 + s.next_f64_unit()))
                    .collect(),
            )
            .unwrap()
        })
        .collect();
    let dirs = Directions::new(4, 16, 0).unwrap();
    let proj: Vec<_> = clouds.iter().map(|c| dirs.project(c).unwrap()).collect();
    let h = vikshep_anomaly::hnsw::Hnsw::build(HnswConfig::default(), proj.len(), &|a, b| {
        sw1(&proj[a], &proj[b])
    })
    .unwrap();
    let (mut hit, mut total) = (0usize, 0usize);
    for q in (0..400).step_by(20) {
        let got: Vec<usize> = h
            .knn(&|j| sw1(&proj[q], &proj[j]), 10)
            .into_iter()
            .map(|x| x.0)
            .collect();
        let mut all: Vec<(f64, usize)> = (0..400).map(|j| (sw1(&proj[q], &proj[j]), j)).collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (_, j) in &all[..10] {
            total += 1;
            hit += usize::from(got.contains(j));
        }
    }
    let recall = hit as f64 / total as f64;
    println!("HNSW recall@10 on SW1: {recall:.3}");
    assert!(recall >= 0.95, "{recall}");
}
