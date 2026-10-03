//! The exact dCorr2 gradient (docs/disco_gradient.md) against central finite
//! differences, and the Pearson proxy against its own finite differences.

use vikshep_numerics::rng::Stream;
use vikshep_stats::dcorr::{dcorr2_exact, dcorr2_grad, pearson_proxy_grad};

fn sample(n: usize, seed: u64, weighted: bool) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut s = Stream::new(seed, 7);
    let m: Vec<f64> = (0..n).map(|_| 50.0 + 100.0 * s.next_f64_unit()).collect();
    // scores in (0, 1), dependent on m
    let sc: Vec<f64> = m
        .iter()
        .map(|v| {
            let z = 0.03 * (v - 100.0) + s.next_normal_f64();
            1.0 / (1.0 + vikshep_detmath::exp(-z))
        })
        .collect();
    let w: Vec<f64> = (0..n)
        .map(|_| {
            if weighted {
                0.05 + 2.0 * s.next_f64_unit()
            } else {
                1.0
            }
        })
        .collect();
    (sc, m, w)
}

fn check(n: usize, seed: u64, weighted: bool) -> f64 {
    let (s, m, w) = sample(n, seed, weighted);
    let (value, g) = dcorr2_grad(&s, &m, Some(&w)).unwrap();
    assert!(value > 0.0);
    let gmax = g.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let mut worst = 0.0f64;
    for k in 0..n {
        let h = 1e-6;
        let mut sp = s.clone();
        let mut sm = s.clone();
        sp[k] += h;
        sm[k] -= h;
        let fd = (dcorr2_exact(&sp, &m, Some(&w)).unwrap()
            - dcorr2_exact(&sm, &m, Some(&w)).unwrap())
            / (2.0 * h);
        worst = worst.max((fd - g[k]).abs() / gmax);
    }
    println!(
        "n={n} seed={seed} weighted={weighted}: dCorr2={value:.6} max|g|={gmax:.3e} worst relative FD error={worst:.3e}"
    );
    worst
}

/// Every component of the analytic gradient agrees with the central finite
/// difference (step 1e-6) to within 1e-6 of the largest component.
#[test]
fn exact_gradient_matches_central_finite_differences() {
    for (n, seed, weighted) in [(40, 1, false), (60, 2, true), (120, 3, true)] {
        assert!(check(n, seed, weighted) < 1e-6);
    }
}

/// The proxy is a different function's gradient: it matches finite
/// differences of the squared weighted Pearson correlation, not of dCorr2.
#[test]
fn pearson_proxy_is_the_gradient_of_r_squared() {
    let (s, m, w) = sample(50, 4, true);
    let r2 = |s: &[f64]| {
        let t: f64 = w.iter().sum::<f64>() + 1e-12;
        let v: Vec<f64> = w.iter().map(|x| x / t).collect();
        let mu_s: f64 = s.iter().zip(&v).map(|(a, b)| a * b).sum();
        let mu_m: f64 = m.iter().zip(&v).map(|(a, b)| a * b).sum();
        let cov: f64 = (0..s.len())
            .map(|i| (s[i] - mu_s) * (m[i] - mu_m) * v[i])
            .sum();
        let vs: f64 = (0..s.len())
            .map(|i| (s[i] - mu_s) * (s[i] - mu_s) * v[i])
            .sum::<f64>()
            + 1e-12;
        let vm: f64 = (0..s.len())
            .map(|i| (m[i] - mu_m) * (m[i] - mu_m) * v[i])
            .sum::<f64>()
            + 1e-12;
        cov * cov / (vs * vm)
    };
    let g = pearson_proxy_grad(&s, &m, &w).unwrap();
    let gmax = g.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    for k in 0..s.len() {
        let mut sp = s.clone();
        let mut sm = s.clone();
        sp[k] += 1e-6;
        sm[k] -= 1e-6;
        let fd = (r2(&sp) - r2(&sm)) / 2e-6;
        assert!((fd - g[k]).abs() < 1e-5 * gmax, "k={k}: {fd} vs {}", g[k]);
    }
    // and it is not the dCorr2 gradient
    let (_, exact) = dcorr2_grad(&s, &m, Some(&w)).unwrap();
    let diff = exact
        .iter()
        .zip(&g)
        .fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
    assert!(diff > 1e-3 * gmax);
}
