//! Parity with the public Vikshep Python metric (VDS-1 section 16.8).
//!
//! Fixture: `oracles/python_disco/fixtures.json`, produced by
//! `oracles/python_disco/gen_fixtures.py` from the public repository's
//! `disco.py` and `cli/_train.py` on the inputs of `tests/test_disco.py`.
//! Stated tolerances: dCorr2 within 1e-12 absolute (values lie in [0, 1]);
//! the Pearson proxy gradient within 1e-12 of its largest component. Every
//! original Python assertion is re-checked on the Rust values.

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;
use vikshep_stats::dcorr::{pearson_proxy_grad, weighted_dcorr2};

/// Stated tolerance for dCorr2 parity (absolute).
const TOL_DCORR2: f64 = 1e-12;
/// Stated tolerance for the proxy gradient (relative to the largest entry).
const TOL_PROXY: f64 = 1e-12;

pub fn f64s(v: &Value) -> Vec<f64> {
    let s = v.as_str().unwrap();
    (0..s.len() / 16)
        .map(|i| {
            let mut b = [0u8; 8];
            for (k, byte) in b.iter_mut().enumerate() {
                *byte = u8::from_str_radix(&s[16 * i + 2 * k..16 * i + 2 * k + 2], 16).unwrap();
            }
            f64::from_le_bytes(b)
        })
        .collect()
}

fn fixture() -> Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracles/python_disco/fixtures.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

#[test]
fn dcorr2_matches_python_and_python_assertions_hold() {
    let doc = fixture();
    let mut ours: HashMap<String, f64> = HashMap::new();
    let mut worst = 0.0f64;
    for c in doc["cases"].as_array().unwrap() {
        if c["kind"] != "dcorr2" {
            continue;
        }
        let name = c["name"].as_str().unwrap().to_string();
        let x = f64s(&c["x"]);
        let y = f64s(&c["y"]);
        let w = if c["w"].is_null() {
            None
        } else {
            Some(f64s(&c["w"]))
        };
        let got = weighted_dcorr2(&x, &y, w.as_deref()).unwrap();
        let want = f64s(&c["expected"])[0];
        let err = (got - want).abs();
        println!("{name:<40} rust={got:.17} python={want:.17} |diff|={err:.2e}");
        assert!(err <= TOL_DCORR2, "{name}: {got} vs {want}");
        worst = worst.max(err);
        ours.insert(name, got);
    }
    println!("worst |diff| = {worst:.2e}");
    // The assertions of backend/ingest/tests/test_disco.py, on our values.
    assert!(ours["independent_variables_approx_zero"] < 0.1);
    assert!(ours["independent_uniform"] < 0.1);
    assert!(ours["nonlinear_dependence_detected"] > 0.2);
    assert!(ours["perfect_linear_gives_one"] > 0.95);
    assert!(ours["skewed_weighted"] > ours["skewed_unweighted"] + 0.05);
    assert!(ours["identical_arrays_returns_one"] > 0.99);
    assert_eq!(ours["single_element_returns_zero"], 0.0);
    assert!((ours["weight_normalization_w1"] - ours["weight_normalization_w7"]).abs() < 1e-8);
    // mismatched lengths raise
    assert!(weighted_dcorr2(&[1.0, 2.0], &[1.0], None).is_err());
}

#[test]
fn pearson_proxy_matches_python() {
    let doc = fixture();
    let c = doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["kind"] == "pearson_proxy_grad")
        .unwrap();
    let got = pearson_proxy_grad(&f64s(&c["x"]), &f64s(&c["y"]), &f64s(&c["w"])).unwrap();
    let want = f64s(&c["expected"]);
    let gmax = want.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let worst = got
        .iter()
        .zip(&want)
        .fold(0.0f64, |a, (g, w)| a.max((g - w).abs()))
        / gmax;
    println!("pearson proxy worst relative diff = {worst:.2e}");
    assert!(worst <= TOL_PROXY);
}
