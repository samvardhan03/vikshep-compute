//! Calibration parity with the public Vikshep CLI's `train_calibrate`
//! (closed-form ridge solved with numpy.linalg.solve). Stated tolerance:
//! 1e-12 relative on coefficients, bias, r2 and residual std (the solvers
//! differ: LU in numpy, Cholesky here).

use std::path::Path;

use serde_json::Value;
use vikshep_train::calibrate::{DEFAULT_RIDGE, ridge};

const TOL: f64 = 1e-12;

fn f64s(v: &Value) -> Vec<f64> {
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

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

#[test]
fn ridge_matches_python_train_calibrate() {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracles/python_disco/fixtures.json");
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let c = doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["kind"] == "calibrate")
        .unwrap();
    let n = c["rows"].as_u64().unwrap() as usize;
    let d = c["cols"].as_u64().unwrap() as usize;
    let fit = ridge(&f64s(&c["x"]), n, d, &f64s(&c["y"]), DEFAULT_RIDGE).unwrap();
    let mut worst = 0.0f64;
    for (a, b) in fit.coef.iter().zip(f64s(&c["expected_coef"])) {
        worst = worst.max(rel(*a, b));
    }
    for (a, b) in fit.standardizer.mu.iter().zip(f64s(&c["expected_mu"])) {
        worst = worst.max(rel(*a, b));
    }
    for (a, b) in fit.standardizer.std.iter().zip(f64s(&c["expected_std"])) {
        worst = worst.max(rel(*a, b));
    }
    worst = worst.max(rel(fit.bias, f64s(&c["expected_bias"])[0]));
    worst = worst.max(rel(fit.r2, f64s(&c["expected_r2"])[0]));
    worst = worst.max(rel(fit.residual_std, f64s(&c["expected_residual_std"])[0]));
    println!("calibration parity: worst relative difference {worst:.2e}");
    assert!(worst <= TOL);
}
