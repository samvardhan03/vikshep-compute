//! Provenance manifests of scattering runs (VDS-1 sections 9.3 and 14.10),
//! built with `vikshep_numerics::provenance`: RFC 8785 canonical JSON
//! following `spec/provenance.schema.json`.
//!
//! Manifests carry identifiers, integers and strings only, never result
//! floats.

pub use vikshep_numerics::jcs::Value;
use vikshep_numerics::jcs::object as obj;
pub use vikshep_numerics::provenance::{Dtype, Execution, Manifest, TensorRef};

use crate::cascade::Scattering;
use crate::config::ScatterConfig;

/// The configuration as a manifest value.
#[must_use]
pub fn config_value(cfg: &ScatterConfig) -> Value {
    obj(vec![
        ("carrier_cutoff", Value::Int(i64::from(cfg.carrier_cutoff))),
        ("dim", Value::Int(cfg.dim as i64)),
        ("group", Value::Str(cfg.group.name().into())),
        ("J", Value::Int(i64::from(cfg.j))),
        ("L", Value::Int(i64::from(cfg.l))),
        ("max_order", Value::Int(i64::from(cfg.max_order))),
        (
            "pad",
            Value::Array(
                cfg.pad
                    .iter()
                    .map(|p| Value::Str(p.name().into()))
                    .collect(),
            ),
        ),
        ("Q", Value::Int(i64::from(cfg.q))),
        (
            "shape",
            Value::Array(cfg.shape.iter().map(|&n| Value::Int(n as i64)).collect()),
        ),
    ])
}

/// Provenance manifest of one scattering run (VDS-1 sections 9.3 and
/// 14.10): input `signal` (`float32`, `[batch, shape...]`), output
/// `coefficients` (`float32`, `[batch, paths, out_shape...]`), the
/// configuration and the filter-bank fingerprint.
#[must_use]
pub fn scattering_manifest(
    sc: &Scattering,
    batch: usize,
    input_oid: &str,
    output_oid: &str,
    execution: Execution,
) -> Manifest {
    let cfg = sc.config();
    let mut in_shape = vec![batch];
    in_shape.extend_from_slice(&cfg.shape);
    let mut out_shape = vec![batch, sc.paths().len()];
    out_shape.extend(cfg.out_shape());
    Manifest {
        operation: "scattering".into(),
        inputs: vec![TensorRef::new(
            "signal",
            input_oid,
            Dtype::Float32,
            &in_shape,
        )],
        config: config_value(cfg),
        filter_bank_sha3: Some(sc.filters().fingerprint()),
        seeds: Vec::new(),
        outputs: vec![TensorRef::new(
            "coefficients",
            output_oid,
            Dtype::Float32,
            &out_shape,
        )],
        execution,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PadPolicy;

    #[test]
    fn canonical_form() {
        let v = obj(vec![
            (
                "b",
                Value::Array(vec![Value::Int(-1), Value::Bool(true), Value::Null]),
            ),
            ("a", Value::Str("x\"y\\z\n\u{1}".into())),
        ]);
        assert_eq!(
            v.canonical().unwrap(),
            r#"{"a":"x\"y\\z\n\u0001","b":[-1,true,null]}"#
        );
    }

    #[test]
    fn keys_sort_by_code_unit() {
        // Uppercase sorts before lowercase in RFC 8785 (code-unit order).
        let s = config_value(&ScatterConfig::one_d(256, 2, 1, PadPolicy::Circular))
            .canonical()
            .unwrap();
        assert!(
            s.starts_with(r#"{"J":2,"L":1,"Q":1,"carrier_cutoff":1,"#),
            "{s}"
        );
    }

    #[test]
    fn scattering_manifest_shapes() {
        let sc = Scattering::new(ScatterConfig::one_d(256, 2, 1, PadPolicy::Circular)).unwrap();
        let m = scattering_manifest(
            &sc,
            3,
            &"a".repeat(28),
            &"b".repeat(28),
            Execution::local("cpu", "0.1.0"),
        );
        assert_eq!(m.inputs[0].shape, vec![3, 256]);
        assert_eq!(m.outputs[0].shape, vec![3, sc.paths().len(), 64]);
        let j = m.to_json().unwrap();
        assert!(j.contains(r#""operation":"scattering""#), "{j}");
        assert!(j.contains(&sc.filters().fingerprint()));
    }
}
