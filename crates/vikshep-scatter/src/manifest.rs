//! Provenance manifests (VDS-1 section 9.3): JSON serialized with the JSON
//! Canonicalization Scheme (RFC 8785, `vikshep_numerics::jcs`) and hashed
//! with SHA3-256.
//!
//! Manifests carry identifiers, integers and strings only, never result
//! floats.

use vikshep_numerics::NUMERICS_VERSION;
pub use vikshep_numerics::jcs::Value;
use vikshep_numerics::jcs::object as obj;

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

/// Provenance manifest of one scattering run. Every manifest records the
/// filter-bank fingerprint (VDS-1 section 14.3.4).
#[must_use]
pub fn scattering_manifest(
    cfg: &ScatterConfig,
    filter_fingerprint: &str,
    backend: &str,
    input_oid: &str,
    output_oid: &str,
) -> Value {
    obj(vec![
        ("backend", Value::Str(backend.into())),
        ("config", config_value(cfg)),
        ("filter_bank_sha3", Value::Str(filter_fingerprint.into())),
        ("input_oid", Value::Str(input_oid.into())),
        ("kind", Value::Str("vikshep.scattering".into())),
        ("numerics_version", Value::Int(i64::from(NUMERICS_VERSION))),
        ("output_dtype", Value::Str("float32".into())),
        ("output_oid", Value::Str(output_oid.into())),
    ])
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
}
