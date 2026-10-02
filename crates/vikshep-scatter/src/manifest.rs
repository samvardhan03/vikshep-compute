//! Provenance manifests (VDS-1 section 9.3): JSON serialized with the JSON
//! Canonicalization Scheme (RFC 8785) and hashed with SHA3-256.
//!
//! Manifests carry identifiers, integers and strings only, never result
//! floats, so the canonical form needs no number formatting beyond
//! integers.

use std::collections::BTreeMap;

use vikshep_numerics::NUMERICS_VERSION;
use vikshep_numerics::oid::{hex, sha3_256};

use crate::config::ScatterConfig;

/// A JSON value without floating-point numbers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    /// `null`
    Null,
    /// `true` / `false`
    Bool(bool),
    /// An integer (|v| <= 2^53 for interoperability).
    Int(i64),
    /// A string.
    Str(String),
    /// An array.
    Array(Vec<Value>),
    /// An object; keys sorted (ASCII keys sort identically by UTF-16 code
    /// units and by bytes).
    Object(BTreeMap<String, Value>),
}

fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

impl Value {
    /// RFC 8785 canonical serialization.
    ///
    /// # Panics
    /// If an object key is not ASCII (key ordering would then need UTF-16
    /// comparison, which manifests never require).
    #[must_use]
    pub fn canonical(&self) -> String {
        let mut s = String::new();
        self.write(&mut s);
        s
    }

    fn write(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Self::Int(i) => out.push_str(&i.to_string()),
            Self::Str(s) => write_str(out, s),
            Self::Array(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write(out);
                }
                out.push(']');
            }
            Self::Object(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    assert!(k.is_ascii(), "manifest keys must be ASCII");
                    if i > 0 {
                        out.push(',');
                    }
                    write_str(out, k);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }

    /// SHA3-256 of the canonical serialization, lowercase hex.
    #[must_use]
    pub fn hash(&self) -> String {
        hex(&sha3_256(self.canonical().as_bytes()))
    }
}

fn obj(entries: Vec<(&str, Value)>) -> Value {
    Value::Object(
        entries
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    )
}

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
            v.canonical(),
            r#"{"a":"x\"y\\z\n\u0001","b":[-1,true,null]}"#
        );
    }

    #[test]
    fn keys_sort_by_code_unit() {
        // Uppercase sorts before lowercase in RFC 8785 (code-unit order).
        let s = config_value(&ScatterConfig::one_d(256, 2, 1, PadPolicy::Circular)).canonical();
        assert!(
            s.starts_with(r#"{"J":2,"L":1,"Q":1,"carrier_cutoff":1,"#),
            "{s}"
        );
    }
}
