//! Canonical JSON (RFC 8785, the JSON Canonicalization Scheme) for reports
//! and manifests (`spec/VDS-1.md` sections 9.3 and 15.6).
//!
//! * Object members are sorted by the UTF-16 code units of their keys.
//! * No insignificant whitespace.
//! * Strings escape only `"`, `\`, and control characters (`\b \f \n \r \t`,
//!   otherwise `\u00xx` in lowercase hex).
//! * Numbers are binary64 values written as ECMAScript `Number.prototype.
//!   toString` does: the shortest decimal digit string that round-trips,
//!   placed in fixed or exponential notation by the ECMAScript rules
//!   (exponential when the decimal exponent is below -6 or at least 21).
//!   `-0` is written `0`; NaN and infinities are not representable and are
//!   rejected.

use std::collections::BTreeMap;
use std::fmt;

use crate::oid::{hex, sha3_256};

/// A JSON value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// `null`
    Null,
    /// `true` / `false`
    Bool(bool),
    /// An integer, written exactly. Callers keep `|v| <= 2^53` so that every
    /// consumer reads it back exactly; 64-bit identifiers such as seeds are
    /// written as decimal strings instead (VDS-1 section 15.6).
    Int(i64),
    /// A finite binary64 number, written in ECMAScript shortest form.
    Num(f64),
    /// A string.
    Str(String),
    /// An array.
    Array(Vec<Value>),
    /// An object. The map's own order is irrelevant: members are emitted in
    /// UTF-16 code-unit order of their keys.
    Object(BTreeMap<String, Value>),
}

/// Canonicalization failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JcsError(pub String);

impl fmt::Display for JcsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "JSON canonicalization: {}", self.0)
    }
}

impl std::error::Error for JcsError {}

/// Build an object from `(key, value)` pairs.
#[must_use]
pub fn object<K: Into<String>>(entries: impl IntoIterator<Item = (K, Value)>) -> Value {
    Value::Object(entries.into_iter().map(|(k, v)| (k.into(), v)).collect())
}

/// ECMAScript `Number::toString(x)` for a finite binary64 `x`.
pub fn format_number(x: f64) -> Result<String, JcsError> {
    if !x.is_finite() {
        return Err(JcsError(format!("{x} is not representable in JSON")));
    }
    if x == 0.0 {
        return Ok("0".to_string());
    }
    let neg = x < 0.0;
    let a = x.abs();
    // Shortest round-trip digit count k from Rust's `{:e}`; then the
    // correctly rounded k-digit value (ties to even, as ECMAScript requires
    // among equally close candidates) if it round-trips, else the shortest
    // digits themselves (only possible next to a power of two).
    let shortest = format!("{a:e}");
    let k_short = shortest
        .split_once('e')
        .expect("exponent marker")
        .0
        .chars()
        .filter(char::is_ascii_digit)
        .count();
    let rounded = format!("{:.*e}", k_short - 1, a);
    let sci = if rounded.parse::<f64>() == Ok(a) {
        rounded
    } else {
        shortest
    };
    let (mant, exp) = sci.split_once('e').expect("exponent marker");
    let digits: String = mant.chars().filter(char::is_ascii_digit).collect();
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    let e: i32 = exp.parse().expect("exponent");
    let k = digits.len() as i32; // number of significant digits
    let n = e + 1; // decimal point position: value = 0.d1d2..dk * 10^n
    let mut s = String::new();
    if neg {
        s.push('-');
    }
    if k <= n && n <= 21 {
        s.push_str(digits);
        s.extend(std::iter::repeat_n('0', (n - k) as usize));
    } else if 0 < n && n <= 21 {
        s.push_str(&digits[..n as usize]);
        s.push('.');
        s.push_str(&digits[n as usize..]);
    } else if -6 < n && n <= 0 {
        s.push_str("0.");
        s.extend(std::iter::repeat_n('0', (-n) as usize));
        s.push_str(digits);
    } else {
        s.push_str(&digits[..1]);
        if k > 1 {
            s.push('.');
            s.push_str(&digits[1..]);
        }
        s.push('e');
        s.push(if n > 0 { '+' } else { '-' });
        s.push_str(&(n - 1).abs().to_string());
    }
    Ok(s)
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

fn utf16_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

impl Value {
    /// RFC 8785 canonical serialization.
    pub fn canonical(&self) -> Result<String, JcsError> {
        let mut s = String::new();
        self.write(&mut s)?;
        Ok(s)
    }

    fn write(&self, out: &mut String) -> Result<(), JcsError> {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Self::Int(i) => out.push_str(&i.to_string()),
            Self::Num(x) => out.push_str(&format_number(*x)?),
            Self::Str(s) => write_str(out, s),
            Self::Array(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write(out)?;
                }
                out.push(']');
            }
            Self::Object(m) => {
                let mut keys: Vec<&String> = m.keys().collect();
                keys.sort_by(|a, b| utf16_cmp(a, b));
                out.push('{');
                for (i, k) in keys.into_iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_str(out, k);
                    out.push(':');
                    m[k].write(out)?;
                }
                out.push('}');
            }
        }
        Ok(())
    }

    /// Lowercase hex SHA3-256 of the canonical serialization.
    pub fn digest(&self) -> Result<String, JcsError> {
        Ok(hex(&sha3_256(self.canonical()?.as_bytes())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Number serialization samples from RFC 8785, Appendix B.
    #[test]
    fn rfc8785_appendix_b_numbers() {
        let cases: &[(u64, &str)] = &[
            (0x0000_0000_0000_0000, "0"),
            (0x8000_0000_0000_0000, "0"),
            (0x0000_0000_0000_0001, "5e-324"),
            (0x8000_0000_0000_0001, "-5e-324"),
            (0x7fef_ffff_ffff_ffff, "1.7976931348623157e+308"),
            (0xffef_ffff_ffff_ffff, "-1.7976931348623157e+308"),
            (0x4340_0000_0000_0000, "9007199254740992"),
            (0xc340_0000_0000_0000, "-9007199254740992"),
            (0x4430_0000_0000_0000, "295147905179352830000"),
            (0x44b5_2d02_c7e1_4af5, "9.999999999999997e+22"),
            (0x44b5_2d02_c7e1_4af6, "1e+23"),
            (0x44b5_2d02_c7e1_4af7, "1.0000000000000001e+23"),
            (0x444b_1ae4_d6e2_ef4e, "999999999999999700000"),
            (0x444b_1ae4_d6e2_ef4f, "999999999999999900000"),
            (0x444b_1ae4_d6e2_ef50, "1e+21"),
            (0x3eb0_c6f7_a0b5_ed8c, "9.999999999999997e-7"),
            (0x3eb0_c6f7_a0b5_ed8d, "0.000001"),
            (0x41b3_de43_5555_5553, "333333333.3333332"),
            (0x41b3_de43_5555_5554, "333333333.33333325"),
            (0x41b3_de43_5555_5555, "333333333.3333333"),
            (0x41b3_de43_5555_5556, "333333333.3333334"),
            (0x41b3_de43_5555_5557, "333333333.33333343"),
            (0xbecb_f647_612f_3696, "-0.0000033333333333333333"),
            (0x4314_3ff3_c1cb_0959, "1424953923781206.2"),
        ];
        for &(bits, want) in cases {
            assert_eq!(
                format_number(f64::from_bits(bits)).unwrap(),
                want,
                "{bits:016x}"
            );
        }
        assert!(format_number(f64::NAN).is_err());
        assert!(format_number(f64::INFINITY).is_err());
    }

    #[test]
    fn canonical_object() {
        let v = object([
            (
                "b",
                Value::Array(vec![Value::Int(-1), Value::Num(0.5), Value::Null]),
            ),
            ("a", Value::Str("x\"y\\z\n\u{1}".into())),
            ("A", Value::Bool(true)),
        ]);
        assert_eq!(
            v.canonical().unwrap(),
            r#"{"A":true,"a":"x\"y\\z\n\u0001","b":[-1,0.5,null]}"#
        );
    }

    #[test]
    fn keys_sort_by_utf16_code_units() {
        // U+FB33 (BMP, 0xFB33) sorts after U+1F600 (surrogates 0xD83D ...)
        // in UTF-16 order although its code point is smaller.
        let v = object([("\u{fb33}", Value::Int(1)), ("\u{1f600}", Value::Int(2))]);
        assert_eq!(v.canonical().unwrap(), "{\"\u{1f600}\":2,\"\u{fb33}\":1}");
    }
}
