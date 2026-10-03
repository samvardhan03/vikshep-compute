//! Provenance manifests (`spec/VDS-1.md` section 9.3; JSON Schema
//! `spec/provenance.schema.json`).
//!
//! A manifest records what was computed (operation, input and output
//! tensors by OID, configuration, filter-bank fingerprint, versions, seeds)
//! and where it ran (backend, platform, executor, wall clock). It is RFC 8785
//! canonical JSON. `manifest_hash` is the SHA3-256 of the canonical form of
//! the manifest **without** its `execution` and `manifest_hash` members, so
//! two conforming executions of the same computation on any backend, platform
//! or executor have the same hash, and wall-clock times never enter it.
//!
//! Manifests never carry result floats: results are tensors referenced by
//! their OIDs.

use crate::jcs::{JcsError, Value, object};
use crate::{NUMERICS_VERSION, TIER2_VERSION};

/// Value of the `schema` member.
pub const SCHEMA: &str = "vikshep.provenance/1";

/// Target triple this crate was compiled for (for example
/// `x86_64-unknown-linux-gnu`).
pub const PLATFORM: &str = env!("VIKSHEP_TARGET");

/// Element type of a tensor referenced by a manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    /// IEEE-754 binary32, little-endian.
    Float32,
    /// IEEE-754 binary64, little-endian.
    Float64,
    /// Unsigned bytes.
    Uint8,
    /// Unsigned 32-bit integers, little-endian.
    Uint32,
    /// UTF-8 text (for example canonical JSON).
    Utf8,
}

impl Dtype {
    /// Canonical name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Float64 => "float64",
            Self::Uint8 => "uint8",
            Self::Uint32 => "uint32",
            Self::Utf8 => "utf8",
        }
    }

    /// Size of one element in bytes.
    #[must_use]
    pub const fn size(self) -> usize {
        match self {
            Self::Float32 | Self::Uint32 => 4,
            Self::Float64 => 8,
            Self::Uint8 | Self::Utf8 => 1,
        }
    }
}

/// A tensor referenced by OID.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRef {
    /// Role of the tensor in the operation (for example `signal`).
    pub name: String,
    /// 28-character OID (VDS-1 section 9.2).
    pub oid: String,
    /// Element type.
    pub dtype: Dtype,
    /// Row-major shape.
    pub shape: Vec<usize>,
}

impl TensorRef {
    /// Build a reference.
    #[must_use]
    pub fn new(name: &str, oid: &str, dtype: Dtype, shape: &[usize]) -> Self {
        Self {
            name: name.into(),
            oid: oid.into(),
            dtype,
            shape: shape.to_vec(),
        }
    }

    fn value(&self) -> Value {
        object([
            ("dtype", Value::Str(self.dtype.name().into())),
            ("name", Value::Str(self.name.clone())),
            ("oid", Value::Str(self.oid.clone())),
            (
                "shape",
                Value::Array(self.shape.iter().map(|&n| Value::Int(n as i64)).collect()),
            ),
        ])
    }
}

/// Where a computation was executed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Executor {
    /// On the user's machine.
    Local,
    /// On a remote (cloud) worker.
    Cloud,
}

impl Executor {
    /// Canonical name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Cloud => "cloud",
        }
    }

    /// Parse `local` or `cloud`.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "local" => Some(Self::Local),
            "cloud" => Some(Self::Cloud),
            _ => None,
        }
    }
}

/// The execution record (excluded from `manifest_hash`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Execution {
    /// Backend name (for example `cpu`).
    pub backend: String,
    /// Backend version string.
    pub backend_version: String,
    /// Target triple.
    pub platform: String,
    /// Local or cloud.
    pub executor: Executor,
    /// Wall-clock start, milliseconds since the Unix epoch.
    pub started_unix_ms: Option<u64>,
    /// Wall-clock end, milliseconds since the Unix epoch.
    pub finished_unix_ms: Option<u64>,
}

impl Execution {
    /// A local execution on this platform without wall-clock times.
    #[must_use]
    pub fn local(backend: &str, backend_version: &str) -> Self {
        Self {
            backend: backend.into(),
            backend_version: backend_version.into(),
            platform: PLATFORM.into(),
            executor: Executor::Local,
            started_unix_ms: None,
            finished_unix_ms: None,
        }
    }

    fn value(&self) -> Value {
        let wall_clock = match (self.started_unix_ms, self.finished_unix_ms) {
            (None, None) => Value::Null,
            (s, f) => object([
                ("finished_unix_ms", f.map_or(Value::Null, ms)),
                ("started_unix_ms", s.map_or(Value::Null, ms)),
            ]),
        };
        object([
            (
                "backend",
                object([
                    ("name", Value::Str(self.backend.clone())),
                    ("version", Value::Str(self.backend_version.clone())),
                ]),
            ),
            ("executor", Value::Str(self.executor.name().into())),
            ("platform", Value::Str(self.platform.clone())),
            ("wall_clock", wall_clock),
        ])
    }
}

/// Milliseconds as a JSON integer (saturating at 2^53 - 1, beyond any date
/// of interest).
fn ms(v: u64) -> Value {
    Value::Int(v.min((1u64 << 53) - 1) as i64)
}

/// A provenance manifest.
#[derive(Clone, Debug, PartialEq)]
pub struct Manifest {
    /// Operation name (for example `scattering`).
    pub operation: String,
    /// Input tensors.
    pub inputs: Vec<TensorRef>,
    /// Operation configuration (canonical JSON object).
    pub config: Value,
    /// Filter-bank fingerprint (VDS-1 section 14.3.4), when filters were used.
    pub filter_bank_sha3: Option<String>,
    /// Named 64-bit seeds, written as decimal strings.
    pub seeds: Vec<(String, u64)>,
    /// Output tensors.
    pub outputs: Vec<TensorRef>,
    /// Execution record.
    pub execution: Execution,
}

impl Manifest {
    /// The hashed subset: every member except `execution` and
    /// `manifest_hash`.
    #[must_use]
    pub fn hashed_value(&self) -> Value {
        let tensors = |v: &[TensorRef]| Value::Array(v.iter().map(TensorRef::value).collect());
        object([
            ("config", self.config.clone()),
            (
                "filter_bank_sha3",
                self.filter_bank_sha3
                    .as_ref()
                    .map_or(Value::Null, |s| Value::Str(s.clone())),
            ),
            ("inputs", tensors(&self.inputs)),
            ("numerics_version", Value::Int(i64::from(NUMERICS_VERSION))),
            ("operation", Value::Str(self.operation.clone())),
            ("outputs", tensors(&self.outputs)),
            ("schema", Value::Str(SCHEMA.into())),
            (
                "seeds",
                object(
                    self.seeds
                        .iter()
                        .map(|(k, v)| (k.clone(), Value::Str(v.to_string()))),
                ),
            ),
            ("tier2_version", Value::Int(i64::from(TIER2_VERSION))),
        ])
    }

    /// Lowercase hex SHA3-256 of the canonical hashed subset.
    pub fn manifest_hash(&self) -> Result<String, JcsError> {
        self.hashed_value().digest()
    }

    /// The full manifest: the hashed subset plus `execution` and
    /// `manifest_hash`.
    pub fn value(&self) -> Result<Value, JcsError> {
        let hash = self.manifest_hash()?;
        let mut v = self.hashed_value();
        if let Value::Object(m) = &mut v {
            m.insert("execution".into(), self.execution.value());
            m.insert("manifest_hash".into(), Value::Str(hash));
        }
        Ok(v)
    }

    /// Canonical JSON of the full manifest (no trailing newline).
    pub fn to_json(&self) -> Result<String, JcsError> {
        self.value()?.canonical()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Manifest {
        Manifest {
            operation: "scattering".into(),
            inputs: vec![TensorRef::new(
                "signal",
                "0123456789abcdef0123456789ab",
                Dtype::Float32,
                &[1, 256],
            )],
            config: object([("J", Value::Int(2))]),
            filter_bank_sha3: Some("ab".repeat(32)),
            seeds: vec![("init".into(), u64::MAX)],
            outputs: vec![],
            execution: Execution::local("cpu", "0.1.0"),
        }
    }

    #[test]
    fn hash_ignores_execution() {
        let a = sample();
        let mut b = sample();
        b.execution.backend = "cuda".into();
        b.execution.platform = "aarch64-apple-darwin".into();
        b.execution.executor = Executor::Cloud;
        b.execution.started_unix_ms = Some(1);
        b.execution.finished_unix_ms = Some(2);
        assert_eq!(a.manifest_hash().unwrap(), b.manifest_hash().unwrap());
        assert_ne!(a.to_json().unwrap(), b.to_json().unwrap());
        let mut c = sample();
        c.seeds[0].1 = 0;
        assert_ne!(a.manifest_hash().unwrap(), c.manifest_hash().unwrap());
    }

    #[test]
    fn layout() {
        let j = sample().to_json().unwrap();
        assert!(j.starts_with(r#"{"config":{"J":2},"execution":{"backend":{"name":"cpu","version":"0.1.0"},"executor":"local","platform":""#), "{j}");
        assert!(j.contains(r#""seeds":{"init":"18446744073709551615"}"#));
        assert!(j.contains(r#""wall_clock":null"#));
        assert!(j.contains(&format!(
            r#""manifest_hash":"{}""#,
            sample().manifest_hash().unwrap()
        )));
        assert!(!PLATFORM.is_empty());
    }
}
