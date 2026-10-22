//! Determinism Certificates (`spec/certificate.schema.json`).
//!
//! A certificate bundles a provenance manifest with conformance evidence
//! (suite version, numerics version, backend, self-test result) and an
//! optional Ed25519 signature block.  The signature proves which installation
//! issued the certificate.  Correctness of the underlying computation is
//! proven separately by anyone re-running it with the open CPU reference:
//! `vikshep-conformance verify <certificate> --inputs <dir>`.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value as Jv;
use vikshep_cpu::CpuBackend;
use vikshep_numerics::oid::oid as oid_of;
use vikshep_numerics::{NUMERICS_VERSION, jcs};
use vikshep_scatter::{Group, PadPolicy, ScatterConfig, Scattering};

/// `schema` field value of a Determinism Certificate.
pub const SCHEMA: &str = "vikshep.certificate/1";

/// Current conformance suite version (matches `conformance/vectors/v3/`).
pub const SUITE_VERSION: u32 = 3;

/// Outcome of verifying one output tensor.
#[derive(Debug)]
pub struct OutputResult {
    /// Tensor role name from the manifest (e.g. `"coefficients"`).
    pub name: String,
    /// OID recorded in the certificate manifest.
    pub expected_oid: String,
    /// OID computed by re-running the operation.
    pub computed_oid: String,
    /// Whether the two OIDs are equal.
    pub matches: bool,
}

/// Full outcome of `verify()`.
#[derive(Debug)]
pub struct VerifyReport {
    /// Operation name from the manifest.
    pub operation: String,
    /// Per-output comparison results.
    pub outputs: Vec<OutputResult>,
    /// Whether the embedded `manifest_hash` is consistent with the manifest body.
    pub manifest_hash_ok: bool,
    /// `true` iff `manifest_hash_ok` and every output OID matches.
    pub verified: bool,
    /// Non-empty when a structural or I/O error prevented verification.
    pub error: Option<String>,
}

impl VerifyReport {
    fn fail(msg: impl Into<String>) -> Self {
        Self {
            operation: String::new(),
            outputs: Vec::new(),
            manifest_hash_ok: false,
            verified: false,
            error: Some(msg.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// Issuing

/// Issue a Determinism Certificate from a provenance manifest JSON string.
///
/// Runs the quick conformance self-test and records the result.  Returns
/// the certificate as compact JSON.
pub fn issue(manifest_json: &str, backend: &str) -> Result<String, String> {
    let manifest_v: Jv =
        serde_json::from_str(manifest_json).map_err(|e| format!("invalid manifest JSON: {e}"))?;

    if manifest_v.get("schema").and_then(Jv::as_str) != Some("vikshep.provenance/1") {
        return Err("manifest schema is not 'vikshep.provenance/1'".into());
    }
    let numerics_version = manifest_v
        .get("numerics_version")
        .and_then(Jv::as_u64)
        .ok_or("manifest.numerics_version missing")? as u32;
    if numerics_version != NUMERICS_VERSION {
        return Err(format!(
            "manifest numerics_version {numerics_version} != binary NUMERICS_VERSION {NUMERICS_VERSION}"
        ));
    }

    let selftest_passed = crate::suite::selftest(&CpuBackend::new(), "quick")
        .map(|r| r.summary.fail == 0)
        .unwrap_or(false);

    let cert = serde_json::json!({
        "schema": SCHEMA,
        "manifest": manifest_v,
        "conformance": {
            "backend": backend,
            "numerics_version": numerics_version,
            "selftest_result": if selftest_passed { "pass" } else { "fail" },
            "suite_version": SUITE_VERSION
        }
    });

    serde_json::to_string(&cert).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Verifying

/// Verify a Determinism Certificate by re-running its computation with the
/// open CPU reference and comparing every output OID.
///
/// `inputs_dir`: directory where raw input tensors are stored as
/// `<oid>.bin` files (28-character lowercase hex OID, no other prefix).
pub fn verify(cert_json: &str, inputs_dir: &Path) -> VerifyReport {
    let cert: Jv = match serde_json::from_str(cert_json) {
        Ok(v) => v,
        Err(e) => return VerifyReport::fail(format!("invalid certificate JSON: {e}")),
    };

    let schema = cert.get("schema").and_then(Jv::as_str).unwrap_or("");
    if schema != SCHEMA {
        return VerifyReport::fail(format!(
            "unexpected certificate schema {schema:?}; expected {SCHEMA:?}"
        ));
    }

    let manifest = match cert.get("manifest") {
        Some(m) => m,
        None => return VerifyReport::fail("certificate has no 'manifest' field"),
    };

    let operation = match manifest.get("operation").and_then(Jv::as_str) {
        Some(op) => op.to_string(),
        None => return VerifyReport::fail("manifest has no 'operation' field"),
    };

    let manifest_hash_ok = validate_manifest_hash(manifest);

    let backend = CpuBackend::new();
    match operation.as_str() {
        "scattering" => {
            verify_scattering(manifest, &operation, manifest_hash_ok, inputs_dir, &backend)
        }
        other => VerifyReport::fail(format!(
            "unsupported operation {other:?}; only 'scattering' is currently verifiable offline"
        )),
    }
}

// ---------------------------------------------------------------------------
// Manifest-hash validation

fn validate_manifest_hash(manifest: &Jv) -> bool {
    let expected = match manifest.get("manifest_hash").and_then(Jv::as_str) {
        Some(h) => h.to_string(),
        None => return false,
    };
    match build_hashed_subset(manifest) {
        Ok(v) => v.digest().is_ok_and(|h| h == expected),
        Err(_) => false,
    }
}

fn build_hashed_subset(manifest: &Jv) -> Result<jcs::Value, String> {
    use jcs::Value as J;
    Ok(jcs::object([
        (
            "config",
            jv_to_jcs(manifest.get("config").ok_or("missing config")?)?,
        ),
        (
            "filter_bank_sha3",
            match manifest.get("filter_bank_sha3") {
                None | Some(Jv::Null) => J::Null,
                Some(Jv::String(s)) => J::Str(s.clone()),
                _ => return Err("filter_bank_sha3 must be a string or null".into()),
            },
        ),
        (
            "inputs",
            jv_to_jcs(manifest.get("inputs").ok_or("missing inputs")?)?,
        ),
        (
            "numerics_version",
            J::Int(
                manifest
                    .get("numerics_version")
                    .and_then(Jv::as_i64)
                    .ok_or("missing numerics_version")?,
            ),
        ),
        (
            "operation",
            J::Str(
                manifest
                    .get("operation")
                    .and_then(Jv::as_str)
                    .ok_or("missing operation")?
                    .into(),
            ),
        ),
        (
            "outputs",
            jv_to_jcs(manifest.get("outputs").ok_or("missing outputs")?)?,
        ),
        ("schema", J::Str("vikshep.provenance/1".into())),
        (
            "seeds",
            jv_to_jcs(manifest.get("seeds").ok_or("missing seeds")?)?,
        ),
        (
            "tier2_version",
            J::Int(
                manifest
                    .get("tier2_version")
                    .and_then(Jv::as_i64)
                    .ok_or("missing tier2_version")?,
            ),
        ),
    ]))
}

fn jv_to_jcs(v: &Jv) -> Result<jcs::Value, String> {
    use jcs::Value as J;
    match v {
        Jv::Null => Ok(J::Null),
        Jv::Bool(b) => Ok(J::Bool(*b)),
        Jv::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(J::Int(i))
            } else if let Some(f) = n.as_f64() {
                Ok(J::Num(f))
            } else {
                Err(format!("number {n} is out of range for JCS"))
            }
        }
        Jv::String(s) => Ok(J::Str(s.clone())),
        Jv::Array(a) => {
            let items: Result<Vec<_>, _> = a.iter().map(jv_to_jcs).collect();
            Ok(J::Array(items?))
        }
        Jv::Object(m) => {
            let entries: Result<BTreeMap<String, J>, _> = m
                .iter()
                .map(|(k, val)| jv_to_jcs(val).map(|j| (k.clone(), j)))
                .collect();
            Ok(J::Object(entries?))
        }
    }
}

// ---------------------------------------------------------------------------
// Operation dispatch

fn verify_scattering(
    manifest: &Jv,
    operation: &str,
    manifest_hash_ok: bool,
    inputs_dir: &Path,
    backend: &CpuBackend,
) -> VerifyReport {
    let config_v = match manifest.get("config") {
        Some(c) => c,
        None => return VerifyReport::fail("manifest has no 'config' field"),
    };
    let cfg = match parse_scatter_config(config_v) {
        Ok(c) => c,
        Err(e) => return VerifyReport::fail(format!("cannot parse scatter config: {e}")),
    };

    let inputs = match manifest.get("inputs").and_then(Jv::as_array) {
        Some(a) if !a.is_empty() => a,
        _ => return VerifyReport::fail("manifest 'inputs' is missing or empty"),
    };
    let input_oid = match inputs[0].get("oid").and_then(Jv::as_str) {
        Some(o) => o,
        None => return VerifyReport::fail("manifest inputs[0] has no 'oid'"),
    };

    let input_path = inputs_dir.join(format!("{input_oid}.bin"));
    let input_bytes = match std::fs::read(&input_path) {
        Ok(b) => b,
        Err(e) => {
            return VerifyReport::fail(format!(
                "cannot read input tensor {}: {e}",
                input_path.display()
            ));
        }
    };

    let file_oid = oid_of(&input_bytes);
    if file_oid != input_oid {
        return VerifyReport::fail(format!(
            "input OID mismatch: file content has OID {file_oid}, manifest claims {input_oid}"
        ));
    }

    if input_bytes.len() % 4 != 0 {
        return VerifyReport::fail("input tensor byte length is not a multiple of 4");
    }
    let input_f32: Vec<f32> = input_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let sc = match Scattering::new(cfg) {
        Ok(s) => s,
        Err(e) => return VerifyReport::fail(format!("cannot build scattering: {e}")),
    };
    let sig_len = sc.config().signal_len();
    if input_f32.is_empty() || !input_f32.len().is_multiple_of(sig_len) {
        return VerifyReport::fail(format!(
            "input element count {} is not a positive multiple of signal_len {sig_len}",
            input_f32.len()
        ));
    }

    let out = match sc.run(backend, &input_f32) {
        Ok(o) => o,
        Err(e) => return VerifyReport::fail(format!("scatter run failed: {e}")),
    };
    let computed_oid = oid_of(&out.canonical_bytes());

    let expected_outputs = match manifest.get("outputs").and_then(Jv::as_array) {
        Some(a) => a,
        None => return VerifyReport::fail("manifest has no 'outputs' array"),
    };

    let mut outputs = Vec::new();
    for (i, exp) in expected_outputs.iter().enumerate() {
        let name = exp
            .get("name")
            .and_then(Jv::as_str)
            .unwrap_or("(unknown)")
            .to_string();
        let expected_oid = match exp.get("oid").and_then(Jv::as_str) {
            Some(o) => o.to_string(),
            None => return VerifyReport::fail(format!("manifest outputs[{i}] has no 'oid'")),
        };
        // Scattering produces exactly one output tensor (coefficients).
        let comp = if i == 0 {
            computed_oid.clone()
        } else {
            String::new()
        };
        let matches = !comp.is_empty() && comp == expected_oid;
        outputs.push(OutputResult {
            name,
            expected_oid,
            computed_oid: comp,
            matches,
        });
    }

    let all_match = !outputs.is_empty() && outputs.iter().all(|o| o.matches);
    VerifyReport {
        operation: operation.to_string(),
        outputs,
        manifest_hash_ok,
        verified: all_match && manifest_hash_ok,
        error: None,
    }
}

// ---------------------------------------------------------------------------
// Config parsing

fn parse_scatter_config(v: &Jv) -> Result<ScatterConfig, String> {
    let u32_field = |key: &str| -> Result<u32, String> {
        v.get(key)
            .and_then(Jv::as_i64)
            .map(|n| n as u32)
            .ok_or_else(|| format!("config.{key} missing or not an integer"))
    };
    let j = u32_field("J")?;
    let q = u32_field("Q")?;
    let l = u32_field("L")?;
    let max_order = u32_field("max_order")?;
    let carrier_cutoff = u32_field("carrier_cutoff")?;
    let dim = v
        .get("dim")
        .and_then(Jv::as_u64)
        .ok_or("config.dim missing or not a non-negative integer")? as usize;
    let group = match v
        .get("group")
        .and_then(Jv::as_str)
        .ok_or("config.group missing")?
    {
        "trivial" => Group::Trivial,
        "so2_relative" | "so2" => Group::So2Relative,
        other => return Err(format!("unknown group {other:?}")),
    };
    let shape: Vec<usize> = v
        .get("shape")
        .and_then(Jv::as_array)
        .ok_or("config.shape missing")?
        .iter()
        .map(|x| {
            x.as_u64()
                .ok_or_else(|| "shape element must be a non-negative integer".to_string())
                .map(|n| n as usize)
        })
        .collect::<Result<_, _>>()?;
    let pad: Vec<PadPolicy> = v
        .get("pad")
        .and_then(Jv::as_array)
        .ok_or("config.pad missing")?
        .iter()
        .map(|x| match x.as_str() {
            Some("circular") => Ok(PadPolicy::Circular),
            Some("zero_pad") => Ok(PadPolicy::ZeroPad),
            other => Err(format!("unknown pad policy {other:?}")),
        })
        .collect::<Result<_, _>>()?;
    Ok(ScatterConfig {
        dim,
        group,
        j,
        q,
        l,
        max_order,
        pad,
        shape,
        carrier_cutoff,
    })
}

// ---------------------------------------------------------------------------
// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use vikshep_cpu::CpuBackend;
    use vikshep_numerics::oid::oid as oid_of;
    use vikshep_numerics::provenance::Execution;
    use vikshep_scatter::manifest::scattering_manifest;
    use vikshep_scatter::{PadPolicy, ScatterConfig, Scattering};

    struct TestDir(std::path::PathBuf);
    impl TestDir {
        fn new(suffix: &str) -> Self {
            let p = std::env::temp_dir()
                .join(format!("vikshep_cert_test_{}_{suffix}", std::process::id()));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn make_cert_and_dir(suffix: &str) -> (String, TestDir) {
        let cfg = ScatterConfig::one_d(256, 2, 1, PadPolicy::ZeroPad);
        let sc = Scattering::new(cfg).unwrap();
        let sig_len = sc.config().signal_len();
        let input: Vec<f32> = (0..sig_len).map(|i| (i as f32) * 0.001).collect();
        let out = sc.run(&CpuBackend::new(), &input).unwrap();

        let input_bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_oid = oid_of(&input_bytes);
        let output_oid = oid_of(&out.canonical_bytes());

        let manifest = scattering_manifest(
            &sc,
            1,
            &input_oid,
            &output_oid,
            Execution::local("cpu", env!("CARGO_PKG_VERSION")),
        );
        let manifest_json = manifest.to_json().unwrap();
        let cert_json = issue(&manifest_json, "cpu").unwrap();

        let dir = TestDir::new(suffix);
        std::fs::write(dir.path().join(format!("{input_oid}.bin")), &input_bytes).unwrap();
        (cert_json, dir)
    }

    #[test]
    fn issued_certificate_verifies() {
        let (cert_json, dir) = make_cert_and_dir("verify");
        let report = verify(&cert_json, dir.path());
        assert!(
            report.error.is_none(),
            "unexpected error: {:?}",
            report.error
        );
        assert!(report.manifest_hash_ok, "manifest_hash should be valid");
        assert!(
            report.outputs.iter().all(|o| o.matches),
            "all outputs should match"
        );
        assert!(report.verified, "certificate should verify");
    }

    #[test]
    fn tampered_output_oid_fails() {
        let (cert_json, dir) = make_cert_and_dir("tamper_oid");
        let mut cert: serde_json::Value = serde_json::from_str(&cert_json).unwrap();
        cert["manifest"]["outputs"][0]["oid"] = serde_json::json!("deadbeefdeadbeefdeadbeefde");
        let tampered = serde_json::to_string(&cert).unwrap();

        let report = verify(&tampered, dir.path());
        assert!(report.error.is_none(), "should parse, not I/O error");
        assert!(!report.verified, "tampered certificate must not verify");
        assert!(
            report.outputs.iter().any(|o| !o.matches),
            "at least one output should mismatch"
        );
    }

    #[test]
    fn tampered_manifest_hash_fails() {
        let (cert_json, dir) = make_cert_and_dir("tamper_hash");
        let mut cert: serde_json::Value = serde_json::from_str(&cert_json).unwrap();
        cert["manifest"]["manifest_hash"] =
            serde_json::json!("0000000000000000000000000000000000000000000000000000000000000000");
        let tampered = serde_json::to_string(&cert).unwrap();

        let report = verify(&tampered, dir.path());
        assert!(!report.manifest_hash_ok, "manifest_hash check should fail");
        assert!(!report.verified, "tampered manifest hash must not verify");
    }
}
