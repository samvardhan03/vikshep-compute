//! Conformance suite v1: a fast subset against the committed vectors (the
//! full suite runs in release in CI via `vikshep-conformance run`), and a
//! fault-injection check of the report.

use vikshep_backend_api::{
    BackendError, Canvas, Capabilities, Complex32, ScatterBackend, SubsampleSpec, Twiddles,
};
use vikshep_conformance::suite::{self, Suite};
use vikshep_cpu::CpuBackend;

fn subset() -> Vec<String> {
    suite::QUICK_SUBSET
        .iter()
        .map(|s| (*s).to_string())
        .collect()
}

#[test]
fn suite_expands_deterministically() {
    let a = Suite::load();
    let b = Suite::load();
    assert_eq!(a.cases.len(), b.cases.len());
    assert!(a.cases.len() >= 300);
    for (x, y) in a.cases.iter().zip(&b.cases) {
        assert_eq!(x.id, y.id);
        assert_eq!(x.stream_id, y.stream_id);
    }
}

#[test]
fn subset_matches_committed_vectors() {
    let report = suite::run(&CpuBackend::new(), &suite::default_vectors_dir(), &subset()).unwrap();
    assert!(report.summary.cases > 100, "{}", report.summary.cases);
    assert_eq!(report.summary.fail, 0, "{}", report.to_json());
}

/// The CPU backend with the first modulus result of every call moved by one
/// ulp.
struct OneUlpOff(CpuBackend);

impl ScatterBackend for OneUlpOff {
    fn name(&self) -> &str {
        "faulty"
    }
    fn numerics_version(&self) -> u32 {
        self.0.numerics_version()
    }
    fn capabilities(&self) -> Capabilities {
        self.0.capabilities()
    }
    fn fft(&self, d: &mut [Complex32], c: Canvas, t: &Twiddles<'_>) -> Result<(), BackendError> {
        self.0.fft(d, c, t)
    }
    fn ifft(&self, d: &mut [Complex32], c: Canvas, t: &Twiddles<'_>) -> Result<(), BackendError> {
        self.0.ifft(d, c, t)
    }
    fn mul_real_filter(
        &self,
        d: &mut [Complex32],
        c: Canvas,
        b: &[f32],
        i: &[u32],
    ) -> Result<(), BackendError> {
        self.0.mul_real_filter(d, c, b, i)
    }
    fn modulus(&self, d: &mut [Complex32]) -> Result<(), BackendError> {
        self.0.modulus(d)?;
        d[0].re = f32::from_bits(d[0].re.to_bits() + 1);
        Ok(())
    }
    fn subsample(
        &self,
        i: &[Complex32],
        c: Canvas,
        s: SubsampleSpec,
        o: &mut [f32],
    ) -> Result<(), BackendError> {
        self.0.subsample(i, c, s, o)
    }
}

#[test]
fn report_pinpoints_a_one_ulp_fault() {
    let only = vec![
        "modulus/n1024/uniform".to_string(),
        "fft1d/n8/forward/uniform".to_string(),
    ];
    let report = suite::run(
        &OneUlpOff(CpuBackend::serial()),
        &suite::default_vectors_dir(),
        &only,
    )
    .unwrap();
    assert_eq!(report.summary.cases, 2);
    assert_eq!(report.summary.fail, 1);
    let bad = report.cases.iter().find(|c| c.status == "FAIL").unwrap();
    assert_eq!(bad.id, "modulus/n1024/uniform");
    let o = &bad.outputs[0];
    assert_eq!(o.first_diff_index, Some(0));
    assert_eq!(o.ulp_distance, Some(1));
    let json = report.to_json();
    assert!(json.contains("\"ulp_distance\": 1"), "{json}");
}

#[test]
fn embedded_selftest_and_helpers() {
    let report = suite::selftest(&CpuBackend::new(), "quick").unwrap();
    assert!(report.summary.cases > 100);
    assert_eq!(report.summary.fail, 0, "{}", report.to_json());
    assert!(suite::selftest(&CpuBackend::new(), "nope").is_err());
    let id = "scatter/1d/256/J4-Q1-L1/zero_pad/trivial/o2/uniform";
    let (cfg, x) = suite::scatter_case(id).unwrap();
    assert_eq!(x.len(), 256);
    let out = vikshep_scatter::Scattering::new(cfg)
        .unwrap()
        .run(&CpuBackend::new(), &x)
        .unwrap();
    let (sha, _) = suite::embedded_expected_output(id, "S").unwrap();
    assert_eq!(
        vikshep_numerics::oid::hex(&vikshep_numerics::oid::sha3_256(&out.canonical_bytes())),
        sha
    );
    assert!(suite::scatter_case("fft1d/n2/forward/uniform").is_none());
}

/// A pooled case gives the same result alone as after its trivial sibling
/// (its input is the sibling's, VDS-1 section 11.2).
#[test]
fn pooled_cases_do_not_depend_on_run_order() {
    let only = vec!["scatter/2d/32x32/J1-Q1-L4/circular.circular/so2_relative/".to_string()];
    let report = suite::run(&CpuBackend::new(), &suite::default_vectors_dir(), &only).unwrap();
    assert_eq!(report.summary.cases, 1);
    assert_eq!(report.summary.fail, 0, "{}", report.to_json());
}
