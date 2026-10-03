//! Conformance suite v1 (VDS-1 section 11): case expansion from
//! `conformance/cases.toml`, normative input generation, execution through a
//! backend, expected-vector files and the JSON report.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use vikshep_backend_api::{BackendError, Canvas, Complex32, ScatterBackend, Twiddles};
use vikshep_numerics::NUMERICS_VERSION;
use vikshep_numerics::fft::Real;
use vikshep_numerics::oid::{hex, sha3_256};
use vikshep_numerics::rng::Stream;
use vikshep_scatter::reduce::{log_mean, log_mean_bytes, r2};
use vikshep_scatter::{Group, PadPolicy, ScatterConfig, ScatterOutput, Scattering};

/// The suite definition, compiled in.
pub const CASES_TOML: &str = include_str!("../../../conformance/cases.toml");

/// Default directory of the expected vectors.
#[must_use]
pub fn default_vectors_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance/vectors/v1")
}

// ---------------------------------------------------------------------------
// cases.toml

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SuiteToml {
    numerics_version: u32,
    master_seed: u64,
    store_bytes_max: usize,
    fft1d: Fft1dGrid,
    fft2d: Fft2dGrid,
    modulus: ModulusGrid,
    mul_real_filter: MulGrid,
    scatter: Vec<ScatterGrid>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fft1dGrid {
    log2_sizes: Vec<u32>,
    directions: Vec<String>,
    inputs: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fft2dGrid {
    shapes: Vec<[usize; 2]>,
    directions: Vec<String>,
    inputs: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModulusGrid {
    lengths: Vec<usize>,
    inputs: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MulGrid {
    canvases: Vec<[usize; 2]>,
    inputs: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScatterGrid {
    dim: usize,
    shapes: Vec<Vec<usize>>,
    #[serde(rename = "J")]
    j: Vec<u32>,
    #[serde(rename = "Q")]
    q: Vec<u32>,
    #[serde(rename = "L")]
    l: Vec<u32>,
    pads: Vec<Vec<String>>,
    groups: Vec<String>,
    max_order: u32,
    inputs: Vec<String>,
}

/// Input families (VDS-1 section 11.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InputKind {
    /// `2u - 1`, `u = next_f32_unit()`.
    Uniform,
    /// All `+0`.
    Zeros,
    /// All `0.75`.
    Constant,
    /// `1` at the centre sample, `+0` elsewhere.
    Impulse,
    /// `+1` / `-1` in a checkerboard (`(-1)^(r + c)`).
    Alternating,
    /// `(2u - 1) * 2^-120`: values straddling the subnormal range.
    NearSubnormal,
    /// `(2u - 1) * 2^40`: large magnitudes that cannot overflow.
    Large,
}

impl InputKind {
    fn parse(s: &str) -> Self {
        match s {
            "uniform" => Self::Uniform,
            "zeros" => Self::Zeros,
            "constant" => Self::Constant,
            "impulse" => Self::Impulse,
            "alternating" => Self::Alternating,
            "near_subnormal" => Self::NearSubnormal,
            "large" => Self::Large,
            other => panic!("cases.toml: unknown input kind {other:?}"),
        }
    }

    /// Canonical name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Zeros => "zeros",
            Self::Constant => "constant",
            Self::Impulse => "impulse",
            Self::Alternating => "alternating",
            Self::NearSubnormal => "near_subnormal",
            Self::Large => "large",
        }
    }

    const fn random_scale_log2(self) -> Option<i32> {
        match self {
            Self::Uniform => Some(0),
            Self::NearSubnormal => Some(-120),
            Self::Large => Some(40),
            _ => None,
        }
    }
}

fn pow2_f32(e: i32) -> f32 {
    if e >= -126 {
        f32::pow2(e)
    } else {
        f32::pow2(-126) * f32::pow2(e + 126)
    }
}

/// Real input of `rows x cols` samples (VDS-1 section 11.2). Random kinds
/// draw one `next_f32_unit` per sample in row-major order.
#[must_use]
pub fn real_input(kind: InputKind, rows: usize, cols: usize, s: &mut Stream) -> Vec<f32> {
    let n = rows * cols;
    if let Some(e) = kind.random_scale_log2() {
        let scale = pow2_f32(e);
        return (0..n)
            .map(|_| (2.0 * s.next_f32_unit() - 1.0) * scale)
            .collect();
    }
    let centre = (rows / 2) * cols + cols / 2;
    (0..n)
        .map(|i| match kind {
            InputKind::Zeros => 0.0,
            InputKind::Constant => 0.75,
            InputKind::Impulse => {
                if i == centre {
                    1.0
                } else {
                    0.0
                }
            }
            InputKind::Alternating => {
                if (i / cols + i % cols).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                }
            }
            _ => unreachable!(),
        })
        .collect()
}

/// Complex input: random kinds draw `re` then `im` per sample; the other
/// kinds put the real pattern in `re` and `+0` in `im`.
#[must_use]
pub fn complex_input(kind: InputKind, rows: usize, cols: usize, s: &mut Stream) -> Vec<Complex32> {
    if let Some(e) = kind.random_scale_log2() {
        let scale = pow2_f32(e);
        return (0..rows * cols)
            .map(|_| {
                let re = (2.0 * s.next_f32_unit() - 1.0) * scale;
                let im = (2.0 * s.next_f32_unit() - 1.0) * scale;
                Complex32::new(re, im)
            })
            .collect();
    }
    real_input(kind, rows, cols, s)
        .into_iter()
        .map(|v| Complex32::new(v, 0.0))
        .collect()
}

/// What a case computes.
#[derive(Clone, Debug)]
pub enum CaseKind {
    /// 2-D (or 1-D with `rows == 1`) FFT through the backend.
    Fft {
        /// Canvas.
        canvas: Canvas,
        /// True for the inverse transform.
        inverse: bool,
        /// Input family.
        input: InputKind,
    },
    /// The modulus kernel.
    Modulus {
        /// Number of values.
        n: usize,
        /// Input family.
        input: InputKind,
    },
    /// The filter-multiplication kernel with a random filter in `[0, 1)`.
    MulRealFilter {
        /// Canvas.
        canvas: Canvas,
        /// Input family.
        input: InputKind,
    },
    /// A full scattering run.
    Scatter {
        /// Configuration.
        config: ScatterConfig,
        /// Input family.
        input: InputKind,
    },
    /// A C0 determinism sweep (`crate::CASES`).
    Sweep {
        /// Index into `crate::CASES`.
        index: usize,
    },
}

/// One conformance case.
#[derive(Clone, Debug)]
pub struct Case {
    /// Normative identifier.
    pub id: String,
    /// Stream id (first 8 bytes of SHA3-256 of the id, little-endian).
    pub stream_id: u64,
    /// What it computes.
    pub kind: CaseKind,
}

/// The expanded suite.
#[derive(Clone, Debug)]
pub struct Suite {
    /// Master seed of every input stream.
    pub master_seed: u64,
    /// Store outputs up to this size in full.
    pub store_bytes_max: usize,
    /// Cases in canonical order.
    pub cases: Vec<Case>,
}

fn stream_id_of(id: &str) -> u64 {
    let h = sha3_256(id.as_bytes());
    u64::from_le_bytes(h[..8].try_into().unwrap())
}

fn pad_of(s: &str) -> PadPolicy {
    match s {
        "circular" => PadPolicy::Circular,
        "zero_pad" => PadPolicy::ZeroPad,
        other => panic!("cases.toml: unknown pad {other:?}"),
    }
}

fn group_of(s: &str) -> Group {
    match s {
        "trivial" => Group::Trivial,
        "so2_relative" => Group::So2Relative,
        other => panic!("cases.toml: unknown group {other:?}"),
    }
}

fn dir_of(s: &str) -> bool {
    match s {
        "forward" => false,
        "inverse" => true,
        other => panic!("cases.toml: unknown direction {other:?}"),
    }
}

/// Canonical id of a scattering configuration and input.
#[must_use]
pub fn scatter_id(cfg: &ScatterConfig, input: InputKind) -> String {
    let shape: Vec<String> = cfg.shape.iter().map(ToString::to_string).collect();
    let pads: Vec<&str> = cfg.pad.iter().map(|p| p.name()).collect();
    format!(
        "scatter/{}d/{}/J{}-Q{}-L{}/{}/{}/o{}/{}",
        cfg.dim,
        shape.join("x"),
        cfg.j,
        cfg.q,
        cfg.l,
        pads.join("."),
        cfg.group.name(),
        cfg.max_order,
        input.name()
    )
}

impl Suite {
    /// Expand `conformance/cases.toml`.
    #[must_use]
    pub fn load() -> Self {
        Self::parse(CASES_TOML)
    }

    /// Expand a suite definition.
    #[must_use]
    pub fn parse(text: &str) -> Self {
        let t: SuiteToml = toml::from_str(text).expect("cases.toml does not parse");
        assert_eq!(
            t.numerics_version, NUMERICS_VERSION,
            "cases.toml numerics_version"
        );
        let mut cases = Vec::new();
        let mut push = |id: String, kind: CaseKind| {
            cases.push(Case {
                stream_id: stream_id_of(&id),
                id,
                kind,
            });
        };
        for &m in &t.fft1d.log2_sizes {
            for d in &t.fft1d.directions {
                for i in &t.fft1d.inputs {
                    push(
                        format!("fft1d/n{}/{d}/{i}", 1u32 << m),
                        CaseKind::Fft {
                            canvas: Canvas::one_d(1 << m),
                            inverse: dir_of(d),
                            input: InputKind::parse(i),
                        },
                    );
                }
            }
        }
        for &[r, c] in &t.fft2d.shapes {
            for d in &t.fft2d.directions {
                for i in &t.fft2d.inputs {
                    push(
                        format!("fft2d/{r}x{c}/{d}/{i}"),
                        CaseKind::Fft {
                            canvas: Canvas::two_d(r, c),
                            inverse: dir_of(d),
                            input: InputKind::parse(i),
                        },
                    );
                }
            }
        }
        for &n in &t.modulus.lengths {
            for i in &t.modulus.inputs {
                push(
                    format!("modulus/n{n}/{i}"),
                    CaseKind::Modulus {
                        n,
                        input: InputKind::parse(i),
                    },
                );
            }
        }
        for &[r, c] in &t.mul_real_filter.canvases {
            for i in &t.mul_real_filter.inputs {
                push(
                    format!("mul_real_filter/{r}x{c}/{i}"),
                    CaseKind::MulRealFilter {
                        canvas: Canvas::two_d(r, c),
                        input: InputKind::parse(i),
                    },
                );
            }
        }
        for g in &t.scatter {
            for shape in &g.shapes {
                for &j in &g.j {
                    for &q in &g.q {
                        for &l in &g.l {
                            for pads in &g.pads {
                                for group in &g.groups {
                                    for input in &g.inputs {
                                        let cfg = ScatterConfig {
                                            dim: g.dim,
                                            group: group_of(group),
                                            j,
                                            q,
                                            l,
                                            max_order: g.max_order,
                                            pad: pads.iter().map(|p| pad_of(p)).collect(),
                                            shape: shape.clone(),
                                            carrier_cutoff:
                                                vikshep_scatter::config::DEFAULT_CARRIER_CUTOFF,
                                        };
                                        if let Err(e) = cfg.validate() {
                                            panic!("cases.toml: {e} ({cfg:?})");
                                        }
                                        let input = InputKind::parse(input);
                                        push(
                                            scatter_id(&cfg, input),
                                            CaseKind::Scatter { config: cfg, input },
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for (index, c) in crate::CASES.iter().enumerate() {
            push(format!("sweep/{}", c.name), CaseKind::Sweep { index });
        }
        let mut seen = std::collections::HashSet::new();
        for c in &cases {
            assert!(seen.insert(c.id.clone()), "duplicate case id {}", c.id);
        }
        Self {
            master_seed: t.master_seed,
            store_bytes_max: t.store_bytes_max,
            cases,
        }
    }
}

// ---------------------------------------------------------------------------
// Execution

/// Element type of an output, for difference reporting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    /// binary32
    F32,
    /// binary64
    F64,
    /// interleaved binary32 complex
    C32,
}

/// One computed output.
#[derive(Clone, Debug)]
pub struct Output {
    /// Output name.
    pub name: &'static str,
    /// Element type.
    pub dtype: Dtype,
    /// Canonical bytes.
    pub bytes: Vec<u8>,
    /// Whether the bytes may be stored in full (subject to the size limit).
    pub storable: bool,
}

fn c32_bytes(v: &[Complex32]) -> Vec<u8> {
    v.iter()
        .flat_map(|z| [z.re.to_le_bytes(), z.im.to_le_bytes()])
        .flatten()
        .collect()
}

fn tables(canvas: Canvas, inverse: bool) -> (Vec<Complex32>, Vec<Complex32>) {
    let t = |n: usize| -> Vec<Complex32> {
        if n == 1 {
            Vec::new()
        } else if inverse {
            f32::twiddles_inverse(n.trailing_zeros()).to_vec()
        } else {
            f32::twiddles(n.trailing_zeros()).to_vec()
        }
    };
    (t(canvas.cols), t(canvas.rows))
}

/// Runs cases, sharing trivial-group scattering results between groups.
pub struct Runner<'a> {
    backend: &'a dyn ScatterBackend,
    master_seed: u64,
    raw_cache: HashMap<String, ScatterOutput>,
}

impl<'a> Runner<'a> {
    /// A runner for `backend`.
    #[must_use]
    pub fn new(backend: &'a dyn ScatterBackend, master_seed: u64) -> Self {
        Self {
            backend,
            master_seed,
            raw_cache: HashMap::new(),
        }
    }

    /// Compute every output of `case`.
    pub fn run(&mut self, case: &Case) -> Result<Vec<Output>, BackendError> {
        let mut s = Stream::new(self.master_seed, case.stream_id);
        let b = self.backend;
        let one = |name, dtype, bytes| Output {
            name,
            dtype,
            bytes,
            storable: true,
        };
        Ok(match &case.kind {
            CaseKind::Fft {
                canvas,
                inverse,
                input,
            } => {
                let mut d = complex_input(*input, canvas.rows, canvas.cols, &mut s);
                let (tc, tr) = tables(*canvas, *inverse);
                let tw = Twiddles {
                    cols: &tc,
                    rows: &tr,
                };
                if *inverse {
                    b.ifft(&mut d, *canvas, &tw)?;
                } else {
                    b.fft(&mut d, *canvas, &tw)?;
                }
                vec![one("out", Dtype::C32, c32_bytes(&d))]
            }
            CaseKind::Modulus { n, input } => {
                let mut d = complex_input(*input, 1, *n, &mut s);
                b.modulus(&mut d)?;
                vec![one("out", Dtype::C32, c32_bytes(&d))]
            }
            CaseKind::MulRealFilter { canvas, input } => {
                let mut d = complex_input(*input, canvas.rows, canvas.cols, &mut s);
                let h: Vec<f32> = (0..canvas.len()).map(|_| s.next_f32_unit()).collect();
                b.mul_real_filter(&mut d, *canvas, &h, &[0])?;
                vec![one("out", Dtype::C32, c32_bytes(&d))]
            }
            CaseKind::Scatter { config, input } => {
                let rows = if config.dim == 1 { 1 } else { config.shape[0] };
                let cols = *config.shape.last().unwrap();
                let x = real_input(*input, rows, cols, &mut s);
                let sc = Scattering::new(config.clone()).expect("validated configuration");
                let mut trivial = config.clone();
                trivial.group = Group::Trivial;
                let key = scatter_id(&trivial, *input);
                if !self.raw_cache.contains_key(&key) {
                    let raw = sc.run_raw(b, &x)?;
                    self.raw_cache.insert(key.clone(), raw);
                }
                let raw = &self.raw_cache[&key];
                let out = match config.group {
                    Group::Trivial => raw.clone(),
                    Group::So2Relative => sc.pool(raw),
                };
                let rr = r2(&out, config.carrier_cutoff);
                vec![
                    one("S", Dtype::F32, out.canonical_bytes()),
                    one("r2", Dtype::F32, rr.canonical_bytes()),
                    one("log_mean", Dtype::F64, log_mean_bytes(&log_mean(&out))),
                    Output {
                        name: "filters",
                        dtype: Dtype::F32,
                        bytes: sc.filters().canonical_bytes(),
                        storable: false,
                    },
                ]
            }
            CaseKind::Sweep { index } => vec![Output {
                name: "out",
                dtype: Dtype::F64,
                bytes: crate::case_bytes(&crate::CASES[*index]),
                storable: false,
            }],
        })
    }
}

// ---------------------------------------------------------------------------
// Expected vectors

/// Expected value of one output.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpectedOutput {
    /// Output name.
    pub name: String,
    /// Element type.
    pub dtype: Dtype,
    /// Length in bytes.
    pub len_bytes: usize,
    /// Lowercase hex SHA3-256 of the bytes.
    pub sha3_256: String,
    /// Offset into `expected.bin` when stored in full.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<usize>,
}

/// Expected outputs of one case.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpectedCase {
    /// Case id.
    pub id: String,
    /// Outputs in order.
    pub outputs: Vec<ExpectedOutput>,
}

/// `expected.json`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Expected {
    /// Numerics version the vectors belong to.
    pub numerics_version: u32,
    /// How the file was produced.
    pub generated_by: String,
    /// Backend that produced it.
    pub backend: String,
    /// File holding stored output bytes.
    pub bytes_file: String,
    /// Cases in canonical order.
    pub cases: Vec<ExpectedCase>,
}

/// Generate expected vectors with `backend` into `dir` (`expected.json` and
/// `expected.bin`).
pub fn generate(backend: &dyn ScatterBackend, dir: &Path) -> Result<Expected, BackendError> {
    let suite = Suite::load();
    let mut runner = Runner::new(backend, suite.master_seed);
    let mut blob = Vec::new();
    let mut cases = Vec::new();
    for case in &suite.cases {
        let outputs = runner.run(case)?;
        let mut eo = Vec::new();
        for o in outputs {
            let offset = if o.storable && o.bytes.len() <= suite.store_bytes_max {
                let off = blob.len();
                blob.extend_from_slice(&o.bytes);
                Some(off)
            } else {
                None
            };
            eo.push(ExpectedOutput {
                name: o.name.to_string(),
                dtype: o.dtype,
                len_bytes: o.bytes.len(),
                sha3_256: hex(&sha3_256(&o.bytes)),
                offset,
            });
        }
        cases.push(ExpectedCase {
            id: case.id.clone(),
            outputs: eo,
        });
    }
    let expected = Expected {
        numerics_version: NUMERICS_VERSION,
        generated_by: "cargo run --release -p vikshep-conformance -- generate".into(),
        backend: backend.name().into(),
        bytes_file: "expected.bin".into(),
        cases,
    };
    std::fs::create_dir_all(dir).expect("create vectors directory");
    let mut json = serde_json::to_string_pretty(&expected).expect("serialize");
    json.push('\n');
    std::fs::write(dir.join("expected.json"), json).expect("write expected.json");
    std::fs::write(dir.join("expected.bin"), &blob).expect("write expected.bin");
    Ok(expected)
}

/// Load expected vectors from `dir`.
#[must_use]
pub fn load_expected(dir: &Path) -> (Expected, Vec<u8>) {
    let json = std::fs::read_to_string(dir.join("expected.json"))
        .unwrap_or_else(|e| panic!("reading {}: {e}", dir.join("expected.json").display()));
    let expected: Expected = serde_json::from_str(&json).expect("expected.json does not parse");
    let blob = std::fs::read(dir.join(&expected.bytes_file)).expect("reading expected.bin");
    (expected, blob)
}

// ---------------------------------------------------------------------------
// Comparison and report

/// Result of comparing one output.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct OutputResult {
    /// Output name.
    pub name: String,
    /// "PASS" or "FAIL".
    pub status: &'static str,
    /// Index (in elements of the dtype; complex values count as two binary32
    /// elements) of the first differing element, when full bytes are known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_diff_index: Option<usize>,
    /// Distance in ulps at that element.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ulp_distance: Option<u64>,
    /// Expected / actual bits at that element (hex).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_bits: Option<String>,
    /// Actual bits at that element (hex).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual_bits: Option<String>,
    /// Detail for hash-only or structural mismatches.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Result of one case.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CaseResult {
    /// Case id.
    pub id: String,
    /// "PASS" or "FAIL".
    pub status: &'static str,
    /// Per-output results (only for failing cases).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<OutputResult>,
}

/// Totals.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Summary {
    /// Number of cases.
    pub cases: usize,
    /// Passing cases.
    pub pass: usize,
    /// Failing cases.
    pub fail: usize,
}

/// The `run` report. Contains no timing or platform data, so reports from
/// different machines compare byte for byte.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Report {
    /// Backend name.
    pub backend: String,
    /// Numerics version of the backend.
    pub numerics_version: u32,
    /// Totals.
    pub summary: Summary,
    /// Every case in canonical order.
    pub cases: Vec<CaseResult>,
}

fn ordered(bits: u64, sign_bit: u32) -> i128 {
    let mag = i128::from(bits & !(1u64 << sign_bit));
    if (bits >> sign_bit) & 1 == 1 {
        -mag - 1
    } else {
        mag
    }
}

fn first_difference(dtype: Dtype, want: &[u8], got: &[u8]) -> (usize, u64, String, String) {
    let width = if dtype == Dtype::F64 { 8 } else { 4 };
    for (i, (w, g)) in want.chunks(width).zip(got.chunks(width)).enumerate() {
        if w != g {
            let (wb, gb, sign) = if width == 8 {
                (
                    u64::from_le_bytes(w.try_into().unwrap()),
                    u64::from_le_bytes(g.try_into().unwrap()),
                    63,
                )
            } else {
                (
                    u64::from(u32::from_le_bytes(w.try_into().unwrap())),
                    u64::from(u32::from_le_bytes(g.try_into().unwrap())),
                    31,
                )
            };
            let d = (ordered(wb, sign) - ordered(gb, sign)).unsigned_abs();
            let hexw = |b: u64| format!("{b:0w$x}", w = width * 2);
            return (i, u64::try_from(d).unwrap_or(u64::MAX), hexw(wb), hexw(gb));
        }
    }
    (
        want.len().min(got.len()) / width,
        0,
        String::new(),
        String::new(),
    )
}

/// Compare one case's outputs with its expectation.
#[must_use]
pub fn compare(
    case_id: &str,
    outputs: &[Output],
    expected: Option<&ExpectedCase>,
    blob: &[u8],
) -> CaseResult {
    let Some(exp) = expected else {
        return CaseResult {
            id: case_id.to_string(),
            status: "FAIL",
            outputs: vec![OutputResult {
                name: String::new(),
                status: "FAIL",
                first_diff_index: None,
                ulp_distance: None,
                expected_bits: None,
                actual_bits: None,
                detail: Some("no expected vectors for this case".into()),
            }],
        };
    };
    let mut results = Vec::new();
    let mut ok = outputs.len() == exp.outputs.len();
    for (o, e) in outputs.iter().zip(&exp.outputs) {
        let sha = hex(&sha3_256(&o.bytes));
        let mut r = OutputResult {
            name: o.name.to_string(),
            status: "PASS",
            first_diff_index: None,
            ulp_distance: None,
            expected_bits: None,
            actual_bits: None,
            detail: None,
        };
        if o.name != e.name || o.bytes.len() != e.len_bytes {
            r.status = "FAIL";
            r.detail = Some(format!(
                "expected output {} of {} bytes, got {} of {} bytes",
                e.name,
                e.len_bytes,
                o.name,
                o.bytes.len()
            ));
        } else if sha != e.sha3_256 {
            r.status = "FAIL";
            if let Some(off) = e.offset {
                let (i, d, wb, gb) =
                    first_difference(e.dtype, &blob[off..off + e.len_bytes], &o.bytes);
                r.first_diff_index = Some(i);
                r.ulp_distance = Some(d);
                r.expected_bits = Some(wb);
                r.actual_bits = Some(gb);
            } else {
                r.detail = Some(format!("SHA3-256 {sha} != expected {}", e.sha3_256));
            }
        }
        ok &= r.status == "PASS";
        results.push(r);
    }
    CaseResult {
        id: case_id.to_string(),
        status: if ok { "PASS" } else { "FAIL" },
        outputs: if ok { Vec::new() } else { results },
    }
}

/// Run the full suite (or the cases whose id starts with one of `filters`)
/// against the vectors in `dir`.
pub fn run(
    backend: &dyn ScatterBackend,
    dir: &Path,
    filters: &[String],
) -> Result<Report, BackendError> {
    let suite = Suite::load();
    let (expected, blob) = load_expected(dir);
    let by_id: BTreeMap<&str, &ExpectedCase> =
        expected.cases.iter().map(|c| (c.id.as_str(), c)).collect();
    let mut runner = Runner::new(backend, suite.master_seed);
    let mut cases = Vec::new();
    for case in &suite.cases {
        if !filters.is_empty() && !filters.iter().any(|f| case.id.starts_with(f.as_str())) {
            continue;
        }
        let outputs = runner.run(case)?;
        cases.push(compare(
            &case.id,
            &outputs,
            by_id.get(case.id.as_str()).copied(),
            &blob,
        ));
    }
    let pass = cases.iter().filter(|c| c.status == "PASS").count();
    Ok(Report {
        backend: backend.name().into(),
        numerics_version: backend.numerics_version(),
        summary: Summary {
            cases: cases.len(),
            pass,
            fail: cases.len() - pass,
        },
        cases,
    })
}

impl Report {
    /// Pretty JSON with a trailing newline.
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut s = serde_json::to_string_pretty(self).expect("serialize report");
        let _ = writeln!(s);
        s
    }
}
