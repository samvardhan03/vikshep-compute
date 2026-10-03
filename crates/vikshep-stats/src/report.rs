//! Lambda-frontier rows and the benchmark report (`spec/VDS-1.md` sections
//! 16.6 and 16.7).
//!
//! `report.json` is RFC 8785 canonical JSON; `report.md` renders the same
//! numbers with exactly six digits after the decimal point. The report states
//! one falsifiable win condition, evaluated at `lambda_star` against the
//! `lambda = 0` baseline: `Delta-sigma > 0 AND Delta-JSD <= 0`, printed
//! `TRUE` or `FALSE`.

use vikshep_numerics::jcs::{Value, object};
use vikshep_numerics::{NUMERICS_VERSION, TIER2_VERSION};

use crate::StatsError;
use crate::dcorr::weighted_dcorr2;
use crate::divergence::{Binning, jsd};
use crate::significance::{ASIMOV_LABEL, asimov_z, cut_at_signal_efficiency};

/// The win condition, verbatim.
pub const WIN_RULE: &str = "Delta-sigma > 0 AND Delta-JSD <= 0 versus the lambda = 0 baseline";

/// Evaluation settings for one frontier row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvalConfig {
    /// Target signal efficiency of the cut (default 0.5).
    pub target_sig_eff: f64,
    /// Number of equal-width bins of the protected variable (default 20).
    pub n_bins: usize,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            target_sig_eff: 0.5,
            n_bins: 20,
        }
    }
}

/// One row of the lambda frontier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontierRow {
    /// DisCo strength.
    pub lambda: f64,
    /// Score threshold of the cut.
    pub threshold: f64,
    /// Weighted signal passing.
    pub s: f64,
    /// Weighted background passing.
    pub b: f64,
    /// Signal efficiency.
    pub sig_eff: f64,
    /// Background efficiency.
    pub bkg_eff: f64,
    /// Asimov proxy, not a Wilks fit; `None` when `b = 0`.
    pub z_asimov: Option<f64>,
    /// JSD between pre-cut and post-cut background histograms of the
    /// protected variable; `None` when no background passes.
    pub jsd: Option<f64>,
    /// Weighted dCorr2 of score and protected variable on background.
    pub dcorr2_bkg: f64,
}

/// Evaluate a scored sample: cut, Asimov proxy, JSD, background dCorr2.
pub fn evaluate(
    lambda: f64,
    scores: &[f64],
    labels: &[u8],
    weights: &[f64],
    protect: &[f64],
    cfg: EvalConfig,
) -> Result<FrontierRow, StatsError> {
    let cut = cut_at_signal_efficiency(scores, labels, weights, cfg.target_sig_eff)?;
    let bkg: Vec<usize> = (0..scores.len()).filter(|&i| labels[i] == 0).collect();
    let pick = |v: &[f64], idx: &[usize]| idx.iter().map(|&i| v[i]).collect::<Vec<f64>>();
    let m_bkg = pick(protect, &bkg);
    let w_bkg = pick(weights, &bkg);
    let s_bkg = pick(scores, &bkg);
    let binning = Binning::from_range(&m_bkg, cfg.n_bins)?;
    let pre = binning.histogram(&m_bkg, &w_bkg)?;
    let passed: Vec<usize> = (0..bkg.len())
        .filter(|&k| s_bkg[k] >= cut.threshold)
        .collect();
    let post = binning.histogram(&pick(&m_bkg, &passed), &pick(&w_bkg, &passed))?;
    let jsd = if passed.is_empty() {
        None
    } else {
        Some(jsd(&pre, &post)?)
    };
    Ok(FrontierRow {
        lambda,
        threshold: cut.threshold,
        s: cut.s,
        b: cut.b,
        sig_eff: cut.sig_eff,
        bkg_eff: cut.bkg_eff,
        z_asimov: asimov_z(cut.s, cut.b),
        jsd,
        dcorr2_bkg: weighted_dcorr2(&s_bkg, &m_bkg, Some(&w_bkg))?,
    })
}

fn opt(v: Option<f64>) -> Value {
    v.map_or(Value::Null, Value::Num)
}

fn row_value(r: &FrontierRow) -> Value {
    object([
        ("b", Value::Num(r.b)),
        ("bkg_eff", Value::Num(r.bkg_eff)),
        ("dcorr2_bkg", Value::Num(r.dcorr2_bkg)),
        ("jsd", opt(r.jsd)),
        ("lambda", Value::Num(r.lambda)),
        ("s", Value::Num(r.s)),
        ("sig_eff", Value::Num(r.sig_eff)),
        ("threshold", Value::Num(r.threshold)),
        ("z_asimov", opt(r.z_asimov)),
    ])
}

/// Outcome of the win condition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Verdict {
    /// lambda the condition is evaluated at.
    pub lambda_star: f64,
    /// `Z_A(lambda_star) - Z_A(0)`, if both are defined.
    pub delta_sigma: Option<f64>,
    /// `JSD(lambda_star) - JSD(0)`, if both are defined.
    pub delta_jsd: Option<f64>,
    /// `delta_sigma > 0 && delta_jsd <= 0` (false if either is undefined).
    pub win: bool,
}

/// Evaluate the win condition at `lambda_star` against the unique
/// `lambda = 0` row.
pub fn verdict(rows: &[FrontierRow], lambda_star: f64) -> Result<Verdict, StatsError> {
    let find = |l: f64| {
        let hits: Vec<&FrontierRow> = rows.iter().filter(|r| r.lambda == l).collect();
        if hits.len() == 1 {
            Ok(hits[0])
        } else {
            Err(StatsError::new(format!(
                "frontier must contain lambda = {l} exactly once"
            )))
        }
    };
    let base = find(0.0)?;
    let star = find(lambda_star)?;
    let delta_sigma = star.z_asimov.zip(base.z_asimov).map(|(a, b)| a - b);
    let delta_jsd = star.jsd.zip(base.jsd).map(|(a, b)| a - b);
    let win = matches!((delta_sigma, delta_jsd), (Some(ds), Some(dj)) if ds > 0.0 && dj <= 0.0);
    Ok(Verdict {
        lambda_star,
        delta_sigma,
        delta_jsd,
        win,
    })
}

/// Six digits after the decimal point, correctly rounded (ties to even),
/// `-0.000000` written as `0.000000`; `n/a` for undefined values.
#[must_use]
pub fn fixed6(v: Option<f64>) -> String {
    match v {
        None => "n/a".to_string(),
        Some(x) => {
            let s = format!("{x:.6}");
            if s == "-0.000000" {
                "0.000000".to_string()
            } else {
                s
            }
        }
    }
}

/// The two report files.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Report {
    /// `report.json` (RFC 8785, trailing newline).
    pub json: String,
    /// `report.md`.
    pub markdown: String,
}

/// Build the benchmark report. `config` and `inputs` are canonical-JSON
/// objects describing the run (seed, model, sweep) and its inputs (OIDs).
pub fn build_report(
    rows: &[FrontierRow],
    lambda_star: f64,
    config: Value,
    inputs: Value,
) -> Result<Report, StatsError> {
    let v = verdict(rows, lambda_star)?;
    let result = if v.win { "TRUE" } else { "FALSE" };
    let doc = object([
        ("config", config.clone()),
        (
            "frontier",
            Value::Array(rows.iter().map(row_value).collect()),
        ),
        ("inputs", inputs.clone()),
        ("kind", Value::Str("vikshep.benchmark_report".into())),
        ("numerics_version", Value::Int(i64::from(NUMERICS_VERSION))),
        ("significance_method", Value::Str(ASIMOV_LABEL.into())),
        ("tier2_version", Value::Int(i64::from(TIER2_VERSION))),
        (
            "win_condition",
            object([
                ("delta_jsd", opt(v.delta_jsd)),
                ("delta_sigma", opt(v.delta_sigma)),
                ("lambda_star", Value::Num(lambda_star)),
                ("result", Value::Str(result.into())),
                ("rule", Value::Str(WIN_RULE.into())),
            ]),
        ),
    ]);
    let mut json = doc
        .canonical()
        .map_err(|e| StatsError::new(e.to_string()))?;
    json.push('\n');

    let mut md = String::new();
    md.push_str("# Vikshep benchmark report\n\n");
    md.push_str(&format!(
        "numerics_version {NUMERICS_VERSION}, tier2_version {TIER2_VERSION}\n\n"
    ));
    md.push_str(&format!("Significance: {ASIMOV_LABEL}.\n\n"));
    md.push_str("Configuration (canonical JSON):\n\n```json\n");
    md.push_str(
        &config
            .canonical()
            .map_err(|e| StatsError::new(e.to_string()))?,
    );
    md.push_str("\n```\n\nInputs (canonical JSON):\n\n```json\n");
    md.push_str(
        &inputs
            .canonical()
            .map_err(|e| StatsError::new(e.to_string()))?,
    );
    md.push_str("\n```\n\n## Lambda frontier\n\n");
    md.push_str("| lambda | Z_A (Asimov proxy, not a Wilks fit) | JSD | dCorr2 (background) | signal eff. | background eff. | threshold |\n");
    md.push_str("|---:|---:|---:|---:|---:|---:|---:|\n");
    for r in rows {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} |\n",
            fixed6(Some(r.lambda)),
            fixed6(r.z_asimov),
            fixed6(r.jsd),
            fixed6(Some(r.dcorr2_bkg)),
            fixed6(Some(r.sig_eff)),
            fixed6(Some(r.bkg_eff)),
            fixed6(Some(r.threshold)),
        ));
    }
    md.push_str("\n## Win condition\n\n");
    md.push_str(&format!("Rule: {WIN_RULE}.\n\n"));
    md.push_str(&format!(
        "At lambda = {}: Delta-sigma = {}, Delta-JSD = {}.\n\n",
        fixed6(Some(lambda_star)),
        fixed6(v.delta_sigma),
        fixed6(v.delta_jsd)
    ));
    md.push_str(&format!("Result: **{result}**\n"));
    Ok(Report { json, markdown: md })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(lambda: f64, z: Option<f64>, j: Option<f64>) -> FrontierRow {
        FrontierRow {
            lambda,
            threshold: 0.5,
            s: 10.0,
            b: 20.0,
            sig_eff: 0.5,
            bkg_eff: 0.1,
            z_asimov: z,
            jsd: j,
            dcorr2_bkg: 0.01,
        }
    }

    #[test]
    fn verdict_logic() {
        let rows = [
            row(0.0, Some(2.0), Some(0.1)),
            row(1.0, Some(2.5), Some(0.05)),
            row(5.0, Some(1.0), Some(0.01)),
        ];
        assert!(verdict(&rows, 1.0).unwrap().win);
        assert!(!verdict(&rows, 5.0).unwrap().win);
        let undefined = [row(0.0, None, Some(0.1)), row(1.0, Some(2.5), Some(0.05))];
        assert!(!verdict(&undefined, 1.0).unwrap().win);
        assert!(verdict(&rows, 3.0).is_err());
        let equal_jsd = [
            row(0.0, Some(2.0), Some(0.1)),
            row(1.0, Some(2.1), Some(0.1)),
        ];
        assert!(verdict(&equal_jsd, 1.0).unwrap().win);
    }

    #[test]
    fn report_files() {
        let rows = [
            row(0.0, Some(2.0), Some(0.1)),
            row(1.0, Some(2.5), Some(0.05)),
        ];
        let r = build_report(
            &rows,
            1.0,
            object([("seed", Value::Int(7))]),
            object([("x", Value::Str("oid".into()))]),
        )
        .unwrap();
        assert!(r.json.ends_with("}\n"));
        assert!(r.json.contains("\"result\":\"TRUE\""));
        assert!(r.json.contains(ASIMOV_LABEL));
        assert!(r.markdown.contains("Result: **TRUE**"));
        assert!(r.markdown.contains("| 1.000000 | 2.500000 | 0.050000 |"));
        assert_eq!(fixed6(Some(-1e-9)), "0.000000");
        assert_eq!(fixed6(Some(0.0000005)), "0.000000");
        assert_eq!(fixed6(Some(0.0000015)), "0.000002");
    }
}
