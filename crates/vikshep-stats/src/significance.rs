//! Cut-based selection and the Asimov significance proxy (`spec/VDS-1.md`
//! sections 16.4 and 16.5).
//!
//! The significance here is an **Asimov proxy, not a Wilks fit**: a
//! closed-form expected significance from weighted signal and background
//! counts passing one cut, not a likelihood-ratio test.

use vikshep_numerics::order::argsort_desc;
use vikshep_numerics::sum::pairwise_sum;

use crate::StatsError;

/// Label attached to every reported significance.
pub const ASIMOV_LABEL: &str = "Asimov proxy, not a Wilks fit";

/// `Z_A = sqrt(2 ((s + b) ln(1 + s / b) - s))` for `s >= 0`, `b > 0`, with
/// the bracket clamped at 0 before the square root; `None` when `b <= 0`
/// (undefined). Asimov proxy, not a Wilks fit.
#[must_use]
pub fn asimov_z(s: f64, b: f64) -> Option<f64> {
    if !s.is_finite() || !b.is_finite() || b <= 0.0 || s < 0.0 {
        return None;
    }
    let inner = (s + b) * vikshep_detmath::ln(1.0 + s / b) - s;
    Some((2.0 * inner.max(0.0)).sqrt())
}

/// Result of a one-sided score cut `score >= threshold`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cut {
    /// Score threshold.
    pub threshold: f64,
    /// Weighted signal passing.
    pub s: f64,
    /// Weighted background passing.
    pub b: f64,
    /// `s / total signal weight`.
    pub sig_eff: f64,
    /// `b / total background weight`.
    pub bkg_eff: f64,
}

/// Cut at a target signal efficiency (VDS-1 section 16.4): sort signal
/// scores descending (total order, ties by index), accumulate their weights
/// sequentially until the running sum reaches `target * total`; the score
/// reached is the threshold, and every event with `score >= threshold`
/// passes. Passing weights are pairwise sums in event order.
pub fn cut_at_signal_efficiency(
    scores: &[f64],
    labels: &[u8],
    weights: &[f64],
    target: f64,
) -> Result<Cut, StatsError> {
    let n = scores.len();
    if labels.len() != n || weights.len() != n {
        return Err(StatsError::new(
            "scores, labels and weights differ in length",
        ));
    }
    if target.is_nan() || target <= 0.0 || target > 1.0 {
        return Err(StatsError::new(
            "target signal efficiency must be in (0, 1]",
        ));
    }
    let sig: Vec<usize> = (0..n).filter(|&i| labels[i] == 1).collect();
    let bkg: Vec<usize> = (0..n).filter(|&i| labels[i] == 0).collect();
    if sig.is_empty() || bkg.is_empty() {
        return Err(StatsError::new("need both signal and background events"));
    }
    let sig_w: Vec<f64> = sig.iter().map(|&i| weights[i]).collect();
    let sig_total = pairwise_sum(&sig_w);
    let bkg_w: Vec<f64> = bkg.iter().map(|&i| weights[i]).collect();
    let bkg_total = pairwise_sum(&bkg_w);
    let sig_scores: Vec<f64> = sig.iter().map(|&i| scores[i]).collect();
    let order = argsort_desc(&sig_scores);
    let goal = target * sig_total;
    let mut acc = 0.0;
    let mut threshold = sig_scores[order[order.len() - 1]];
    for &o in &order {
        acc += sig_w[o];
        if acc >= goal {
            threshold = sig_scores[o];
            break;
        }
    }
    let pass = |idx: &[usize]| {
        let w: Vec<f64> = idx
            .iter()
            .filter(|&&i| scores[i] >= threshold)
            .map(|&i| weights[i])
            .collect();
        pairwise_sum(&w)
    };
    let s = pass(&sig);
    let b = pass(&bkg);
    Ok(Cut {
        threshold,
        s,
        b,
        sig_eff: s / sig_total,
        bkg_eff: b / bkg_total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asimov_limits() {
        assert_eq!(asimov_z(0.0, 10.0), Some(0.0));
        assert_eq!(asimov_z(1.0, 0.0), None);
        // s << b: Z ~ s / sqrt(b)
        let z = asimov_z(10.0, 10_000.0).unwrap();
        assert!((z - 0.1).abs() < 1e-3, "{z}");
        // Z_A = sqrt(2 (75 ln 3 - 50)) = 8.049338... (Python math), below s / sqrt(b) = 10.
        let z = asimov_z(50.0, 25.0).unwrap();
        assert!(
            (z - 8.049_338_065_966_447).abs() < 1e-12 && z < 50.0 / 25.0f64.sqrt(),
            "{z}"
        );
    }

    #[test]
    fn cut_selection() {
        let scores = [0.9, 0.8, 0.7, 0.6, 0.95, 0.1];
        let labels = [1, 1, 1, 1, 0, 0];
        let weights = [1.0; 6];
        let c = cut_at_signal_efficiency(&scores, &labels, &weights, 0.5).unwrap();
        assert_eq!(c.threshold, 0.8);
        assert_eq!((c.s, c.b), (2.0, 1.0));
        assert_eq!((c.sig_eff, c.bkg_eff), (0.5, 0.5));
    }
}
