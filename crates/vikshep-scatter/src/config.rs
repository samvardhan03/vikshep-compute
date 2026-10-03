//! Scattering configuration and derived geometry (VDS-1 sections 14.1 and
//! 14.4).

use core::fmt;

use vikshep_backend_api::{Canvas, SubsampleSpec};

/// Boundary handling per axis (VDS-1 section 14.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PadPolicy {
    /// Periodic boundary: the canvas is the signal (length must be a power
    /// of two).
    Circular,
    /// Zero padding: the signal is placed at offset `2^J` in a zero canvas of
    /// length `next_pow2(n + 2 * 2^J)`.
    ZeroPad,
}

impl PadPolicy {
    /// Canonical name used in manifests and conformance files.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Circular => "circular",
            Self::ZeroPad => "zero_pad",
        }
    }
}

/// Symmetry group of the output (VDS-1 section 14.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Group {
    /// Every path is kept.
    Trivial,
    /// 2-D only: pooled over absolute orientation, keeping the relative
    /// orientation `(l2 - l1) mod L` of second-order paths.
    So2Relative,
}

impl Group {
    /// Canonical name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Trivial => "trivial",
            Self::So2Relative => "so2_relative",
        }
    }
}

/// Default first scale kept by the r2 ratio (drops `j1 = 0` carriers).
pub const DEFAULT_CARRIER_CUTOFF: u32 = 1;

/// A scattering configuration.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScatterConfig {
    /// 1 or 2.
    pub dim: usize,
    /// Output group.
    pub group: Group,
    /// Number of octaves; outputs are subsampled by `2^J`.
    pub j: u32,
    /// First-order wavelets per octave (1-D only; must be 1 in 2-D).
    pub q: u32,
    /// Number of orientations (2-D only; must be 1 in 1-D).
    pub l: u32,
    /// 1 or 2.
    pub max_order: u32,
    /// Pad policy per axis (`dim` entries, rows first).
    pub pad: Vec<PadPolicy>,
    /// Signal shape (`dim` entries, rows first).
    pub shape: Vec<usize>,
    /// r2 keeps second-order paths with `j1 >= carrier_cutoff`.
    pub carrier_cutoff: u32,
}

/// A configuration error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid scattering configuration: {}", self.0)
    }
}

impl std::error::Error for ConfigError {}

/// Geometry of one axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AxisGeometry {
    /// Signal length.
    pub n: usize,
    /// Canvas (FFT) length.
    pub canvas: usize,
    /// Index of the first signal sample in the canvas.
    pub offset: usize,
    /// Output length `n / 2^J`.
    pub out: usize,
    /// Output offset in output units (`offset / 2^J`).
    pub out_offset: usize,
}

impl ScatterConfig {
    /// A 1-D configuration with defaults (trivial group, order 2).
    #[must_use]
    pub fn one_d(n: usize, j: u32, q: u32, pad: PadPolicy) -> Self {
        Self {
            dim: 1,
            group: Group::Trivial,
            j,
            q,
            l: 1,
            max_order: 2,
            pad: vec![pad],
            shape: vec![n],
            carrier_cutoff: DEFAULT_CARRIER_CUTOFF,
        }
    }

    /// A 2-D configuration with defaults (order 2).
    #[must_use]
    pub fn two_d(
        rows: usize,
        cols: usize,
        j: u32,
        l: u32,
        pad: [PadPolicy; 2],
        group: Group,
    ) -> Self {
        Self {
            dim: 2,
            group,
            j,
            q: 1,
            l,
            max_order: 2,
            pad: pad.to_vec(),
            shape: vec![rows, cols],
            carrier_cutoff: DEFAULT_CARRIER_CUTOFF,
        }
    }

    /// Check every constraint of VDS-1 section 14.1.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let err = |m: &str| Err(ConfigError(m.to_string()));
        if self.dim != 1 && self.dim != 2 {
            return err("dim must be 1 or 2");
        }
        if self.pad.len() != self.dim || self.shape.len() != self.dim {
            return err("pad and shape must have dim entries");
        }
        if self.j == 0 || self.j > 11 {
            return err("J must be in 1..=11");
        }
        if self.max_order != 1 && self.max_order != 2 {
            return err("max_order must be 1 or 2");
        }
        if self.dim == 1 {
            if self.q == 0 || self.q > 32 {
                return err("Q must be in 1..=32");
            }
            if self.l != 1 {
                return err("L must be 1 in 1-D");
            }
            if self.group != Group::Trivial {
                return err("so2_relative requires dim = 2");
            }
        } else {
            if self.q != 1 {
                return err("Q must be 1 in 2-D");
            }
            if self.l == 0 || !self.l.is_multiple_of(2) || self.l > 32 {
                return err("L must be even and in 2..=32");
            }
        }
        for axis in 0..self.dim {
            self.axis_checked(axis)?;
        }
        Ok(())
    }

    fn axis_checked(&self, axis: usize) -> Result<AxisGeometry, ConfigError> {
        let n = self.shape[axis];
        let step = 1usize << self.j;
        if n == 0 || !n.is_multiple_of(step) {
            return Err(ConfigError(format!(
                "axis {axis}: length {n} is not a positive multiple of 2^J = {step}"
            )));
        }
        let (canvas, offset) = match self.pad[axis] {
            PadPolicy::Circular => {
                if !n.is_power_of_two() {
                    return Err(ConfigError(format!(
                        "axis {axis}: circular axes need a power-of-two length, got {n}"
                    )));
                }
                (n, 0)
            }
            PadPolicy::ZeroPad => ((n + 2 * step).next_power_of_two(), step),
        };
        if !(2..=4096).contains(&canvas) {
            return Err(ConfigError(format!(
                "axis {axis}: canvas length {canvas} outside 2..=4096"
            )));
        }
        Ok(AxisGeometry {
            n,
            canvas,
            offset,
            out: n / step,
            out_offset: offset / step,
        })
    }

    /// Geometry of each axis (rows first). Panics on an invalid config.
    #[must_use]
    pub fn axes(&self) -> Vec<AxisGeometry> {
        (0..self.dim)
            .map(|a| self.axis_checked(a).expect("validated configuration"))
            .collect()
    }

    /// The canvas (FFT shape).
    #[must_use]
    pub fn canvas(&self) -> Canvas {
        let a = self.axes();
        if self.dim == 1 {
            Canvas::one_d(a[0].canvas)
        } else {
            Canvas::two_d(a[0].canvas, a[1].canvas)
        }
    }

    /// The output subsampling window.
    #[must_use]
    pub fn subsample_spec(&self) -> SubsampleSpec {
        let a = self.axes();
        let factor = 1usize << self.j;
        if self.dim == 1 {
            SubsampleSpec {
                factor,
                offset_rows: 0,
                offset_cols: a[0].out_offset,
                out_rows: 1,
                out_cols: a[0].out,
            }
        } else {
            SubsampleSpec {
                factor,
                offset_rows: a[0].out_offset,
                offset_cols: a[1].out_offset,
                out_rows: a[0].out,
                out_cols: a[1].out,
            }
        }
    }

    /// Spatial output shape (`dim` entries).
    #[must_use]
    pub fn out_shape(&self) -> Vec<usize> {
        self.axes().iter().map(|a| a.out).collect()
    }

    /// Number of input samples per signal.
    #[must_use]
    pub fn signal_len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Number of output samples per path.
    #[must_use]
    pub fn out_len(&self) -> usize {
        self.out_shape().iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_pad_geometry() {
        let c = ScatterConfig::two_d(
            64,
            64,
            3,
            8,
            [PadPolicy::ZeroPad, PadPolicy::Circular],
            Group::Trivial,
        );
        c.validate().unwrap();
        let a = c.axes();
        assert_eq!(
            (a[0].canvas, a[0].offset, a[0].out, a[0].out_offset),
            (128, 8, 8, 1)
        );
        assert_eq!(
            (a[1].canvas, a[1].offset, a[1].out, a[1].out_offset),
            (64, 0, 8, 0)
        );
        assert_eq!(c.canvas(), Canvas::two_d(128, 64));
        let s = c.subsample_spec();
        assert_eq!(
            (
                s.factor,
                s.offset_rows,
                s.offset_cols,
                s.out_rows,
                s.out_cols
            ),
            (8, 1, 0, 8, 8)
        );
    }

    #[test]
    fn rejects_bad_configs() {
        assert!(
            ScatterConfig::one_d(100, 2, 1, PadPolicy::Circular)
                .validate()
                .is_err()
        );
        assert!(
            ScatterConfig::one_d(100, 2, 1, PadPolicy::ZeroPad)
                .validate()
                .is_ok()
        );
        assert!(
            ScatterConfig::one_d(4096, 2, 1, PadPolicy::ZeroPad)
                .validate()
                .is_err()
        );
        let mut c = ScatterConfig::one_d(256, 2, 1, PadPolicy::Circular);
        c.group = Group::So2Relative;
        assert!(c.validate().is_err());
    }
}
