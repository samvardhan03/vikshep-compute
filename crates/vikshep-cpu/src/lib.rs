//! CPU reference backend: the normative implementation of the five Tier-1
//! kernels of `spec/VDS-1.md` section 14.6, against which every other
//! backend is checked bit for bit.
//!
//! VDS-1.1 flush-to-zero (section 8.1) is emulated in software: every
//! binary32 operand is flushed before use and every result of `+`, `-`,
//! `*` and `sqrt` is flushed after rounding (`vikshep_numerics::flush::ftz`).
//! `subsample` performs no arithmetic and copies values unchanged.
//!
//! Parallelism (rayon) is used only across independent canvases: each
//! canvas is processed by exactly one task and written to its own fixed
//! location, so the result does not depend on the number of threads.

use rayon::prelude::*;
use vikshep_backend_api::{
    BackendError, Canvas, Capabilities, Complex32, ScatterBackend, SubsampleSpec, Twiddles,
    validate,
};
use vikshep_numerics::NUMERICS_VERSION;
use vikshep_numerics::fft::{Workspace, fft_2d_tables};
use vikshep_numerics::flush::ftz;

/// Minimum number of elements per parallel task for element-wise kernels.
const ELEMENTWISE_CHUNK: usize = 1 << 14;

/// The CPU reference backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend {
    /// Process canvases in parallel (results are identical either way).
    pub parallel: bool,
}

impl CpuBackend {
    /// Parallel CPU backend.
    #[must_use]
    pub const fn new() -> Self {
        Self { parallel: true }
    }

    /// Single-threaded CPU backend.
    #[must_use]
    pub const fn serial() -> Self {
        Self { parallel: false }
    }

    fn transform(
        &self,
        data: &mut [Complex32],
        canvas: Canvas,
        tw: &Twiddles<'_>,
        inverse: bool,
    ) -> Result<(), BackendError> {
        validate::fft(data, canvas, tw)?;
        let run = |chunk: &mut [Complex32], ws: &mut Workspace<f32>| {
            fft_2d_tables(
                chunk,
                canvas.rows,
                canvas.cols,
                tw.cols,
                tw.rows,
                inverse,
                ws,
            );
        };
        if self.parallel {
            data.par_chunks_exact_mut(canvas.len())
                .for_each_init(Workspace::new, |ws, chunk| run(chunk, ws));
        } else {
            let mut ws = Workspace::new();
            for chunk in data.chunks_exact_mut(canvas.len()) {
                run(chunk, &mut ws);
            }
        }
        Ok(())
    }
}

/// `(re, im) -> (re * h, im * h)`, operands and results flushed.
#[inline]
fn mul_real(v: &mut Complex32, h: f32) {
    let h = ftz(h);
    v.re = ftz(ftz(v.re) * h);
    v.im = ftz(ftz(v.im) * h);
}

/// `(re, im) -> (sqrt(re * re + im * im), +0)`, operands and every result
/// (both squares, the sum, the root) flushed.
#[inline]
fn modulus_one(v: &mut Complex32) {
    let (re, im) = (ftz(v.re), ftz(v.im));
    let s = ftz(ftz(re * re) + ftz(im * im));
    *v = Complex32::new(ftz(s.sqrt()), 0.0);
}

fn subsample_one(input: &[Complex32], canvas: Canvas, spec: SubsampleSpec, out: &mut [f32]) {
    let f = spec.factor;
    for a in 0..spec.out_rows {
        let r = if canvas.rows == 1 {
            0
        } else {
            f * (spec.offset_rows + a)
        };
        for c in 0..spec.out_cols {
            let col = f * (spec.offset_cols + c);
            out[a * spec.out_cols + c] = input[r * canvas.cols + col].re;
        }
    }
}

impl ScatterBackend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn numerics_version(&self) -> u32 {
        NUMERICS_VERSION
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            max_log2_len: vikshep_numerics::fft::MAX_LOG2_LEN,
            parallel: self.parallel,
            preserves_subnormals: false,
        }
    }

    fn fft(
        &self,
        data: &mut [Complex32],
        canvas: Canvas,
        tw: &Twiddles<'_>,
    ) -> Result<(), BackendError> {
        self.transform(data, canvas, tw, false)
    }

    fn ifft(
        &self,
        data: &mut [Complex32],
        canvas: Canvas,
        tw_conj: &Twiddles<'_>,
    ) -> Result<(), BackendError> {
        self.transform(data, canvas, tw_conj, true)
    }

    fn mul_real_filter(
        &self,
        data: &mut [Complex32],
        canvas: Canvas,
        bank: &[f32],
        index: &[u32],
    ) -> Result<(), BackendError> {
        validate::mul_real_filter(data, canvas, bank, index)?;
        let len = canvas.len();
        let run = |(b, chunk): (usize, &mut [Complex32])| {
            let start = index[b] as usize * len;
            let h = &bank[start..start + len];
            for (v, &w) in chunk.iter_mut().zip(h) {
                mul_real(v, w);
            }
        };
        if self.parallel {
            data.par_chunks_exact_mut(len).enumerate().for_each(run);
        } else {
            data.chunks_exact_mut(len).enumerate().for_each(run);
        }
        Ok(())
    }

    fn modulus(&self, data: &mut [Complex32]) -> Result<(), BackendError> {
        if self.parallel {
            data.par_chunks_mut(ELEMENTWISE_CHUNK)
                .for_each(|c| c.iter_mut().for_each(modulus_one));
        } else {
            data.iter_mut().for_each(modulus_one);
        }
        Ok(())
    }

    fn subsample(
        &self,
        input: &[Complex32],
        canvas: Canvas,
        spec: SubsampleSpec,
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        validate::subsample(input, canvas, spec, out)?;
        let in_chunks = input.chunks_exact(canvas.len());
        let out_chunks = out.chunks_exact_mut(spec.out_len());
        for (i, o) in in_chunks.zip(out_chunks) {
            subsample_one(i, canvas, spec, o);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vikshep_numerics::fft::Real;

    fn tw(canvas: Canvas, inverse: bool) -> (Vec<Complex32>, Vec<Complex32>) {
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

    #[test]
    fn parallel_equals_serial_bitwise() {
        let canvas = Canvas::two_d(16, 32);
        let mut s = vikshep_numerics::rng::Stream::new(5, 0);
        let data: Vec<Complex32> = (0..canvas.len() * 7)
            .map(|_| Complex32::new(s.next_f32_unit() - 0.5, s.next_f32_unit() - 0.5))
            .collect();
        let (c, r) = tw(canvas, false);
        let t = Twiddles { cols: &c, rows: &r };
        let mut a = data.clone();
        let mut b = data;
        CpuBackend::new().fft(&mut a, canvas, &t).unwrap();
        CpuBackend::serial().fft(&mut b, canvas, &t).unwrap();
        CpuBackend::new().modulus(&mut a).unwrap();
        CpuBackend::serial().modulus(&mut b).unwrap();
        let bits = |v: &[Complex32]| {
            v.iter()
                .map(|z| (z.re.to_bits(), z.im.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&a), bits(&b));
    }

    #[test]
    fn modulus_definition() {
        let mut v = [
            Complex32::new(3.0, -4.0),
            Complex32::new(-0.0, 0.0),
            Complex32::new(f32::from_bits(1), 0.0),
        ];
        CpuBackend::serial().modulus(&mut v).unwrap();
        assert_eq!(v[0], Complex32::new(5.0, 0.0));
        assert_eq!(v[1].re.to_bits(), 0);
        // subnormal operand is flushed: sqrt(0) = 0
        assert_eq!(v[2].re, 0.0);
    }

    #[test]
    fn flush_to_zero_in_kernels() {
        let min = f32::MIN_POSITIVE;
        // squares just below, at and above FLT_MIN (2^-63 squared is 2^-126)
        let r = f32::from_bits(0x2000_0000); // 2^-63
        let below = f32::from_bits(r.to_bits() - 1);
        let above = f32::from_bits(r.to_bits() + 1);
        let mut v = [
            Complex32::new(below, 0.0),
            Complex32::new(r, 0.0),
            Complex32::new(above, -0.0),
        ];
        CpuBackend::serial().modulus(&mut v).unwrap();
        assert_eq!(v[0].re, 0.0, "square below FLT_MIN is flushed");
        assert_eq!(v[1].re, r, "square exactly FLT_MIN is kept");
        assert!(v[2].re > r);
        // product that rounds up to exactly FLT_MIN is kept (tininess after
        // rounding); a tiny negative product flushes to -0
        let canvas = Canvas::one_d(2);
        let mut d = [Complex32::new(min, -min), Complex32::new(-1e-20, 1e-20)];
        let one_minus = f32::from_bits(1.0f32.to_bits() - 1); // 1 - 2^-24
        let bank = [one_minus, 1e-20];
        CpuBackend::serial()
            .mul_real_filter(&mut d[..1], Canvas::one_d(1), &bank[..1], &[0])
            .unwrap();
        assert_eq!(d[0].re.to_bits(), min.to_bits());
        assert_eq!(d[0].im.to_bits(), (-min).to_bits());
        let mut e = [Complex32::new(-1e-20, 1e-20), Complex32::new(3.0, 0.0)];
        CpuBackend::serial()
            .mul_real_filter(&mut e, canvas, &[1e-20, 1e-20], &[0])
            .unwrap();
        assert_eq!(e[0].re.to_bits(), 0x8000_0000, "signed zero preserved");
        assert_eq!(e[0].im.to_bits(), 0);
        // subnormal operands are treated as zero
        let mut f = [Complex32::new(f32::from_bits(5), 2.0)];
        CpuBackend::serial()
            .mul_real_filter(&mut f, Canvas::one_d(1), &[f32::from_bits(7)], &[0])
            .unwrap();
        assert_eq!((f[0].re.to_bits(), f[0].im.to_bits()), (0, 0));
    }

    #[test]
    fn subsample_and_filter() {
        let canvas = Canvas::two_d(4, 4);
        let input: Vec<Complex32> = (0..16).map(|i| Complex32::new(i as f32, -1.0)).collect();
        let spec = SubsampleSpec {
            factor: 2,
            offset_rows: 0,
            offset_cols: 1,
            out_rows: 2,
            out_cols: 1,
        };
        let mut out = vec![0.0; 2];
        CpuBackend::serial()
            .subsample(&input, canvas, spec, &mut out)
            .unwrap();
        assert_eq!(out, vec![2.0, 10.0]);
        let mut data = input.clone();
        let bank: Vec<f32> = (0..32).map(|i| if i < 16 { 2.0 } else { 0.5 }).collect();
        CpuBackend::serial()
            .mul_real_filter(&mut data, canvas, &bank, &[1])
            .unwrap();
        assert_eq!(data[3], Complex32::new(1.5, -0.5));
    }
}
