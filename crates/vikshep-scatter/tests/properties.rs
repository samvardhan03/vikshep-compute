//! Property tests (VDS-1 section 12.2): Littlewood-Paley bounds, discarded
//! imaginary parts, non-expansiveness, circular-shift covariance, and the
//! VDS-1.1 flush-to-zero semantics of the CPU reference (section 8.1).

use vikshep_backend_api::{Canvas, Complex32, ScatterBackend, Twiddles};
use vikshep_cpu::CpuBackend;
use vikshep_numerics::fft::Real;
use vikshep_numerics::rng::Stream;
use vikshep_scatter::{Group, PadPolicy, ScatterConfig, Scattering};

fn input(n: usize, seed: u64) -> Vec<f32> {
    let mut s = Stream::new(seed, 1);
    (0..n).map(|_| s.next_f32_unit() * 2.0 - 1.0).collect()
}

fn configs() -> Vec<ScatterConfig> {
    vec![
        ScatterConfig::one_d(256, 4, 1, PadPolicy::Circular),
        ScatterConfig::one_d(1024, 6, 8, PadPolicy::Circular),
        ScatterConfig::two_d(32, 32, 3, 8, [PadPolicy::Circular; 2], Group::Trivial),
        ScatterConfig::two_d(64, 64, 4, 4, [PadPolicy::Circular; 2], Group::Trivial),
    ]
}

/// Littlewood-Paley bounds `(min, max)` of each bank, recorded in VDS-1
/// section 14.3.5 and asserted to 1e-6. The value 1 is attained at
/// frequency 0, where phi equals 1 and every wavelet vanishes.
const LP_RECORDED: &[(&str, f64, f64)] = &[
    ("1d n=256 J=4 Q=1 psi1", 0.165797, 1.000000),
    ("1d n=256 J=4 Q=1 psi2", 0.165797, 1.000000),
    ("1d n=1024 J=6 Q=1 psi1", 0.165824, 1.000000),
    ("1d n=1024 J=6 Q=1 psi2", 0.165824, 1.000000),
    ("1d n=1024 J=6 Q=8 psi1", 0.000290, 1.000000),
    ("1d n=1024 J=6 Q=8 psi2", 0.165824, 1.000000),
    ("2d 64x64 J=3 L=4 psi", 0.132018, 1.000000),
    ("2d 64x64 J=3 L=8 psi", 0.122981, 1.013572),
    ("2d 128x128 J=4 L=8 psi", 0.122981, 1.046507),
];

#[test]
fn littlewood_paley_bounds() {
    let cases = [
        (
            "1d n=256 J=4 Q=1",
            ScatterConfig::one_d(256, 4, 1, PadPolicy::Circular),
        ),
        (
            "1d n=1024 J=6 Q=1",
            ScatterConfig::one_d(1024, 6, 1, PadPolicy::Circular),
        ),
        (
            "1d n=1024 J=6 Q=8",
            ScatterConfig::one_d(1024, 6, 8, PadPolicy::Circular),
        ),
        (
            "2d 64x64 J=3 L=4",
            ScatterConfig::two_d(64, 64, 3, 4, [PadPolicy::Circular; 2], Group::Trivial),
        ),
        (
            "2d 64x64 J=3 L=8",
            ScatterConfig::two_d(64, 64, 3, 8, [PadPolicy::Circular; 2], Group::Trivial),
        ),
        (
            "2d 128x128 J=4 L=8",
            ScatterConfig::two_d(128, 128, 4, 8, [PadPolicy::Circular; 2], Group::Trivial),
        ),
    ];
    let mut measured = Vec::new();
    for (name, cfg) in cases {
        let sc = Scattering::new(cfg).unwrap();
        let f = sc.filters();
        let banks: Vec<(&str, &[vikshep_scatter::filters::Filter])> = if f.psi2.is_some() {
            vec![("psi1", &f.psi1[..]), ("psi2", f.second_order())]
        } else {
            vec![("psi", &f.psi1[..])]
        };
        for (bank_name, bank) in banks {
            let (lo, hi) = f.littlewood_paley(bank);
            println!("    (\"{name} {bank_name}\", {lo:.6}, {hi:.6}),");
            measured.push((format!("{name} {bank_name}"), lo, hi));
        }
    }
    assert_eq!(measured.len(), LP_RECORDED.len(), "update LP_RECORDED");
    for ((name, lo, hi), (rname, rlo, rhi)) in measured.iter().zip(LP_RECORDED) {
        assert_eq!(name, rname);
        assert!(
            (lo - rlo).abs() <= 1e-6 + 5e-7 && (hi - rhi).abs() <= 1e-6 + 5e-7,
            "{name}: ({lo}, {hi})"
        );
    }
}

/// The imaginary part discarded from 2-D Fourier filters is below 1e-5 of
/// the largest real part (VDS-1 section 14.3.3).
#[test]
fn discarded_imaginary_part_is_small() {
    for (n, j, l) in [(32, 4, 8), (64, 4, 8), (32, 1, 4), (64, 3, 8)] {
        let cfg = ScatterConfig::two_d(n, n, j, l, [PadPolicy::Circular; 2], Group::Trivial);
        let r = Scattering::new(cfg).unwrap().filters().max_imag_ratio;
        println!("{n}x{n} J={j} L={l}: max |imag| / max |real| = {r:.3e}");
        assert!(r <= 1e-5, "{r}");
    }
}

fn l2(v: &[f32]) -> f64 {
    v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>()
}

/// ||Sx - Sy||^2 * 2^(J d) <= (1 + tol) ||x - y||^2: the scattering energy of
/// a difference, corrected for the final 2^J subsampling, does not exceed
/// the input energy (Littlewood-Paley upper bound 1).
#[test]
fn non_expansive_on_random_pairs() {
    let backend = CpuBackend::new();
    for cfg in configs() {
        let sc = Scattering::new(cfg.clone()).unwrap();
        let n = cfg.signal_len();
        let mut worst = 0.0f64;
        for seed in 0..4u64 {
            let x = input(n, 100 + seed);
            let y: Vec<f32> = if seed % 2 == 0 {
                input(n, 200 + seed)
            } else {
                // a small perturbation of x
                x.iter()
                    .zip(input(n, 300 + seed))
                    .map(|(a, e)| a + 0.01 * e)
                    .collect()
            };
            let sx = sc.run(&backend, &x).unwrap().coefficients;
            let sy = sc.run(&backend, &y).unwrap().coefficients;
            let ds: Vec<f32> = sx.iter().zip(&sy).map(|(a, b)| a - b).collect();
            let dx: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a - b).collect();
            let ratio = l2(&ds) * f64::from(1u32 << (cfg.j as usize * cfg.dim)) / l2(&dx);
            worst = worst.max(ratio);
        }
        println!(
            "{:?} J={} Q={} L={}: max energy ratio {worst:.4}",
            cfg.shape, cfg.j, cfg.q, cfg.l
        );
        assert!(worst <= 1.05, "non-expansiveness violated: {worst}");
    }
}

/// On circular axes a shift of the input by 2^J samples shifts every output
/// path by one sample (within binary32 tolerance; the arithmetic paths
/// differ, so the identity is not bitwise).
#[test]
fn circular_shift_by_grid_step() {
    let backend = CpuBackend::new();
    let cfg = ScatterConfig::two_d(32, 32, 2, 4, [PadPolicy::Circular; 2], Group::Trivial);
    let sc = Scattering::new(cfg.clone()).unwrap();
    let (n, step) = (32usize, 4usize);
    let x = input(n * n, 7);
    let mut xs = vec![0.0f32; n * n];
    for r in 0..n {
        for c in 0..n {
            xs[((r + step) % n) * n + (c + step) % n] = x[r * n + c];
        }
    }
    let a = sc.run(&backend, &x).unwrap();
    let b = sc.run(&backend, &xs).unwrap();
    let m = n / step;
    let peak = a.coefficients.iter().fold(0.0f32, |p, v| p.max(v.abs()));
    for p in 0..a.paths.len() {
        let (pa, pb) = (a.path(0, p), b.path(0, p));
        for r in 0..m {
            for c in 0..m {
                let want = pa[r * m + c];
                let got = pb[((r + 1) % m) * m + (c + 1) % m];
                assert!((want - got).abs() <= 1e-5 * peak, "path {p}");
            }
        }
    }
    // 1-D as well
    let cfg = ScatterConfig::one_d(256, 3, 2, PadPolicy::Circular);
    let sc = Scattering::new(cfg).unwrap();
    let x = input(256, 8);
    let xs: Vec<f32> = (0..256).map(|i| x[(i + 256 - 8) % 256]).collect();
    let a = sc.run(&backend, &x).unwrap();
    let b = sc.run(&backend, &xs).unwrap();
    let peak = a.coefficients.iter().fold(0.0f32, |p, v| p.max(v.abs()));
    for p in 0..a.paths.len() {
        for i in 0..32 {
            assert!((a.path(0, p)[i] - b.path(0, p)[(i + 1) % 32]).abs() <= 1e-5 * peak);
        }
    }
}

/// VDS-1.1 (section 8.1): subnormal operands of the Tier-1 kernels are
/// treated as signed zeros and subnormal results are flushed to signed zero;
/// the scattering driver flushes subnormal input samples on the host.
#[test]
fn cpu_flushes_subnormals() {
    let backend = CpuBackend::serial();
    assert!(!backend.capabilities().preserves_subnormals);
    let tiny = f32::from_bits(0x0000_0400); // 2^-139, subnormal
    assert!(tiny.is_subnormal());
    // FFT of a subnormal impulse: the operand is flushed, the spectrum is 0.
    let canvas = Canvas::one_d(16);
    let mut d = vec![Complex32::default(); 16];
    d[0] = Complex32::new(tiny, 0.0);
    let tw = Twiddles {
        cols: f32::twiddles(4),
        rows: &[],
    };
    backend.fft(&mut d, canvas, &tw).unwrap();
    assert!(d.iter().all(|v| v.re == 0.0 && v.im == 0.0));
    // The inverse scaling by 2^-4 of FLT_MIN would be subnormal: flushed.
    let twi = Twiddles {
        cols: f32::twiddles_inverse(4),
        rows: &[],
    };
    let min = f32::MIN_POSITIVE;
    let mut e = vec![Complex32::default(); 16];
    e[0] = Complex32::new(min, -min);
    backend.ifft(&mut e, canvas, &twi).unwrap();
    assert_eq!(e[0].re.to_bits(), 0);
    assert_eq!(e[0].im.to_bits(), 0x8000_0000, "sign of zero preserved");
    // A butterfly difference of two close normal numbers is subnormal (exact
    // by Sterbenz) and is flushed.
    let mut b = [
        Complex32::new(min, 0.0),
        Complex32::new(f32::from_bits(0x0080_0001), 0.0),
    ];
    let tw2 = Twiddles {
        cols: f32::twiddles(1),
        rows: &[],
    };
    backend.fft(&mut b, Canvas::one_d(2), &tw2).unwrap();
    assert!(b[0].re > min);
    // d = -2^-149 flushes to -0; the rotation by TW[0] = (1, -0) then gives
    // -0 * 1 - (+0 * -0) = -0 - (-0) = +0 under IEEE rules.
    assert_eq!(b[1].re.to_bits(), 0);
    // Filter multiplication: FLT_MIN times 0.5 is subnormal: flushed.
    let mut f = vec![Complex32::new(min, -min); 16];
    backend
        .mul_real_filter(&mut f, canvas, &[0.5; 16], &[0])
        .unwrap();
    assert_eq!(f[3].re.to_bits(), 0);
    assert_eq!(f[3].im.to_bits(), 0x8000_0000);
    // The scattering driver flushes subnormal samples before any kernel: a
    // signal of subnormals scatters exactly like zeros.
    let sc = Scattering::new(ScatterConfig::one_d(64, 2, 1, PadPolicy::Circular)).unwrap();
    let subn: Vec<f32> = (0..64u32).map(|i| f32::from_bits(1 + i * 1000)).collect();
    let a = sc.run(&backend, &subn).unwrap();
    let z = sc.run(&backend, &vec![0.0; 64]).unwrap();
    assert_eq!(a.canonical_bytes(), z.canonical_bytes());
}
