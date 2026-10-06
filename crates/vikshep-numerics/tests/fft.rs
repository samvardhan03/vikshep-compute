//! Correctness of the production FFT (VDS-1 section 6) against a naive
//! binary64 DFT, round trips, 2-D separability and table special values.

use vikshep_detmath::{cos, sin};
use vikshep_numerics::fft::{self, Complex32, Complex64, Direction, Real};
use vikshep_numerics::rng::Stream;

fn signal(n: usize, seed: u64) -> Vec<Complex32> {
    let mut s = Stream::new(seed, n as u64);
    (0..n)
        .map(|_| Complex32::new(s.next_f32_unit() * 2.0 - 1.0, s.next_f32_unit() * 2.0 - 1.0))
        .collect()
}

fn naive(x: &[Complex32], sign: f64) -> Vec<Complex64> {
    let n = x.len();
    (0..n)
        .map(|k| {
            let mut acc = Complex64::new(0.0, 0.0);
            for (j, v) in x.iter().enumerate() {
                let t = sign * core::f64::consts::TAU * ((j * k) % n) as f64 / n as f64;
                let (c, s) = (cos(t), sin(t));
                let (re, im) = (f64::from(v.re), f64::from(v.im));
                acc.re += re * c - im * s;
                acc.im += re * s + im * c;
            }
            acc
        })
        .collect()
}

fn norm2(x: &[Complex32]) -> f64 {
    x.iter()
        .map(|v| f64::from(v.re) * f64::from(v.re) + f64::from(v.im) * f64::from(v.im))
        .sum::<f64>()
        .sqrt()
}

/// (a) every N = 2..4096 against the naive DFT; tolerance
/// 4 * log2(N) * eps32 * ||x||_2 on every output element.
#[test]
fn forward_matches_naive_dft_for_all_sizes() {
    for m in 1..=12u32 {
        let n = 1usize << m;
        let x = signal(n, 7);
        let want = naive(&x, -1.0);
        let mut got = x.clone();
        fft::fft_batch(&mut got, n, Direction::Forward);
        let tol = 4.0 * f64::from(m) * f64::from(f32::EPSILON) * norm2(&x);
        for (k, (g, w)) in got.iter().zip(&want).enumerate() {
            let dr = f64::from(g.re) - w.re;
            let di = f64::from(g.im) - w.im;
            let err = (dr * dr + di * di).sqrt();
            assert!(err <= tol, "N={n} k={k} err={err:e} tol={tol:e}");
        }
    }
}

/// (b) inverse(forward(x)) within 4 * log2(N) * eps32 * max|x| per element.
#[test]
fn round_trip_recovers_input() {
    for m in 1..=12u32 {
        let n = 1usize << m;
        let x = signal(n, 9);
        let mut y = x.clone();
        fft::fft_batch(&mut y, n, Direction::Forward);
        fft::fft_batch(&mut y, n, Direction::Inverse);
        let tol = 4.0 * f64::from(m) * f64::from(f32::EPSILON) * 2.0;
        for (a, b) in x.iter().zip(&y) {
            assert!(f64::from((a.re - b.re).abs()) <= tol, "N={n}");
            assert!(f64::from((a.im - b.im).abs()) <= tol, "N={n}");
        }
    }
}

/// (c) exact table values.
#[test]
fn table_special_values_are_exact() {
    let s = 0.5f32.sqrt();
    assert_eq!(s.to_bits(), 0x3F35_04F3);
    for m in 1..=12u32 {
        let n = 1usize << m;
        let tw = f32::twiddles(m);
        assert_eq!(tw[0], Complex32::new(1.0, 0.0));
        if n >= 4 {
            assert_eq!(tw[n / 4], Complex32::new(0.0, -1.0));
        }
        if n >= 8 {
            assert_eq!(tw[n / 8], Complex32::new(s, -s));
            assert_eq!(tw[3 * n / 8], Complex32::new(-s, -s));
        }
        let inv = f32::twiddles_inverse(m);
        for (a, b) in tw.iter().zip(inv) {
            assert_eq!(a.re.to_bits(), b.re.to_bits());
            assert_eq!(a.im.to_bits(), (-b.im).to_bits());
        }
    }
}

/// 2-D = rows then columns, checked against separable naive DFTs, for square
/// and rectangular shapes including a single row.
#[test]
fn two_dimensional_transform() {
    for (rows, cols) in [(1, 16), (8, 8), (4, 32), (32, 2), (16, 64)] {
        let x = signal(rows * cols, 13);
        let mut got = x.clone();
        fft::fft_2d(&mut got, rows, cols, Direction::Forward);
        // naive: rows then columns in binary64
        let mut tmp = vec![Complex64::new(0.0, 0.0); rows * cols];
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let t = if cols > 1 {
                naive(row, -1.0)
            } else {
                vec![Complex64::new(f64::from(row[0].re), f64::from(row[0].im))]
            };
            tmp[r * cols..(r + 1) * cols].copy_from_slice(&t);
        }
        let mut want = tmp.clone();
        if rows > 1 {
            for c in 0..cols {
                for k in 0..rows {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for j in 0..rows {
                        let t = -core::f64::consts::TAU * ((j * k) % rows) as f64 / rows as f64;
                        let v = tmp[j * cols + c];
                        acc.re += v.re * cos(t) - v.im * sin(t);
                        acc.im += v.re * sin(t) + v.im * cos(t);
                    }
                    want[k * cols + c] = acc;
                }
            }
        }
        let tol = 8.0 * 12.0 * f64::from(f32::EPSILON) * norm2(&x);
        for (g, w) in got.iter().zip(&want) {
            assert!((f64::from(g.re) - w.re).abs() <= tol, "{rows}x{cols}");
            assert!((f64::from(g.im) - w.im).abs() <= tol, "{rows}x{cols}");
        }
        let mut back = got.clone();
        fft::fft_2d(&mut back, rows, cols, Direction::Inverse);
        for (a, b) in x.iter().zip(&back) {
            assert!((a.re - b.re).abs() <= 1e-5 && (a.im - b.im).abs() <= 1e-5);
        }
    }
}

/// The 2-D transform equals explicit row transforms followed by explicit
/// column transforms, bit for bit (the transpose is pure data movement).
#[test]
fn two_dimensional_is_rows_then_columns_bitwise() {
    let (rows, cols) = (16, 32);
    let x = signal(rows * cols, 21);
    let mut got = x.clone();
    fft::fft_2d(&mut got, rows, cols, Direction::Forward);
    let mut want = x;
    fft::fft_batch(&mut want, cols, Direction::Forward);
    for c in 0..cols {
        let mut col: Vec<Complex32> = (0..rows).map(|r| want[r * cols + c]).collect();
        fft::fft_batch(&mut col, rows, Direction::Forward);
        for r in 0..rows {
            want[r * cols + c] = col[r];
        }
    }
    let b = |v: &[Complex32]| {
        v.iter()
            .map(|c| (c.re.to_bits(), c.im.to_bits()))
            .collect::<Vec<_>>()
    };
    assert_eq!(b(&got), b(&want));
}

/// Inverse scaling by 2^-m is exact: the inverse of a spectrum that is a
/// multiple of N returns exactly representable values.
#[test]
fn inverse_of_delta_spectrum_is_exact_constant() {
    for m in 1..=12u32 {
        let n = 1usize << m;
        let mut x = vec![Complex32::default(); n];
        x[0] = Complex32::new(n as f32, 0.0);
        fft::fft_batch(&mut x, n, Direction::Inverse);
        assert!(x.iter().all(|v| *v == Complex32::new(1.0, 0.0)), "N={n}");
    }
}

/// VDS-1.1: the per-stage fast path of the Stockham FFT (no result flushes
/// when the operand bounds hold) is bit-identical to flushing every result
/// in every stage, on inputs spanning every magnitude range, including the
/// fast-path thresholds and the subnormal range.
#[test]
fn fast_path_equals_always_flushed() {
    use vikshep_numerics::fft::{Real, stockham, stockham_always_flush};
    let mut s = vikshep_numerics::rng::Stream::new(0x1111, 7);
    let mut cases = 0;
    for m in 1..=12u32 {
        let n = 1usize << m;
        for scale_log2 in [
            0, -40, -60, -68, -69, -70, -75, -90, -100, -110, -120, -126, -130, -140,
        ] {
            for mix in 0..3 {
                let x: Vec<Complex32> = (0..n)
                    .map(|i| {
                        let mut v = || {
                            let u = 2.0 * s.next_f32_unit() - 1.0;
                            let e = match mix {
                                0 => scale_log2,
                                // mixed magnitudes: some large, some tiny
                                1 => {
                                    if i % 5 == 0 {
                                        0
                                    } else {
                                        scale_log2
                                    }
                                }
                                // values packed near FLT_MIN and 2^-69
                                _ => {
                                    if i % 2 == 0 {
                                        -126
                                    } else {
                                        -69
                                    }
                                }
                            };
                            let b = (u.to_bits() as i64 + (i64::from(e) << 23))
                                .clamp(0, i64::from(u32::MAX))
                                as u32;
                            if e == 0 {
                                u
                            } else {
                                f32::from_bits((u.to_bits() & 0x8000_0000) | (b & 0x7fff_ffff))
                            }
                        };
                        Complex32::new(v(), v())
                    })
                    .collect();
                for table in [f32::twiddles(m), f32::twiddles_inverse(m)] {
                    let mut a = x.clone();
                    let mut b = x.clone();
                    let mut sa = vec![Complex32::default(); n];
                    let mut sb = vec![Complex32::default(); n];
                    stockham(&mut a, &mut sa, table);
                    stockham_always_flush(&mut b, &mut sb, table);
                    let bits = |v: &[Complex32]| {
                        v.iter()
                            .map(|z| (z.re.to_bits(), z.im.to_bits()))
                            .collect::<Vec<_>>()
                    };
                    assert_eq!(bits(&a), bits(&b), "n {n} scale {scale_log2} mix {mix}");
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 12 * 14 * 3 * 2);
}
