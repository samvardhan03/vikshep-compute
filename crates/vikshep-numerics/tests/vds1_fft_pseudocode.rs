//! Verification of the normative FFT pseudo-code and twiddle-table
//! construction in `spec/VDS-1.md` sections 6 and 7.
//!
//! This is a literal transcription of the specification text, used to prove
//! that the text describes a correct DFT and to pin down where the result
//! lives. It is not the production FFT (milestone C1 adds that, and C1's
//! conformance vectors are checked against it).

use vikshep_detmath::{cos, sin};

#[derive(Clone, Copy, Debug, PartialEq)]
struct C32 {
    re: f32,
    im: f32,
}

/// VDS-1 section 7: TW_N[p] for p in 0..N/2, built from the first octant.
fn twiddles(m: u32) -> Vec<C32> {
    let n = 1usize << m;
    let quarter = n / 4;
    let eighth = n / 8;
    let s = 0.5f32.sqrt();
    // Exact: 2*pi (binary64) times 2^-m, a power-of-two scaling.
    let step = core::f64::consts::TAU * (1.0 / (1u64 << m) as f64);
    // first quadrant Q[k] = (cos a_k, sin a_k), a_k = 2*pi*k/N, k in 0..=N/4
    let mut q = vec![(0.0f32, 0.0f32); quarter + 1];
    for (k, slot) in q.iter_mut().enumerate().take(eighth + 1) {
        let theta = k as f64 * step;
        *slot = (cos(theta) as f32, sin(theta) as f32);
    }
    if n >= 8 {
        q[eighth] = (s, s);
    }
    for k in (eighth + 1)..=quarter {
        let (c, sn) = q[quarter - k];
        q[k] = (sn, c);
    }
    (0..n / 2)
        .map(|p| {
            if p <= quarter {
                C32 {
                    re: q[p].0,
                    im: -q[p].1,
                }
            } else {
                let (c, sn) = q[p - quarter];
                C32 { re: -sn, im: -c }
            }
        })
        .collect()
}

/// VDS-1 section 6, `fft0`, transcribed literally.
#[allow(clippy::too_many_arguments)]
fn fft0(big_n: usize, n: usize, s: usize, eo: bool, x: &mut [C32], y: &mut [C32], tw: &[C32]) {
    if n == 1 {
        if eo {
            y[..s].copy_from_slice(&x[..s]);
        }
        return;
    }
    let m = n / 2;
    for p in 0..m {
        let w = tw[p * (big_n / n)];
        for q in 0..s {
            let a = x[q + s * p];
            let b = x[q + s * (p + m)];
            y[q + s * (2 * p)] = C32 {
                re: a.re + b.re,
                im: a.im + b.im,
            };
            let d = C32 {
                re: a.re - b.re,
                im: a.im - b.im,
            };
            y[q + s * (2 * p + 1)] = C32 {
                re: d.re * w.re - d.im * w.im,
                im: d.re * w.im + d.im * w.re,
            };
        }
    }
    fft0(big_n, n / 2, 2 * s, !eo, y, x, tw);
}

fn conj_table(tw: &[C32]) -> Vec<C32> {
    tw.iter()
        .map(|w| C32 {
            re: w.re,
            im: -w.im,
        })
        .collect()
}

/// Forward transform; returns (result held in x, final contents of scratch y).
fn forward(m: u32, input: &[C32]) -> (Vec<C32>, Vec<C32>) {
    let n = 1usize << m;
    let tw = twiddles(m);
    let mut x = input.to_vec();
    let mut y = vec![
        C32 {
            re: f32::NAN,
            im: f32::NAN
        };
        n
    ];
    fft0(n, n, 1, false, &mut x, &mut y, &tw);
    (x, y)
}

fn inverse(m: u32, input: &[C32]) -> Vec<C32> {
    let n = 1usize << m;
    let tw = conj_table(&twiddles(m));
    let mut x = input.to_vec();
    let mut y = vec![C32 { re: 0.0, im: 0.0 }; n];
    fft0(n, n, 1, false, &mut x, &mut y, &tw);
    let scale = 1.0f32 / (1u32 << m) as f32; // exact 2^-m
    x.iter()
        .map(|v| C32 {
            re: v.re * scale,
            im: v.im * scale,
        })
        .collect()
}

/// Naive binary64 DFT of the (exactly widened) binary32 input.
fn naive_dft(input: &[C32], sign: f64) -> Vec<(f64, f64)> {
    let n = input.len();
    (0..n)
        .map(|k| {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (j, v) in input.iter().enumerate() {
                let theta = sign * core::f64::consts::TAU * ((j * k) % n) as f64 / n as f64;
                let (c, s) = (cos(theta), sin(theta));
                re += f64::from(v.re) * c - f64::from(v.im) * s;
                im += f64::from(v.re) * s + f64::from(v.im) * c;
            }
            (re, im)
        })
        .collect()
}

/// Squared magnitude of (a, b).
fn hypot2(a: f64, b: f64) -> f64 {
    a * a + b * b
}

fn test_signal(m: u32, seed: u64) -> Vec<C32> {
    let mut s = vikshep_numerics::rng::Stream::new(seed, u64::from(m));
    (0..(1usize << m))
        .map(|_| C32 {
            re: s.next_f32_unit() * 2.0 - 1.0,
            im: s.next_f32_unit() * 2.0 - 1.0,
        })
        .collect()
}

#[test]
fn twiddle_special_values_are_exact() {
    let s = 0.5f32.sqrt();
    for m in 1..=12u32 {
        let n = 1usize << m;
        let tw = twiddles(m);
        assert_eq!(tw.len(), n / 2);
        assert_eq!(tw[0], C32 { re: 1.0, im: 0.0 });
        if n >= 4 {
            assert_eq!(tw[n / 4], C32 { re: 0.0, im: -1.0 });
        }
        if n >= 8 {
            assert_eq!(tw[n / 8], C32 { re: s, im: -s });
            assert_eq!(tw[3 * n / 8], C32 { re: -s, im: -s });
        }
    }
}

#[test]
fn twiddles_are_close_to_direct_evaluation() {
    for m in 1..=12u32 {
        let n = 1usize << m;
        for (p, w) in twiddles(m).iter().enumerate() {
            let theta = core::f64::consts::TAU * p as f64 / n as f64;
            assert!((f64::from(w.re) - cos(theta)).abs() <= 6e-8, "N={n} p={p}");
            assert!((f64::from(w.im) + sin(theta)).abs() <= 6e-8, "N={n} p={p}");
        }
    }
}

#[test]
fn pseudocode_is_a_correct_dft_for_all_sizes() {
    for m in 1..=12u32 {
        let n = 1usize << m;
        let input = test_signal(m, 11);
        let (got, scratch) = forward(m, &input);
        let want = naive_dft(&input, -1.0);
        // Error bound for radix-2 in binary32: O(log2 N * eps * ||x||_2 * sqrt N)
        let norm = input
            .iter()
            .map(|v| hypot2(f64::from(v.re), f64::from(v.im)))
            .sum::<f64>()
            .sqrt();
        let tol = 4.0 * f64::from(m) * f64::from(f32::EPSILON) * norm;
        for k in 0..n {
            let (re, im) = want[k];
            let err = hypot2(f64::from(got[k].re) - re, f64::from(got[k].im) - im).sqrt();
            assert!(err <= tol, "N={n} k={k} err={err} tol={tol}");
        }
        // Result location: always the input array x. For odd m the final
        // n == 1 step (eo == true) copies the result from the scratch array
        // into x, so both arrays hold it; for even m no copy happens and the
        // scratch array holds the output of the second-to-last stage.
        if m % 2 == 1 {
            assert_eq!(scratch, got, "odd m: scratch must equal the result");
        } else {
            assert_ne!(scratch, got, "even m: scratch holds an intermediate stage");
        }
    }
}

#[test]
fn inverse_recovers_input() {
    for m in 1..=12u32 {
        let input = test_signal(m, 23);
        let (spectrum, _) = forward(m, &input);
        let back = inverse(m, &spectrum);
        for (a, b) in input.iter().zip(&back) {
            assert!((a.re - b.re).abs() <= 1e-5 && (a.im - b.im).abs() <= 1e-5);
        }
        // Inverse against the naive inverse DFT (sign +1, scaled by 1/N).
        let want = naive_dft(&spectrum, 1.0);
        let n = (1usize << m) as f64;
        for (k, v) in back.iter().enumerate() {
            assert!((f64::from(v.re) - want[k].0 / n).abs() <= 1e-5);
            assert!((f64::from(v.im) - want[k].1 / n).abs() <= 1e-5);
        }
    }
}

#[test]
fn delta_and_constant_are_exact() {
    for m in 1..=12u32 {
        let n = 1usize << m;
        let mut delta = vec![C32 { re: 0.0, im: 0.0 }; n];
        delta[0].re = 1.0;
        let (spec, _) = forward(m, &delta);
        assert!(spec.iter().all(|v| *v == C32 { re: 1.0, im: 0.0 }));
        let ones = vec![C32 { re: 1.0, im: 0.0 }; n];
        let (spec, _) = forward(m, &ones);
        assert_eq!(
            spec[0],
            C32 {
                re: n as f32,
                im: 0.0
            }
        );
        assert!(spec[1..].iter().all(|v| v.re == 0.0 && v.im == 0.0));
    }
}

// ---------------------------------------------------------------------------
// The production FFT (`vikshep_numerics::fft`) must equal the literal
// transcription above bit for bit.

use vikshep_numerics::fft::{self, Complex32, Direction, Real};

fn to_c(v: &[C32]) -> Vec<Complex32> {
    v.iter().map(|c| Complex32::new(c.re, c.im)).collect()
}

fn bits(v: &[Complex32]) -> Vec<(u32, u32)> {
    v.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
}

#[test]
fn production_twiddles_equal_literal_construction() {
    for m in 1..=12u32 {
        assert_eq!(bits(f32::twiddles(m)), bits(&to_c(&twiddles(m))), "m={m}");
    }
}

#[test]
fn production_fft_equals_literal_recursion_bitwise() {
    for m in 1..=12u32 {
        let n = 1usize << m;
        let input = test_signal(m, 101);
        let (want, _) = forward(m, &input);
        let mut got = to_c(&input);
        let mut scratch = vec![Complex32::default(); n];
        fft::fft_1d(&mut got, &mut scratch, Direction::Forward);
        assert_eq!(bits(&got), bits(&to_c(&want)), "forward m={m}");

        let want_inv = inverse(m, &want);
        let mut got_inv = to_c(&want);
        fft::fft_1d(&mut got_inv, &mut scratch, Direction::Inverse);
        assert_eq!(bits(&got_inv), bits(&to_c(&want_inv)), "inverse m={m}");
    }
}
