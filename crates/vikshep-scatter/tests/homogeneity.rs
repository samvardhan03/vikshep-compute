//! EXACT property (VDS-1 section 14.8): for any input x and integer k such
//! that no intermediate value overflows or underflows,
//! r2(2^k x) == r2(x) bit for bit, and S(2^k x) == 2^k S(x) bit for bit.

use vikshep_cpu::CpuBackend;
use vikshep_numerics::rng::Stream;
use vikshep_scatter::reduce::r2;
use vikshep_scatter::{Group, PadPolicy, ScatterConfig, Scattering};

fn input(n: usize, seed: u64) -> Vec<f32> {
    let mut s = Stream::new(seed, 0);
    (0..n).map(|_| s.next_f32_unit() * 2.0 - 1.0).collect()
}

fn pow2(k: i32) -> f32 {
    f32::from_bits(((k + 127) as u32) << 23)
}

fn check(cfg: ScatterConfig, seed: u64) {
    let sc = Scattering::new(cfg.clone()).unwrap();
    let backend = CpuBackend::new();
    let x = input(cfg.signal_len(), seed);
    let base = sc.run(&backend, &x).unwrap();
    let base_r2 = r2(&base, cfg.carrier_cutoff);
    assert!(!base_r2.values.is_empty(), "config has no r2 paths");
    for k in [-6, -3, -1, 1, 2, 5] {
        let scale = pow2(k);
        let xs: Vec<f32> = x.iter().map(|v| v * scale).collect();
        let out = sc.run(&backend, &xs).unwrap();
        for (i, (a, b)) in out.coefficients.iter().zip(&base.coefficients).enumerate() {
            assert_eq!(
                a.to_bits(),
                (b * scale).to_bits(),
                "S not exactly 1-homogeneous: k={k} i={i} {cfg:?}"
            );
        }
        let rr = r2(&out, cfg.carrier_cutoff);
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&rr.values),
            bits(&base_r2.values),
            "r2 changed: k={k} {cfg:?}"
        );
    }
}

#[test]
fn r2_exactly_scale_invariant_1d() {
    check(ScatterConfig::one_d(256, 4, 1, PadPolicy::Circular), 1);
    let mut c = ScatterConfig::one_d(256, 3, 4, PadPolicy::ZeroPad);
    c.carrier_cutoff = 0;
    check(c, 2);
    check(ScatterConfig::one_d(1024, 6, 8, PadPolicy::Circular), 5);
}

#[test]
fn r2_exactly_scale_invariant_2d() {
    check(
        ScatterConfig::two_d(32, 32, 3, 4, [PadPolicy::Circular; 2], Group::Trivial),
        3,
    );
    let mut c = ScatterConfig::two_d(
        32,
        32,
        2,
        8,
        [PadPolicy::ZeroPad, PadPolicy::Circular],
        Group::So2Relative,
    );
    c.carrier_cutoff = 0;
    check(c, 4);
    check(
        ScatterConfig::two_d(64, 64, 4, 8, [PadPolicy::ZeroPad; 2], Group::So2Relative),
        6,
    );
}
