//! Throughput baseline (measurement only, not a product claim): 2-D 64x64,
//! J = 3, L = 8, order 2, circular padding, trivial group, a batch of
//! `VIKSHEP_BENCH_EVENTS` events (default 1000) on the CPU reference.
//!
//!     cargo bench -p vikshep-scatter --bench scatter_2d
//!
//! Criterion reports the time per batch and the throughput in events per
//! second ("elem/s").

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use vikshep_cpu::CpuBackend;
use vikshep_numerics::rng::Stream;
use vikshep_scatter::{Group, PadPolicy, ScatterConfig, Scattering};

fn events() -> usize {
    std::env::var("VIKSHEP_BENCH_EVENTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000)
}

fn bench(c: &mut Criterion) {
    let cfg = ScatterConfig::two_d(64, 64, 3, 8, [PadPolicy::Circular; 2], Group::Trivial);
    let sc = Scattering::new(cfg.clone()).unwrap();
    let n = events();
    let mut s = Stream::new(1, 0);
    let x: Vec<f32> = (0..n * cfg.signal_len())
        .map(|_| s.next_f32_unit() * 2.0 - 1.0)
        .collect();
    let backend = CpuBackend::new();
    let mut g = c.benchmark_group("scatter_2d_64x64_J3_L8_order2");
    g.throughput(Throughput::Elements(n as u64));
    g.sample_size(10);
    g.bench_function(format!("cpu_batch_{n}"), |b| {
        b.iter(|| sc.run(&backend, &x).unwrap());
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
