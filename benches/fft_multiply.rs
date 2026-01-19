//! Benchmarks for FFT-based circulant multiplication.
//!
//! Run with: `cargo bench`

use circulant_rs::core::{naive_circulant_mul, Circulant};
use circulant_rs::traits::CirculantOps;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use num_complex::Complex;

fn generate_data(n: usize) -> (Circulant<f64>, Vec<Complex<f64>>) {
    let gen: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
        .collect();

    let c = Circulant::new(gen).unwrap();

    let x: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new(((i * 2) as f64).cos(), ((i * 3) as f64).sin()))
        .collect();

    (c, x)
}

fn bench_fft_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("circulant_multiply");

    for size in [64, 128, 256, 512, 1024, 2048].iter() {
        let (circulant, x) = generate_data(*size);

        group.bench_with_input(BenchmarkId::new("fft", size), size, |b, _| {
            b.iter(|| {
                black_box(circulant.mul_vec(black_box(&x)).unwrap());
            })
        });
    }

    group.finish();
}

fn bench_naive_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("naive_multiply");

    // Only benchmark smaller sizes for naive (it's O(n²))
    for size in [64, 128, 256].iter() {
        let (circulant, x) = generate_data(*size);
        let gen = circulant.generator().to_vec();

        group.bench_with_input(BenchmarkId::new("naive", size), size, |b, _| {
            b.iter(|| {
                black_box(naive_circulant_mul(black_box(&gen), black_box(&x)));
            })
        });
    }

    group.finish();
}

fn bench_fft_vs_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_vs_naive");

    for size in [64, 128, 256].iter() {
        let (circulant, x) = generate_data(*size);
        let gen = circulant.generator().to_vec();

        group.bench_with_input(BenchmarkId::new("fft", size), size, |b, _| {
            b.iter(|| {
                black_box(circulant.mul_vec(black_box(&x)).unwrap());
            })
        });

        group.bench_with_input(BenchmarkId::new("naive", size), size, |b, _| {
            b.iter(|| {
                black_box(naive_circulant_mul(black_box(&gen), black_box(&x)));
            })
        });
    }

    group.finish();
}

fn bench_precomputed(c: &mut Criterion) {
    let mut group = c.benchmark_group("precomputed");

    for size in [256, 512, 1024].iter() {
        let (mut circulant, x) = generate_data(*size);

        group.bench_with_input(BenchmarkId::new("cold", size), size, |b, _| {
            b.iter(|| {
                black_box(circulant.mul_vec(black_box(&x)).unwrap());
            })
        });

        // Precompute spectrum
        circulant.precompute();

        group.bench_with_input(BenchmarkId::new("precomputed", size), size, |b, _| {
            b.iter(|| {
                black_box(circulant.mul_vec(black_box(&x)).unwrap());
            })
        });
    }

    group.finish();
}

#[cfg(feature = "physics")]
fn bench_quantum_walk(c: &mut Criterion) {
    use circulant_rs::physics::{Coin, CoinedWalk1D, QuantumState, QuantumWalk};

    let mut group = c.benchmark_group("quantum_walk");

    for size in [64, 128, 256, 512].iter() {
        let walk = CoinedWalk1D::<f64>::new(*size, Coin::Hadamard);
        let state = QuantumState::localized(*size / 2, *size, 2).unwrap();

        group.bench_with_input(BenchmarkId::new("single_step", size), size, |b, _| {
            let mut s = state.clone();
            b.iter(|| {
                walk.step(black_box(&mut s));
            })
        });

        group.bench_with_input(BenchmarkId::new("10_steps", size), size, |b, _| {
            b.iter(|| {
                black_box(walk.simulate(state.clone(), 10));
            })
        });
    }

    group.finish();
}

#[cfg(feature = "physics")]
criterion_group!(
    benches,
    bench_fft_multiply,
    bench_naive_multiply,
    bench_fft_vs_naive,
    bench_precomputed,
    bench_quantum_walk
);

#[cfg(not(feature = "physics"))]
criterion_group!(
    benches,
    bench_fft_multiply,
    bench_naive_multiply,
    bench_fft_vs_naive,
    bench_precomputed
);

criterion_main!(benches);
