//! Comprehensive scaling benchmark for circulant-rs
//!
//! Run with: cargo bench --bench scaling_benchmark

use circulant_rs::core::{BlockCirculant, Circulant};
use circulant_rs::traits::{BlockOps, CirculantOps};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array2;
use num_complex::Complex;

#[cfg(feature = "physics")]
use circulant_rs::physics::{Coin, CoinedWalk1D, QuantumState, QuantumWalk};

/// Generate random complex vector
fn random_complex_vec(n: usize) -> Vec<Complex<f64>> {
    (0..n)
        .map(|i| {
            let phase = (i as f64) * 0.1;
            Complex::new(phase.cos(), phase.sin())
        })
        .collect()
}

/// Benchmark 1D circulant multiplication across sizes
fn bench_1d_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("1d_circulant_multiply");

    for size in [64, 256, 1024, 4096, 16384, 65536, 262144] {
        group.throughput(Throughput::Elements(size as u64));

        let generator = random_complex_vec(size);
        let x = random_complex_vec(size);
        let circulant = Circulant::new(generator).unwrap();

        // Without precomputation
        group.bench_with_input(BenchmarkId::new("no_precompute", size), &size, |b, _| {
            b.iter(|| black_box(circulant.mul_vec(&x).unwrap()))
        });

        // With precomputation
        let mut circulant_cached = circulant.clone();
        circulant_cached.precompute();

        group.bench_with_input(BenchmarkId::new("precomputed", size), &size, |b, _| {
            b.iter(|| black_box(circulant_cached.mul_vec(&x).unwrap()))
        });
    }

    group.finish();
}

/// Benchmark 1D vs naive O(N²) for small sizes
fn bench_1d_vs_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("1d_fft_vs_naive");

    // Only test small sizes where naive is feasible
    for size in [64, 256, 1024, 2048] {
        let generator = random_complex_vec(size);
        let x = random_complex_vec(size);

        // FFT-based
        let circulant = Circulant::new(generator.clone()).unwrap();
        group.bench_with_input(BenchmarkId::new("fft", size), &size, |b, _| {
            b.iter(|| black_box(circulant.mul_vec(&x).unwrap()))
        });

        // Naive O(N²)
        group.bench_with_input(BenchmarkId::new("naive", size), &size, |b, _| {
            b.iter(|| black_box(naive_multiply(&generator, &x)))
        });
    }

    group.finish();
}

/// Naive O(N²) multiplication for comparison
fn naive_multiply(generator: &[Complex<f64>], x: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let n = generator.len();
    let mut result = vec![Complex::new(0.0, 0.0); n];

    for i in 0..n {
        for j in 0..n {
            let idx = (j + n - i) % n;
            result[i] += generator[idx] * x[j];
        }
    }

    result
}

/// Benchmark 2D BCCB multiplication
fn bench_2d_bccb(c: &mut Criterion) {
    let mut group = c.benchmark_group("2d_bccb_multiply");

    for size in [32, 64, 128, 256, 512] {
        let total_elements = size * size;
        group.throughput(Throughput::Elements(total_elements as u64));

        // Create random generator
        let gen_data: Vec<Complex<f64>> = (0..total_elements)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let generator = Array2::from_shape_vec((size, size), gen_data).unwrap();

        // Create random input
        let x_data: Vec<Complex<f64>> = (0..total_elements)
            .map(|i| Complex::new(1.0 / (i + 1) as f64, 0.0))
            .collect();
        let x = Array2::from_shape_vec((size, size), x_data).unwrap();

        let bccb = BlockCirculant::new(generator).unwrap();

        group.bench_with_input(BenchmarkId::new("bccb", size), &size, |b, _| {
            b.iter(|| black_box(bccb.mul_array(&x).unwrap()))
        });
    }

    group.finish();
}

/// Benchmark quantum walk simulation
#[cfg(feature = "physics")]
fn bench_quantum_walk(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_walk");

    for n_positions in [256, 1024, 4096, 16384, 65536] {
        group.throughput(Throughput::Elements(n_positions as u64));

        let walk = CoinedWalk1D::<f64>::new(n_positions, Coin::Hadamard).unwrap();
        let initial = QuantumState::localized(n_positions / 2, n_positions, 2).unwrap();

        // Single step benchmark
        group.bench_with_input(
            BenchmarkId::new("single_step", n_positions),
            &n_positions,
            |b, _| {
                let mut state = initial.clone();
                b.iter(|| {
                    walk.step(&mut state);
                    black_box(&state);
                })
            },
        );

        // 100 steps benchmark
        group.bench_with_input(
            BenchmarkId::new("100_steps", n_positions),
            &n_positions,
            |b, _| b.iter(|| black_box(walk.simulate(initial.clone(), 100))),
        );
    }

    group.finish();
}

/// Benchmark memory allocation patterns
fn bench_memory_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");

    for size in [1024, 4096, 16384, 65536, 262144, 1048576] {
        group.bench_with_input(BenchmarkId::new("circulant_create", size), &size, |b, _| {
            b.iter(|| {
                let gen = random_complex_vec(size);
                black_box(Circulant::new(gen).unwrap())
            })
        });
    }

    group.finish();
}

#[cfg(feature = "physics")]
criterion_group!(
    benches,
    bench_1d_multiply,
    bench_1d_vs_naive,
    bench_2d_bccb,
    bench_quantum_walk,
    bench_memory_scaling,
);

#[cfg(not(feature = "physics"))]
criterion_group!(
    benches,
    bench_1d_multiply,
    bench_1d_vs_naive,
    bench_2d_bccb,
    bench_memory_scaling,
);

criterion_main!(benches);
