//! Benchmarks for N-dimensional CirculantTensor operations.
//!
//! Run with: `cargo bench --bench tensor_benchmark`
//!
//! @event: DE-2026-002

use circulant_rs::core::{naive_tensor_mul, CirculantTensor};
use circulant_rs::TensorOps;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{ArrayD, IxDyn};
use num_complex::Complex;

fn complex(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

/// Generate test data for 2D tensor operations.
fn generate_2d_data(n: usize) -> (CirculantTensor<f64, 2>, ArrayD<Complex<f64>>) {
    let total = n * n;
    let gen = ArrayD::from_shape_vec(
        IxDyn(&[n, n]),
        (0..total)
            .map(|i| complex((i as f64).sin(), (i as f64).cos()))
            .collect(),
    )
    .expect("valid shape");

    let tensor = CirculantTensor::<f64, 2>::new(gen).expect("valid tensor");

    let x = ArrayD::from_shape_vec(
        IxDyn(&[n, n]),
        (0..total)
            .map(|i| complex(((i * 2) as f64).cos(), ((i * 3) as f64).sin()))
            .collect(),
    )
    .expect("valid shape");

    (tensor, x)
}

/// Generate test data for 3D tensor operations.
fn generate_3d_data(n: usize) -> (CirculantTensor<f64, 3>, ArrayD<Complex<f64>>) {
    let total = n * n * n;
    let gen = ArrayD::from_shape_vec(
        IxDyn(&[n, n, n]),
        (0..total)
            .map(|i| complex((i as f64 * 0.1).sin(), (i as f64 * 0.1).cos()))
            .collect(),
    )
    .expect("valid shape");

    let tensor = CirculantTensor::<f64, 3>::new(gen).expect("valid tensor");

    let x = ArrayD::from_shape_vec(
        IxDyn(&[n, n, n]),
        (0..total)
            .map(|i| complex(((i * 2) as f64 * 0.1).cos(), ((i * 3) as f64 * 0.1).sin()))
            .collect(),
    )
    .expect("valid shape");

    (tensor, x)
}

/// Generate test data for 4D tensor operations.
fn generate_4d_data(n: usize) -> (CirculantTensor<f64, 4>, ArrayD<Complex<f64>>) {
    let total = n * n * n * n;
    let gen = ArrayD::from_shape_vec(
        IxDyn(&[n, n, n, n]),
        (0..total)
            .map(|i| complex((i as f64 * 0.01).sin(), (i as f64 * 0.01).cos()))
            .collect(),
    )
    .expect("valid shape");

    let tensor = CirculantTensor::<f64, 4>::new(gen).expect("valid tensor");

    let x = ArrayD::from_shape_vec(
        IxDyn(&[n, n, n, n]),
        (0..total)
            .map(|i| complex(((i * 2) as f64 * 0.01).cos(), ((i * 3) as f64 * 0.01).sin()))
            .collect(),
    )
    .expect("valid shape");

    (tensor, x)
}

/// Benchmark 2D tensor multiplication.
fn bench_2d_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_2d_multiply");

    for size in [16, 32, 64, 128, 256].iter() {
        let (tensor, x) = generate_2d_data(*size);

        group.bench_with_input(BenchmarkId::new("fft", size), size, |b, _| {
            b.iter(|| {
                black_box(tensor.mul_tensor(black_box(&x)).unwrap());
            })
        });
    }

    group.finish();
}

/// Benchmark 3D tensor multiplication.
fn bench_3d_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_3d_multiply");

    for size in [8, 16, 32, 64].iter() {
        let (tensor, x) = generate_3d_data(*size);

        group.bench_with_input(BenchmarkId::new("fft", size), size, |b, _| {
            b.iter(|| {
                black_box(tensor.mul_tensor(black_box(&x)).unwrap());
            })
        });
    }

    group.finish();
}

/// Benchmark 4D tensor multiplication.
fn bench_4d_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_4d_multiply");

    for size in [4, 8, 12, 16].iter() {
        let (tensor, x) = generate_4d_data(*size);

        group.bench_with_input(BenchmarkId::new("fft", size), size, |b, _| {
            b.iter(|| {
                black_box(tensor.mul_tensor(black_box(&x)).unwrap());
            })
        });
    }

    group.finish();
}

/// Benchmark FFT vs naive for 2D (smaller sizes).
fn bench_2d_fft_vs_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_2d_fft_vs_naive");

    for size in [8, 16, 24].iter() {
        let (tensor, x) = generate_2d_data(*size);
        let gen = tensor.generator().clone();

        group.bench_with_input(BenchmarkId::new("fft", size), size, |b, _| {
            b.iter(|| {
                black_box(tensor.mul_tensor(black_box(&x)).unwrap());
            })
        });

        group.bench_with_input(BenchmarkId::new("naive", size), size, |b, _| {
            b.iter(|| {
                black_box(naive_tensor_mul(black_box(&gen), black_box(&x)));
            })
        });
    }

    group.finish();
}

/// Benchmark FFT vs naive for 3D (smaller sizes).
fn bench_3d_fft_vs_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_3d_fft_vs_naive");

    for size in [4, 6, 8].iter() {
        let (tensor, x) = generate_3d_data(*size);
        let gen = tensor.generator().clone();

        group.bench_with_input(BenchmarkId::new("fft", size), size, |b, _| {
            b.iter(|| {
                black_box(tensor.mul_tensor(black_box(&x)).unwrap());
            })
        });

        group.bench_with_input(BenchmarkId::new("naive", size), size, |b, _| {
            b.iter(|| {
                black_box(naive_tensor_mul(black_box(&gen), black_box(&x)));
            })
        });
    }

    group.finish();
}

/// Benchmark precomputation benefit.
fn bench_precompute(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_precompute");

    for size in [32, 64, 128].iter() {
        let (mut tensor, x) = generate_2d_data(*size);

        group.bench_with_input(BenchmarkId::new("cold", size), size, |b, _| {
            b.iter(|| {
                black_box(tensor.mul_tensor(black_box(&x)).unwrap());
            })
        });

        tensor.precompute();

        group.bench_with_input(BenchmarkId::new("precomputed", size), size, |b, _| {
            b.iter(|| {
                black_box(tensor.mul_tensor(black_box(&x)).unwrap());
            })
        });
    }

    group.finish();
}

/// Benchmark 3D precomputation.
fn bench_3d_precompute(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_3d_precompute");

    for size in [16, 32, 64].iter() {
        let (mut tensor, x) = generate_3d_data(*size);

        group.bench_with_input(BenchmarkId::new("cold", size), size, |b, _| {
            b.iter(|| {
                black_box(tensor.mul_tensor(black_box(&x)).unwrap());
            })
        });

        tensor.precompute();

        group.bench_with_input(BenchmarkId::new("precomputed", size), size, |b, _| {
            b.iter(|| {
                black_box(tensor.mul_tensor(black_box(&x)).unwrap());
            })
        });
    }

    group.finish();
}

/// Benchmark parallel vs sequential (requires parallel feature).
#[cfg(feature = "parallel")]
fn bench_parallel_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_3d_parallel");

    for size in [32, 48, 64].iter() {
        let (tensor, x) = generate_3d_data(*size);

        group.bench_with_input(BenchmarkId::new("sequential", size), size, |b, _| {
            b.iter(|| {
                black_box(tensor.mul_tensor(black_box(&x)).unwrap());
            })
        });

        group.bench_with_input(BenchmarkId::new("parallel", size), size, |b, _| {
            b.iter(|| {
                black_box(tensor.mul_tensor_parallel(black_box(&x)).unwrap());
            })
        });

        group.bench_with_input(BenchmarkId::new("auto", size), size, |b, _| {
            b.iter(|| {
                black_box(tensor.mul_tensor_auto(black_box(&x)).unwrap());
            })
        });
    }

    group.finish();
}

#[cfg(feature = "parallel")]
criterion_group!(
    benches,
    bench_2d_multiply,
    bench_3d_multiply,
    bench_4d_multiply,
    bench_2d_fft_vs_naive,
    bench_3d_fft_vs_naive,
    bench_precompute,
    bench_3d_precompute,
    bench_parallel_3d
);

#[cfg(not(feature = "parallel"))]
criterion_group!(
    benches,
    bench_2d_multiply,
    bench_3d_multiply,
    bench_4d_multiply,
    bench_2d_fft_vs_naive,
    bench_3d_fft_vs_naive,
    bench_precompute,
    bench_3d_precompute
);

criterion_main!(benches);
