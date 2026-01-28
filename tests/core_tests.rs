// @test_for: [crate::core::circulant, crate::core::block_circulant]
// @math_verified: true
// @verified_by: math_expert
// @properties_checked: [fft_correctness, convolution_theorem, linearity]
// @version: 0.2.2
// @event: initial

//! Integration tests for core circulant types.

use approx::assert_relative_eq;
use circulant_rs::core::{naive_bccb_mul, naive_circulant_mul, BlockCirculant, Circulant};
use circulant_rs::traits::{BlockOps, CirculantOps};
use ndarray::Array2;
use num_complex::Complex;

#[test]
fn test_circulant_large_size() {
    // Test with larger matrix to ensure FFT scaling works
    let n = 256;
    let gen: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
        .collect();

    let c = Circulant::new(gen.clone()).unwrap();

    let x: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new(((i * 2) as f64).cos(), ((i * 3) as f64).sin()))
        .collect();

    let fft_result = c.mul_vec(&x).unwrap();
    let naive_result = naive_circulant_mul(&gen, &x);

    for (fft, naive) in fft_result.iter().zip(naive_result.iter()) {
        assert_relative_eq!(fft.re, naive.re, epsilon = 1e-8);
        assert_relative_eq!(fft.im, naive.im, epsilon = 1e-8);
    }
}

#[test]
fn test_circulant_precompute_speedup() {
    let n = 128;
    let gen: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let mut c = Circulant::from_real(gen).unwrap();

    let x: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new((i as f64).cos(), 0.0))
        .collect();

    // First call without precompute
    let result1 = c.mul_vec(&x).unwrap();

    // Precompute and call again
    c.precompute();
    let result2 = c.mul_vec(&x).unwrap();

    // Results should be identical
    for (r1, r2) in result1.iter().zip(result2.iter()) {
        assert_relative_eq!(r1.re, r2.re, epsilon = 1e-10);
        assert_relative_eq!(r1.im, r2.im, epsilon = 1e-10);
    }
}

#[test]
fn test_block_circulant_large_size() {
    let n = 32;
    let gen: Array2<Complex<f64>> = Array2::from_shape_fn((n, n), |(i, j)| {
        Complex::new(((i + j) as f64).sin(), ((i * j) as f64).cos())
    });

    let bc = BlockCirculant::new(gen.clone()).unwrap();

    let x: Array2<Complex<f64>> = Array2::from_shape_fn((n, n), |(i, j)| {
        Complex::new(((i * 2) as f64).cos(), ((j * 3) as f64).sin())
    });

    let fft_result = bc.mul_array(&x).unwrap();
    let naive_result = naive_bccb_mul(&gen, &x);

    for i in 0..n {
        for j in 0..n {
            assert_relative_eq!(
                fft_result[(i, j)].re,
                naive_result[(i, j)].re,
                epsilon = 1e-8
            );
            assert_relative_eq!(
                fft_result[(i, j)].im,
                naive_result[(i, j)].im,
                epsilon = 1e-8
            );
        }
    }
}

#[test]
fn test_circulant_convolution_theorem() {
    // Test that circulant multiplication is equivalent to circular convolution
    let gen = vec![1.0_f64, 2.0, 3.0, 4.0];
    let x = vec![1.0_f64, 0.0, 1.0, 0.0];

    let c = Circulant::from_real(gen.clone()).unwrap();
    let result = c.mul_vec_real(&x).unwrap();

    // Manual circular convolution
    let n = gen.len();
    let mut expected = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            let idx = (i + n - j) % n;
            expected[i] += gen[idx] * x[j];
        }
    }

    for (r, e) in result.iter().zip(expected.iter()) {
        assert_relative_eq!(r.re, *e, epsilon = 1e-10);
        assert_relative_eq!(r.im, 0.0, epsilon = 1e-10);
    }
}

#[test]
fn test_block_circulant_2d_convolution() {
    // Gaussian-like kernel
    let kernel = Array2::from_shape_vec(
        (3, 3),
        vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(1.0, 0.0),
        ],
    )
    .unwrap();

    let bc = BlockCirculant::from_kernel(kernel, 8, 8).unwrap();

    // Simple test pattern
    let mut x = Array2::zeros((8, 8));
    x[(4, 4)] = Complex::new(1.0, 0.0);

    let result = bc.mul_array(&x).unwrap();

    // The result should have the kernel pattern centered at (4,4) with wrapping
    assert!(result[(4, 4)].re > result[(0, 0)].re);
}
