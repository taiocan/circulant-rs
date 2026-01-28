//! Example: Generic N-Dimensional Circulant Tensor Operations
//!
//! This example demonstrates using CirculantTensor for operations
//! across different dimensions (1D, 2D, 3D, 4D) with a unified API.
//!
//! Run with: `cargo run --example tensor_nd`
//!
//! @event: DE-2026-002

use circulant_rs::{Circulant1D, Circulant2D, Circulant3D, Circulant4D, CirculantTensor, TensorOps};
use ndarray::{ArrayD, IxDyn};
use num_complex::Complex;

fn complex(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

/// Demonstrate 1D circulant tensor (equivalent to Circulant<T>).
fn demo_1d() {
    println!("--- 1D Circulant Tensor ---");

    // Create a simple averaging kernel: [0.25, 0.5, 0.25]
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[8]),
        vec![
            complex(0.5, 0.0),
            complex(0.25, 0.0),
            complex(0.0, 0.0),
            complex(0.0, 0.0),
            complex(0.0, 0.0),
            complex(0.0, 0.0),
            complex(0.0, 0.0),
            complex(0.25, 0.0), // Wraps to index -1
        ],
    )
    .expect("valid shape");

    let tensor: Circulant1D<f64> = CirculantTensor::new(kernel).expect("valid tensor");

    // Create input signal with a spike
    let mut signal = ArrayD::zeros(IxDyn(&[8]));
    signal[IxDyn(&[4])] = complex(1.0, 0.0);

    let smoothed = tensor.mul_tensor(&signal).expect("multiply failed");

    println!("Shape: {:?}", tensor.shape());
    println!("Input:  {:?}", signal.iter().map(|c| c.re).collect::<Vec<_>>());
    println!(
        "Output: {:?}",
        smoothed.iter().map(|c| format!("{:.2}", c.re)).collect::<Vec<_>>()
    );
    println!("Eigenvalue count: {}\n", tensor.eigenvalues_nd().len());
}

/// Demonstrate 2D circulant tensor (equivalent to BlockCirculant<T>).
fn demo_2d() {
    println!("--- 2D Circulant Tensor ---");

    // Create a simple 2D averaging kernel
    let n = 8;
    let mut kernel = ArrayD::zeros(IxDyn(&[n, n]));
    kernel[IxDyn(&[0, 0])] = complex(0.25, 0.0);
    kernel[IxDyn(&[0, 1])] = complex(0.125, 0.0);
    kernel[IxDyn(&[1, 0])] = complex(0.125, 0.0);
    kernel[IxDyn(&[0, n - 1])] = complex(0.125, 0.0);
    kernel[IxDyn(&[n - 1, 0])] = complex(0.125, 0.0);
    kernel[IxDyn(&[1, 1])] = complex(0.0625, 0.0);
    kernel[IxDyn(&[1, n - 1])] = complex(0.0625, 0.0);
    kernel[IxDyn(&[n - 1, 1])] = complex(0.0625, 0.0);
    kernel[IxDyn(&[n - 1, n - 1])] = complex(0.0625, 0.0);

    let tensor: Circulant2D<f64> = CirculantTensor::new(kernel).expect("valid tensor");

    // Create input with a point source
    let mut image = ArrayD::zeros(IxDyn(&[n, n]));
    image[IxDyn(&[4, 4])] = complex(1.0, 0.0);

    let blurred = tensor.mul_tensor(&image).expect("multiply failed");

    println!("Shape: {:?}", tensor.shape());
    println!("Total size: {}", tensor.total_size());
    println!("Center before: {:.4}", image[IxDyn(&[4, 4])].re);
    println!("Center after: {:.4}", blurred[IxDyn(&[4, 4])].re);
    println!("Neighbor after: {:.4}", blurred[IxDyn(&[4, 5])].re);
    println!("Eigenvalue count: {}\n", tensor.eigenvalues_nd().len());
}

/// Demonstrate 3D circulant tensor.
fn demo_3d() {
    println!("--- 3D Circulant Tensor ---");

    // Create a simple 3D averaging kernel
    let n = 8;
    let mut kernel = ArrayD::zeros(IxDyn(&[n, n, n]));

    // Center weight
    kernel[IxDyn(&[0, 0, 0])] = complex(0.25, 0.0);

    // Face neighbors (6 faces)
    let face_weight = complex(0.125 / 6.0, 0.0);
    kernel[IxDyn(&[1, 0, 0])] = face_weight;
    kernel[IxDyn(&[n - 1, 0, 0])] = face_weight;
    kernel[IxDyn(&[0, 1, 0])] = face_weight;
    kernel[IxDyn(&[0, n - 1, 0])] = face_weight;
    kernel[IxDyn(&[0, 0, 1])] = face_weight;
    kernel[IxDyn(&[0, 0, n - 1])] = face_weight;

    let tensor: Circulant3D<f64> = CirculantTensor::new(kernel).expect("valid tensor");

    // Create input with a point source
    let mut volume = ArrayD::zeros(IxDyn(&[n, n, n]));
    volume[IxDyn(&[4, 4, 4])] = complex(1.0, 0.0);

    let blurred = tensor.mul_tensor(&volume).expect("multiply failed");

    println!("Shape: {:?}", tensor.shape());
    println!("Total size: {}", tensor.total_size());
    println!("Center before: {:.4}", volume[IxDyn(&[4, 4, 4])].re);
    println!("Center after: {:.4}", blurred[IxDyn(&[4, 4, 4])].re);
    println!("Face neighbor after: {:.4}", blurred[IxDyn(&[5, 4, 4])].re);
    println!("Eigenvalue count: {}\n", tensor.eigenvalues_nd().len());
}

/// Demonstrate 4D circulant tensor.
fn demo_4d() {
    println!("--- 4D Circulant Tensor ---");

    // Create a simple 4D averaging kernel
    let n = 4;
    let mut kernel = ArrayD::zeros(IxDyn(&[n, n, n, n]));

    // Center weight only for simplicity
    kernel[IxDyn(&[0, 0, 0, 0])] = complex(0.5, 0.0);

    // Immediate neighbors in each dimension
    let neighbor_weight = complex(0.5 / 8.0, 0.0);
    kernel[IxDyn(&[1, 0, 0, 0])] = neighbor_weight;
    kernel[IxDyn(&[n - 1, 0, 0, 0])] = neighbor_weight;
    kernel[IxDyn(&[0, 1, 0, 0])] = neighbor_weight;
    kernel[IxDyn(&[0, n - 1, 0, 0])] = neighbor_weight;
    kernel[IxDyn(&[0, 0, 1, 0])] = neighbor_weight;
    kernel[IxDyn(&[0, 0, n - 1, 0])] = neighbor_weight;
    kernel[IxDyn(&[0, 0, 0, 1])] = neighbor_weight;
    kernel[IxDyn(&[0, 0, 0, n - 1])] = neighbor_weight;

    let tensor: Circulant4D<f64> = CirculantTensor::new(kernel).expect("valid tensor");

    // Create input with a point source
    let mut hypervol = ArrayD::zeros(IxDyn(&[n, n, n, n]));
    hypervol[IxDyn(&[2, 2, 2, 2])] = complex(1.0, 0.0);

    let blurred = tensor.mul_tensor(&hypervol).expect("multiply failed");

    println!("Shape: {:?}", tensor.shape());
    println!("Total size: {} ({}^4)", tensor.total_size(), n);
    println!("Center before: {:.4}", hypervol[IxDyn(&[2, 2, 2, 2])].re);
    println!("Center after: {:.4}", blurred[IxDyn(&[2, 2, 2, 2])].re);
    println!("Eigenvalue count: {}\n", tensor.eigenvalues_nd().len());
}

/// Demonstrate linearity property: C(ax + by) = aC(x) + bC(y).
fn demo_linearity() {
    println!("--- Linearity Property Verification ---");

    let n = 8;
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[n, n]),
        (0..n * n).map(|i| complex((i as f64 * 0.1).sin(), 0.0)).collect(),
    )
    .expect("valid shape");

    let tensor: Circulant2D<f64> = CirculantTensor::new(kernel).expect("valid tensor");

    // Create two random-ish inputs
    let x = ArrayD::from_shape_vec(
        IxDyn(&[n, n]),
        (0..n * n).map(|i| complex(i as f64, 0.0)).collect(),
    )
    .expect("valid shape");

    let y = ArrayD::from_shape_vec(
        IxDyn(&[n, n]),
        (0..n * n).map(|i| complex((i + 1) as f64 * 0.5, 0.0)).collect(),
    )
    .expect("valid shape");

    let alpha = complex(2.0, 0.0);
    let beta = complex(0.5, 0.0);

    // Compute C(αx + βy)
    let ax: ArrayD<Complex<f64>> = x.mapv(|v| alpha * v);
    let by: ArrayD<Complex<f64>> = y.mapv(|v| beta * v);
    let sum: ArrayD<Complex<f64>> = &ax + &by;
    let c_sum = tensor.mul_tensor(&sum).expect("multiply failed");

    // Compute αC(x) + βC(y)
    let c_x = tensor.mul_tensor(&x).expect("multiply failed");
    let c_y = tensor.mul_tensor(&y).expect("multiply failed");
    let ac_x: ArrayD<Complex<f64>> = c_x.mapv(|v| alpha * v);
    let bc_y: ArrayD<Complex<f64>> = c_y.mapv(|v| beta * v);
    let expected: ArrayD<Complex<f64>> = &ac_x + &bc_y;

    // Check error
    let max_error: f64 = c_sum
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0_f64, f64::max);

    println!("Testing: C({}*x + {}*y) = {}*C(x) + {}*C(y)", alpha.re, beta.re, alpha.re, beta.re);
    println!("Maximum error: {:.2e}", max_error);
    println!("Linearity verified: {}\n", max_error < 1e-10);
}

/// Demonstrate precomputation benefit.
fn demo_precompute() {
    println!("--- Precomputation Demo ---");

    let n = 32;
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[n, n]),
        (0..n * n).map(|i| complex((i as f64 * 0.01).cos(), 0.0)).collect(),
    )
    .expect("valid shape");

    let input = ArrayD::from_shape_vec(
        IxDyn(&[n, n]),
        (0..n * n).map(|i| complex(i as f64, 0.0)).collect(),
    )
    .expect("valid shape");

    // Without precomputation
    let tensor_cold = CirculantTensor::<f64, 2>::new(kernel.clone()).expect("valid tensor");
    println!("Tensor is precomputed: {}", tensor_cold.is_precomputed());

    // With precomputation
    let mut tensor_warm = CirculantTensor::<f64, 2>::new(kernel).expect("valid tensor");
    tensor_warm.precompute();
    println!("Tensor is precomputed: {}", tensor_warm.is_precomputed());

    // Both should give same result
    let result_cold = tensor_cold.mul_tensor(&input).expect("multiply failed");
    let result_warm = tensor_warm.mul_tensor(&input).expect("multiply failed");

    let max_diff: f64 = result_cold
        .iter()
        .zip(result_warm.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0_f64, f64::max);

    println!("Results match: {} (diff = {:.2e})\n", max_diff < 1e-14, max_diff);
}

fn main() {
    println!("=== N-Dimensional Circulant Tensor Examples ===\n");

    demo_1d();
    demo_2d();
    demo_3d();
    demo_4d();
    demo_linearity();
    demo_precompute();

    println!("=== Examples Complete ===");
}
