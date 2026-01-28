//! Example: 3D Circulant Tensor Operations
//!
//! This example demonstrates using CirculantTensor for 3D convolution operations,
//! such as applying a 3D Gaussian blur to a volumetric data set.
//!
//! Run with: `cargo run --example tensor_3d`
//!
//! @event: DE-2026-002

use circulant_rs::{CirculantTensor, TensorOps};
use ndarray::{ArrayD, IxDyn};
use num_complex::Complex;
use std::f64::consts::PI;

fn complex(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

/// Create a 3D Gaussian kernel.
fn gaussian_3d(shape: [usize; 3], sigma: f64) -> ArrayD<Complex<f64>> {
    let [nx, ny, nz] = shape;
    let mut kernel = ArrayD::zeros(IxDyn(&[nx, ny, nz]));

    let center_x = nx / 2;
    let center_y = ny / 2;
    let center_z = nz / 2;

    let sigma_sq = sigma * sigma;
    let norm = 1.0 / ((2.0 * PI * sigma_sq).powf(1.5));

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                // Wrap indices for circulant structure
                let di = if i <= center_x {
                    i as f64
                } else {
                    (i as isize - nx as isize) as f64
                };
                let dj = if j <= center_y {
                    j as f64
                } else {
                    (j as isize - ny as isize) as f64
                };
                let dk = if k <= center_z {
                    k as f64
                } else {
                    (k as isize - nz as isize) as f64
                };

                let r_sq = di * di + dj * dj + dk * dk;
                let value = norm * (-r_sq / (2.0 * sigma_sq)).exp();
                kernel[IxDyn(&[i, j, k])] = complex(value, 0.0);
            }
        }
    }

    // Normalize so sum = 1
    let sum: f64 = kernel.iter().map(|c| c.re).sum();
    kernel.mapv(|c| complex(c.re / sum, 0.0))
}

/// Create a test volume with a point source at center.
fn point_source(shape: [usize; 3]) -> ArrayD<Complex<f64>> {
    let [nx, ny, nz] = shape;
    let mut volume = ArrayD::zeros(IxDyn(&[nx, ny, nz]));
    volume[IxDyn(&[nx / 2, ny / 2, nz / 2])] = complex(1.0, 0.0);
    volume
}

/// Create a test volume with a small cube at center.
fn small_cube(shape: [usize; 3], cube_size: usize) -> ArrayD<Complex<f64>> {
    let [nx, ny, nz] = shape;
    let mut volume = ArrayD::zeros(IxDyn(&[nx, ny, nz]));

    let start_x = nx / 2 - cube_size / 2;
    let start_y = ny / 2 - cube_size / 2;
    let start_z = nz / 2 - cube_size / 2;

    for i in 0..cube_size {
        for j in 0..cube_size {
            for k in 0..cube_size {
                volume[IxDyn(&[start_x + i, start_y + j, start_z + k])] = complex(1.0, 0.0);
            }
        }
    }

    volume
}

fn main() {
    println!("=== 3D Circulant Tensor Example ===\n");

    let shape = [32, 32, 32];
    let total_elements = shape[0] * shape[1] * shape[2];

    println!("Volume shape: {:?}", shape);
    println!("Total elements: {}\n", total_elements);

    // Create a 3D Gaussian blur kernel
    let sigma = 2.0;
    println!("Creating 3D Gaussian kernel (sigma = {})...", sigma);
    let kernel = gaussian_3d(shape, sigma);

    // Build the circulant tensor
    println!("Building CirculantTensor...");
    let mut tensor = CirculantTensor::<f64, 3>::new(kernel).expect("Failed to create tensor");

    // Precompute for faster repeated operations
    tensor.precompute();
    println!("Precomputed FFT spectrum.\n");

    // Test 1: Apply blur to point source
    println!("--- Test 1: Blur Point Source ---");
    let point = point_source(shape);
    let blurred_point = tensor.mul_tensor(&point).expect("Multiplication failed");

    // Analyze the result
    let center_value = blurred_point[IxDyn(&[16, 16, 16])].re;
    let neighbor_value = blurred_point[IxDyn(&[16, 17, 16])].re;

    println!("Center value after blur: {:.6}", center_value);
    println!("Neighbor value after blur: {:.6}", neighbor_value);
    println!("Blur ratio (neighbor/center): {:.3}", neighbor_value / center_value);

    // Verify energy conservation
    let input_sum: f64 = point.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    let output_sum: f64 = blurred_point
        .iter()
        .map(|c| c.re.abs())
        .sum::<f64>();
    println!("Input energy: {:.6}", input_sum);
    println!("Output sum: {:.6} (energy spreads)\n", output_sum);

    // Test 2: Apply blur to small cube
    println!("--- Test 2: Blur Small Cube ---");
    let cube = small_cube(shape, 4);
    let blurred_cube = tensor.mul_tensor(&cube).expect("Multiplication failed");

    // Compare variance before and after
    let compute_variance = |vol: &ArrayD<Complex<f64>>| -> f64 {
        let mean: f64 = vol.iter().map(|c| c.re).sum::<f64>() / total_elements as f64;
        vol.iter()
            .map(|c| (c.re - mean).powi(2))
            .sum::<f64>()
            / total_elements as f64
    };

    let var_before = compute_variance(&cube);
    let var_after = compute_variance(&blurred_cube);

    println!("Variance before blur: {:.8}", var_before);
    println!("Variance after blur: {:.8}", var_after);
    println!("Variance reduction: {:.1}%", (1.0 - var_after / var_before) * 100.0);
    println!();

    // Test 3: Eigenvalue analysis
    println!("--- Test 3: Eigenvalue Analysis ---");
    let eigenvalues = tensor.eigenvalues_nd();
    println!("Number of eigenvalues: {}", eigenvalues.len());

    // Find min/max eigenvalue magnitudes
    let magnitudes: Vec<f64> = eigenvalues.iter().map(|c| c.norm()).collect();
    let max_mag = magnitudes.iter().cloned().fold(0.0_f64, f64::max);
    let min_mag = magnitudes.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("Max eigenvalue magnitude: {:.6}", max_mag);
    println!("Min eigenvalue magnitude: {:.6}", min_mag);
    println!("Condition number estimate: {:.2}", max_mag / min_mag);

    // Verify eigenvalues sum (should equal sum of kernel)
    let eigenvalue_sum: Complex<f64> = eigenvalues.iter().sum();
    println!(
        "Sum of eigenvalues: {:.6} + {:.6}i",
        eigenvalue_sum.re, eigenvalue_sum.im
    );
    println!();

    println!("=== 3D Tensor Example Complete ===");
}
