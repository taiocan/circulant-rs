//! Example: Gaussian Image Blur using BCCB Filter
//!
//! This example demonstrates how to use the BCCB (Block Circulant with Circulant Blocks)
//! filter for efficient image convolution. Gaussian blur is O(N log N) instead of O(N * k^2).
//!
//! Run with: `cargo run --example image_blur --features vision`

use circulant_rs::vision::{BCCBFilter, Kernel};
use ndarray::Array2;

fn main() {
    println!("=== Gaussian Blur with BCCB Filter ===\n");

    // Create a simple test image (64x64 with some features)
    let size = 64;
    let mut image = Array2::zeros((size, size));

    // Add some features: a bright rectangle in the center
    for i in 24..40 {
        for j in 24..40 {
            image[[i, j]] = 1.0;
        }
    }

    // Add a small bright spot
    image[[10, 10]] = 1.0;
    image[[10, 11]] = 1.0;
    image[[11, 10]] = 1.0;
    image[[11, 11]] = 1.0;

    // Add a horizontal line
    for j in 45..60 {
        image[[50, j]] = 1.0;
    }

    println!("Image size: {}x{}", size, size);
    println!("Features: rectangle (24-40, 24-40), spot (10-11, 10-11), line (50, 45-60)");
    println!();

    // Create Gaussian kernels with different sigma values
    let sigmas = [1.0, 2.0, 4.0];

    for sigma in sigmas {
        let kernel_size = (6.0_f64 * sigma).ceil() as usize | 1; // Ensure odd size
        let kernel = Kernel::<f64>::gaussian(sigma, kernel_size).expect("Failed to create kernel");

        println!(
            "Gaussian kernel: sigma={}, size={}x{}",
            sigma, kernel_size, kernel_size
        );

        // Create the BCCB filter
        let filter = BCCBFilter::new(kernel, size, size).expect("Failed to create filter");

        // Apply the blur
        let blurred = filter.apply(&image).expect("Failed to apply filter");

        // Compute statistics
        let original_max = image.iter().cloned().fold(0.0_f64, f64::max);
        let blurred_max = blurred.iter().cloned().fold(0.0_f64, f64::max);
        let original_sum: f64 = image.iter().sum();
        let blurred_sum: f64 = blurred.iter().sum();

        println!(
            "  Original max: {:.4}, sum: {:.2}",
            original_max, original_sum
        );
        println!("  Blurred max: {:.4}, sum: {:.2}", blurred_max, blurred_sum);

        // Show a cross-section through the center of the rectangle
        println!("  Center row (row 32) values:");
        print!("    ");
        for j in (20..44).step_by(4) {
            print!("[{}]={:.3} ", j, blurred[[32, j]]);
        }
        println!();
        println!();
    }

    // Demonstrate kernel properties
    println!("Kernel properties:");
    let kernel = Kernel::<f64>::gaussian(2.0, 9).expect("Failed to create kernel");
    println!(
        "  Gaussian(sigma=2.0, size=9): kernel_size = {:?}",
        kernel.size()
    );

    // Show that kernel sums to 1 (normalized)
    let kernel_sum: f64 = kernel.data().iter().map(|c| c.re).sum();
    println!("  Kernel sum: {:.10} (should be 1.0)", kernel_sum);

    println!();
    println!("BCCB filtering uses FFT-based convolution for O(N log N) performance,");
    println!("making it efficient for large images regardless of kernel size.");
}
