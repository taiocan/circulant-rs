//! Example: Edge Detection using BCCB Filter
//!
//! This example demonstrates edge detection using Sobel operators
//! via the BCCB (Block Circulant with Circulant Blocks) filter.
//!
//! Run with: `cargo run --example edge_detection --features vision`

use circulant_rs::vision::{BCCBFilter, Kernel};
use ndarray::Array2;

fn main() {
    println!("=== Edge Detection with Sobel Filters ===\n");

    // Create a test image with clear edges
    let size = 32;
    let mut image = Array2::zeros((size, size));

    // Create a stepped pattern (good for testing edge detection)
    // Left half darker, right half brighter
    for i in 0..size {
        for j in 0..size {
            if j < size / 2 {
                image[[i, j]] = 0.2;
            } else {
                image[[i, j]] = 0.8;
            }
        }
    }

    // Add a rectangle
    for i in 8..24 {
        for j in 8..16 {
            image[[i, j]] = 0.9;
        }
    }

    println!("Test image: {}x{}", size, size);
    println!("Features: vertical edge at j=16, rectangle (8-24, 8-16)");
    println!();

    // Create Sobel kernels
    let sobel_x = Kernel::<f64>::sobel_x();
    let sobel_y = Kernel::<f64>::sobel_y();
    let laplacian = Kernel::<f64>::laplacian();

    println!("Sobel X kernel (horizontal edge detector):");
    print_kernel(&sobel_x);

    println!("Sobel Y kernel (vertical edge detector):");
    print_kernel(&sobel_y);

    println!("Laplacian kernel (edge detector):");
    print_kernel(&laplacian);

    // Create filters
    let filter_x = BCCBFilter::new(sobel_x, size, size).expect("Failed to create filter");
    let filter_y = BCCBFilter::new(sobel_y, size, size).expect("Failed to create filter");
    let filter_lap = BCCBFilter::new(laplacian, size, size).expect("Failed to create filter");

    // Apply filters
    let edges_x = filter_x.apply(&image).expect("Failed to apply Sobel X");
    let edges_y = filter_y.apply(&image).expect("Failed to apply Sobel Y");
    let edges_lap = filter_lap.apply(&image).expect("Failed to apply Laplacian");

    // Compute gradient magnitude: sqrt(Gx^2 + Gy^2)
    let gradient_vec: Vec<f64> = edges_x
        .iter()
        .zip(edges_y.iter())
        .map(|(&gx, &gy)| (gx * gx + gy * gy).sqrt())
        .collect();
    let gradient_mag = Array2::from_shape_vec((size, size), gradient_vec).expect("Shape error");

    // Show results
    println!("Edge detection results:");
    println!(
        "  Sobel X range: [{:.4}, {:.4}]",
        min(&edges_x),
        max(&edges_x)
    );
    println!(
        "  Sobel Y range: [{:.4}, {:.4}]",
        min(&edges_y),
        max(&edges_y)
    );
    println!(
        "  Laplacian range: [{:.4}, {:.4}]",
        min(&edges_lap),
        max(&edges_lap)
    );
    println!("  Gradient magnitude max: {:.4}", max(&gradient_mag));
    println!();

    // ASCII visualization of gradient magnitude
    println!("Gradient magnitude (ASCII visualization):");
    println!("{}", "-".repeat(size + 2));

    let mag_max = max(&gradient_mag);
    let chars = [' ', '.', ':', '+', '*', '#'];

    for i in 0..size {
        print!("|");
        for j in 0..size {
            let val = gradient_mag[[i, j]];
            let level = ((val / mag_max) * (chars.len() - 1) as f64) as usize;
            let level = level.min(chars.len() - 1);
            print!("{}", chars[level]);
        }
        println!("|");
    }

    println!("{}", "-".repeat(size + 2));
    println!();

    // Show cross-section through the vertical edge
    println!("Cross-section at row 16 (Sobel X response):");
    for j in 10..22 {
        let val = edges_x[[16, j]];
        let bar = if val > 0.0 {
            "+".repeat((val * 20.0).abs() as usize)
        } else {
            "-".repeat((val * 20.0).abs() as usize)
        };
        println!("  j={:2}: {:+.4} {}", j, val, bar);
    }

    println!();
    println!("Sobel filters detect edges by computing image gradients.");
    println!("Strong responses occur at intensity transitions.");
}

fn print_kernel(kernel: &Kernel<f64>) {
    let (rows, cols) = kernel.size();
    for i in 0..rows {
        print!("  ");
        for j in 0..cols {
            print!("{:+.1} ", kernel.data()[[i, j]].re);
        }
        println!();
    }
    println!();
}

fn min(arr: &Array2<f64>) -> f64 {
    arr.iter().cloned().fold(f64::INFINITY, f64::min)
}

fn max(arr: &Array2<f64>) -> f64 {
    arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}
