// @module: crate::vision::kernel
// @status: stable
// @owner: math_expert
// @feature: vision
// @depends: [crate::error, crate::traits, ndarray, num_complex]
// @tests: [unit]
// @version: 0.3.0
// @event: DE-2026-001

//! Convolution kernel definitions and builders.

use ndarray::Array2;
use num_complex::Complex;
use rustfft::FftNum;

use crate::error::{CirculantError, Result};
use crate::traits::Scalar;

/// A convolution kernel for image filtering.
///
/// Kernels can be created from predefined types (Gaussian, Sobel, etc.)
/// or custom coefficient arrays.
pub struct Kernel<T: Scalar> {
    data: Array2<Complex<T>>,
    name: &'static str,
}

impl<T: Scalar + FftNum> Kernel<T> {
    /// Create a Gaussian blur kernel.
    ///
    /// # Arguments
    ///
    /// * `sigma` - Standard deviation of the Gaussian
    /// * `size` - Kernel size (must be odd)
    ///
    /// # Returns
    ///
    /// A normalized Gaussian kernel.
    ///
    /// # Errors
    ///
    /// Returns `InvalidKernel` if size is even or sigma is non-positive.
    pub fn gaussian(sigma: f64, size: usize) -> Result<Self> {
        if size.is_multiple_of(2) {
            return Err(CirculantError::InvalidKernel(
                "kernel size must be odd".to_string(),
            ));
        }
        if sigma <= 0.0 {
            return Err(CirculantError::InvalidKernel(
                "sigma must be positive".to_string(),
            ));
        }

        let half = (size / 2) as isize;
        let mut data = Array2::zeros((size, size));
        let mut sum = T::zero();

        for i in 0..size {
            for j in 0..size {
                let x = (i as isize - half) as f64;
                let y = (j as isize - half) as f64;
                let val = (-(x * x + y * y) / (2.0 * sigma * sigma)).exp();
                let val_t = T::from(val).unwrap_or_else(T::zero);
                data[[i, j]] = Complex::new(val_t, T::zero());
                sum += val_t;
            }
        }

        // Normalize
        for elem in data.iter_mut() {
            elem.re /= sum;
        }

        Ok(Self {
            data,
            name: "gaussian",
        })
    }

    /// Create a Sobel edge detection kernel for horizontal edges.
    pub fn sobel_x() -> Self {
        let data = Array2::from_shape_vec(
            (3, 3),
            vec![
                Complex::new(T::from(-1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::zero(), T::zero()),
                Complex::new(T::from(1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::from(-2.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::zero(), T::zero()),
                Complex::new(T::from(2.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::from(-1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::zero(), T::zero()),
                Complex::new(T::from(1.0).unwrap_or_else(T::zero), T::zero()),
            ],
        )
        .unwrap_or_else(|_| Array2::zeros((3, 3)));

        Self {
            data,
            name: "sobel_x",
        }
    }

    /// Create a Sobel edge detection kernel for vertical edges.
    pub fn sobel_y() -> Self {
        let data = Array2::from_shape_vec(
            (3, 3),
            vec![
                Complex::new(T::from(-1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::from(-2.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::from(-1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::zero(), T::zero()),
                Complex::new(T::zero(), T::zero()),
                Complex::new(T::zero(), T::zero()),
                Complex::new(T::from(1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::from(2.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::from(1.0).unwrap_or_else(T::zero), T::zero()),
            ],
        )
        .unwrap_or_else(|_| Array2::zeros((3, 3)));

        Self {
            data,
            name: "sobel_y",
        }
    }

    /// Create a Prewitt edge detection kernel for vertical edges.
    ///
    /// The Prewitt operator uses uniform weights (1,1,1) compared to
    /// Sobel's weighted center (1,2,1), making it simpler but slightly
    /// more sensitive to noise.
    pub fn prewitt_x() -> Self {
        // Coefficients: [[-1,0,1],[-1,0,1],[-1,0,1]]
        let data = Array2::from_shape_vec(
            (3, 3),
            vec![
                Complex::new(T::from(-1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::zero(), T::zero()),
                Complex::new(T::from(1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::from(-1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::zero(), T::zero()),
                Complex::new(T::from(1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::from(-1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::zero(), T::zero()),
                Complex::new(T::from(1.0).unwrap_or_else(T::zero), T::zero()),
            ],
        )
        .unwrap_or_else(|_| Array2::zeros((3, 3)));

        Self {
            data,
            name: "prewitt_x",
        }
    }

    /// Create a Prewitt edge detection kernel for horizontal edges.
    ///
    /// The Prewitt Y kernel is the transpose of Prewitt X, detecting
    /// intensity gradients in the vertical direction.
    pub fn prewitt_y() -> Self {
        // Coefficients: [[-1,-1,-1],[0,0,0],[1,1,1]]
        let data = Array2::from_shape_vec(
            (3, 3),
            vec![
                Complex::new(T::from(-1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::from(-1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::from(-1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::zero(), T::zero()),
                Complex::new(T::zero(), T::zero()),
                Complex::new(T::zero(), T::zero()),
                Complex::new(T::from(1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::from(1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::from(1.0).unwrap_or_else(T::zero), T::zero()),
            ],
        )
        .unwrap_or_else(|_| Array2::zeros((3, 3)));

        Self {
            data,
            name: "prewitt_y",
        }
    }

    /// Create a 3x3 sharpen kernel.
    ///
    /// Enhances edges by amplifying the center pixel relative to neighbors.
    /// The kernel structure is:
    /// ```text
    /// [[ 0, -1,  0],
    ///  [-1,  5, -1],
    ///  [ 0, -1,  0]]
    /// ```
    ///
    /// Sum of coefficients is 1, preserving overall brightness.
    /// This is equivalent to identity + negative Laplacian.
    pub fn sharpen() -> Self {
        let data = Array2::from_shape_vec(
            (3, 3),
            vec![
                Complex::new(T::zero(), T::zero()),
                Complex::new(T::from(-1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::zero(), T::zero()),
                Complex::new(T::from(-1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::from(5.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::from(-1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::zero(), T::zero()),
                Complex::new(T::from(-1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::zero(), T::zero()),
            ],
        )
        .unwrap_or_else(|_| Array2::zeros((3, 3)));

        Self {
            data,
            name: "sharpen",
        }
    }

    /// Create a Laplacian edge detection kernel.
    pub fn laplacian() -> Self {
        let data = Array2::from_shape_vec(
            (3, 3),
            vec![
                Complex::new(T::zero(), T::zero()),
                Complex::new(T::from(1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::zero(), T::zero()),
                Complex::new(T::from(1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::from(-4.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::from(1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::zero(), T::zero()),
                Complex::new(T::from(1.0).unwrap_or_else(T::zero), T::zero()),
                Complex::new(T::zero(), T::zero()),
            ],
        )
        .unwrap_or_else(|_| Array2::zeros((3, 3)));

        Self {
            data,
            name: "laplacian",
        }
    }

    /// Create a box blur kernel.
    ///
    /// # Arguments
    ///
    /// * `size` - Kernel size (must be odd)
    ///
    /// # Errors
    ///
    /// Returns `InvalidKernel` if size is even.
    pub fn box_blur(size: usize) -> Result<Self> {
        if size.is_multiple_of(2) {
            return Err(CirculantError::InvalidKernel(
                "kernel size must be odd".to_string(),
            ));
        }

        let n = size * size;
        let val = T::from(1.0 / n as f64).unwrap_or_else(T::zero);
        let data = Array2::from_elem((size, size), Complex::new(val, T::zero()));

        Ok(Self {
            data,
            name: "box_blur",
        })
    }

    /// Create a custom kernel from coefficients.
    ///
    /// # Arguments
    ///
    /// * `coeffs` - 2D array of kernel coefficients
    ///
    /// # Errors
    ///
    /// Returns `InvalidKernel` if dimensions are not odd.
    pub fn custom(coeffs: Array2<T>) -> Result<Self> {
        let (rows, cols) = coeffs.dim();
        if rows.is_multiple_of(2) || cols.is_multiple_of(2) {
            return Err(CirculantError::InvalidKernel(
                "kernel dimensions must be odd".to_string(),
            ));
        }

        let data = coeffs.mapv(|v| Complex::new(v, T::zero()));

        Ok(Self {
            data,
            name: "custom",
        })
    }

    /// Get the kernel data.
    pub fn data(&self) -> &Array2<Complex<T>> {
        &self.data
    }

    /// Get the kernel size as (rows, cols).
    pub fn size(&self) -> (usize, usize) {
        self.data.dim()
    }

    /// Get the kernel name.
    pub fn name(&self) -> &'static str {
        self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gaussian_kernel_normalized() {
        let kernel = Kernel::<f64>::gaussian(1.0, 5).unwrap();
        let sum: f64 = kernel.data().iter().map(|c| c.re).sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gaussian_invalid_size() {
        let result = Kernel::<f64>::gaussian(1.0, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_gaussian_invalid_sigma() {
        let result = Kernel::<f64>::gaussian(-1.0, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_sobel_x_structure() {
        let kernel = Kernel::<f64>::sobel_x();
        assert_eq!(kernel.size(), (3, 3));
        assert_eq!(kernel.name(), "sobel_x");
    }

    #[test]
    fn test_sobel_y_structure() {
        let kernel = Kernel::<f64>::sobel_y();
        assert_eq!(kernel.size(), (3, 3));
        assert_eq!(kernel.name(), "sobel_y");
    }

    #[test]
    fn test_laplacian_structure() {
        let kernel = Kernel::<f64>::laplacian();
        assert_eq!(kernel.size(), (3, 3));
        // Center should be -4
        assert_relative_eq!(kernel.data()[[1, 1]].re, -4.0);
    }

    #[test]
    fn test_box_blur_normalized() {
        let kernel = Kernel::<f64>::box_blur(3).unwrap();
        let sum: f64 = kernel.data().iter().map(|c| c.re).sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_custom_kernel() {
        let coeffs = Array2::from_shape_vec((3, 3), vec![0.0; 9]).unwrap();
        let kernel = Kernel::<f64>::custom(coeffs).unwrap();
        assert_eq!(kernel.size(), (3, 3));
        assert_eq!(kernel.name(), "custom");
    }

    #[test]
    fn test_custom_kernel_invalid_size() {
        let coeffs = Array2::from_shape_vec((4, 4), vec![0.0; 16]).unwrap();
        let result = Kernel::<f64>::custom(coeffs);
        assert!(result.is_err());
    }

    // @math_verified: true
    // @verified_by: math_expert
    // @properties_checked: [coefficients, total_sum_zero]
    #[test]
    fn test_prewitt_x_structure() {
        let kernel = Kernel::<f64>::prewitt_x();
        assert_eq!(kernel.size(), (3, 3));
        assert_eq!(kernel.name(), "prewitt_x");

        // Expected: [[-1,0,1],[-1,0,1],[-1,0,1]]
        let data = kernel.data();
        // Row 0
        assert_relative_eq!(data[[0, 0]].re, -1.0);
        assert_relative_eq!(data[[0, 1]].re, 0.0);
        assert_relative_eq!(data[[0, 2]].re, 1.0);
        // Row 1
        assert_relative_eq!(data[[1, 0]].re, -1.0);
        assert_relative_eq!(data[[1, 1]].re, 0.0);
        assert_relative_eq!(data[[1, 2]].re, 1.0);
        // Row 2
        assert_relative_eq!(data[[2, 0]].re, -1.0);
        assert_relative_eq!(data[[2, 1]].re, 0.0);
        assert_relative_eq!(data[[2, 2]].re, 1.0);
    }

    // @math_verified: true
    // @verified_by: math_expert
    // @properties_checked: [coefficients, total_sum_zero]
    #[test]
    fn test_prewitt_y_structure() {
        let kernel = Kernel::<f64>::prewitt_y();
        assert_eq!(kernel.size(), (3, 3));
        assert_eq!(kernel.name(), "prewitt_y");

        // Expected: [[-1,-1,-1],[0,0,0],[1,1,1]]
        let data = kernel.data();
        // Row 0
        assert_relative_eq!(data[[0, 0]].re, -1.0);
        assert_relative_eq!(data[[0, 1]].re, -1.0);
        assert_relative_eq!(data[[0, 2]].re, -1.0);
        // Row 1
        assert_relative_eq!(data[[1, 0]].re, 0.0);
        assert_relative_eq!(data[[1, 1]].re, 0.0);
        assert_relative_eq!(data[[1, 2]].re, 0.0);
        // Row 2
        assert_relative_eq!(data[[2, 0]].re, 1.0);
        assert_relative_eq!(data[[2, 1]].re, 1.0);
        assert_relative_eq!(data[[2, 2]].re, 1.0);
    }

    // @math_verified: true
    // @verified_by: math_expert
    // @properties_checked: [column_sums, total_sum_zero]
    #[test]
    fn test_prewitt_x_vertical_edge_detection() {
        // Prewitt X should detect vertical edges (intensity changes in x direction)
        // Test with a simple vertical edge pattern
        let kernel = Kernel::<f64>::prewitt_x();
        let data = kernel.data();

        // Sum of left column should be -3 (detects left side of vertical edge)
        let left_sum: f64 = data[[0, 0]].re + data[[1, 0]].re + data[[2, 0]].re;
        assert_relative_eq!(left_sum, -3.0);

        // Sum of right column should be +3 (detects right side of vertical edge)
        let right_sum: f64 = data[[0, 2]].re + data[[1, 2]].re + data[[2, 2]].re;
        assert_relative_eq!(right_sum, 3.0);

        // Sum of middle column should be 0
        let mid_sum: f64 = data[[0, 1]].re + data[[1, 1]].re + data[[2, 1]].re;
        assert_relative_eq!(mid_sum, 0.0);
    }

    // @math_verified: true
    // @verified_by: math_expert
    // @properties_checked: [row_sums, total_sum_zero]
    #[test]
    fn test_prewitt_y_horizontal_edge_detection() {
        // Prewitt Y should detect horizontal edges (intensity changes in y direction)
        // Test with a simple horizontal edge pattern
        let kernel = Kernel::<f64>::prewitt_y();
        let data = kernel.data();

        // Sum of top row should be -3 (detects top side of horizontal edge)
        let top_sum: f64 = data[[0, 0]].re + data[[0, 1]].re + data[[0, 2]].re;
        assert_relative_eq!(top_sum, -3.0);

        // Sum of bottom row should be +3 (detects bottom side of horizontal edge)
        let bottom_sum: f64 = data[[2, 0]].re + data[[2, 1]].re + data[[2, 2]].re;
        assert_relative_eq!(bottom_sum, 3.0);

        // Sum of middle row should be 0
        let mid_sum: f64 = data[[1, 0]].re + data[[1, 1]].re + data[[1, 2]].re;
        assert_relative_eq!(mid_sum, 0.0);
    }

    // @math_verified: true
    // @verified_by: math_expert
    // @properties_checked: [transpose_relation]
    #[test]
    fn test_prewitt_transpose_relation() {
        // Mathematical property: Prewitt Y = transpose(Prewitt X)
        let kernel_x = Kernel::<f64>::prewitt_x();
        let kernel_y = Kernel::<f64>::prewitt_y();
        let data_x = kernel_x.data();
        let data_y = kernel_y.data();

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(
                    data_y[[i, j]].re,
                    data_x[[j, i]].re,
                    epsilon = 1e-10
                );
            }
        }
    }

    // @math_verified: true
    // @verified_by: math_expert
    // @properties_checked: [coefficient_correctness, sum_unity, rotational_symmetry]
    // @version: 0.3.0
    // @event: DE-2026-001
    #[test]
    fn test_sharpen_kernel_properties() {
        let kernel = Kernel::<f64>::sharpen();
        let data = kernel.data();

        // Test: kernel is 3x3
        assert_eq!(kernel.size(), (3, 3));
        assert_eq!(kernel.name(), "sharpen");

        // Test: center value is 5 (amplifies center pixel)
        assert_relative_eq!(data[[1, 1]].re, 5.0);

        // Test: sum of coefficients is 1 (preserves brightness)
        let sum: f64 = data.iter().map(|c| c.re).sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);

        // Test: symmetric structure (rotational symmetry)
        // Expected: [[ 0, -1,  0],
        //            [-1,  5, -1],
        //            [ 0, -1,  0]]
        // Corners are 0
        assert_relative_eq!(data[[0, 0]].re, 0.0);
        assert_relative_eq!(data[[0, 2]].re, 0.0);
        assert_relative_eq!(data[[2, 0]].re, 0.0);
        assert_relative_eq!(data[[2, 2]].re, 0.0);

        // Edge centers are -1
        assert_relative_eq!(data[[0, 1]].re, -1.0);
        assert_relative_eq!(data[[1, 0]].re, -1.0);
        assert_relative_eq!(data[[1, 2]].re, -1.0);
        assert_relative_eq!(data[[2, 1]].re, -1.0);

        // All imaginary parts are 0
        for elem in data.iter() {
            assert_relative_eq!(elem.im, 0.0);
        }
    }
}
