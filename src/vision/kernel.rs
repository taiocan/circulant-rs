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
}
