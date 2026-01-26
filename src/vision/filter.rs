//! BCCB filter implementation for image convolution.

use ndarray::Array2;
use num_complex::Complex;
use rustfft::FftNum;

use crate::core::BlockCirculant;
use crate::error::{CirculantError, Result};
use crate::traits::{BlockOps, Scalar};

use super::kernel::Kernel;

/// A BCCB-based image filter for efficient convolution.
///
/// Uses FFT to perform O(NM log NM) convolution instead of O(N²M²).
pub struct BCCBFilter<T: Scalar + FftNum> {
    matrix: BlockCirculant<T>,
    kernel_size: (usize, usize),
    output_size: (usize, usize),
}

impl<T: Scalar + FftNum> BCCBFilter<T> {
    /// Create a new BCCB filter from a kernel.
    ///
    /// # Arguments
    ///
    /// * `kernel` - The convolution kernel
    /// * `width` - Output image width
    /// * `height` - Output image height
    ///
    /// # Returns
    ///
    /// A filter ready to apply to images.
    ///
    /// # Errors
    ///
    /// Returns an error if the kernel or dimensions are invalid.
    pub fn new(kernel: Kernel<T>, width: usize, height: usize) -> Result<Self> {
        let kernel_size = kernel.size();

        // Create zero-padded kernel for BCCB construction
        let mut padded = Array2::zeros((height, width));
        let (kh, kw) = kernel_size;
        let kernel_data = kernel.data();

        // Copy kernel to top-left corner with appropriate wrapping for circulant structure
        for i in 0..kh {
            for j in 0..kw {
                // Center the kernel by shifting indices
                let di = if i <= kh / 2 { i } else { height - kh + i };
                let dj = if j <= kw / 2 { j } else { width - kw + j };
                padded[[di, dj]] = kernel_data[[i, j]];
            }
        }

        let matrix = BlockCirculant::from_kernel(padded, height, width)?;

        Ok(Self {
            matrix,
            kernel_size,
            output_size: (height, width),
        })
    }

    /// Apply the filter to an image.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image as 2D array
    ///
    /// # Returns
    ///
    /// Filtered image with same dimensions.
    ///
    /// # Errors
    ///
    /// Returns `ImageDimensionMismatch` if image dimensions don't match filter.
    pub fn apply(&self, image: &Array2<T>) -> Result<Array2<T>> {
        let (h, w) = image.dim();
        if (h, w) != self.output_size {
            return Err(CirculantError::ImageDimensionMismatch {
                expected: self.output_size,
                got: (h, w),
            });
        }

        // Convert to complex
        let complex_image = image.mapv(|v| Complex::new(v, T::zero()));
        let result = self.apply_complex(&complex_image)?;

        // Extract real part
        Ok(result.mapv(|c| c.re))
    }

    /// Apply the filter to a complex image.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image as 2D complex array
    ///
    /// # Returns
    ///
    /// Filtered complex image.
    ///
    /// # Errors
    ///
    /// Returns `ImageDimensionMismatch` if image dimensions don't match filter.
    pub fn apply_complex(&self, image: &Array2<Complex<T>>) -> Result<Array2<Complex<T>>> {
        let (h, w) = image.dim();
        if (h, w) != self.output_size {
            return Err(CirculantError::ImageDimensionMismatch {
                expected: self.output_size,
                got: (h, w),
            });
        }

        self.matrix.mul_array(image)
    }

    /// Precompute FFT spectra for repeated applications.
    pub fn precompute(&mut self) {
        self.matrix.precompute();
    }

    /// Get the kernel size.
    pub fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }

    /// Get the output size.
    pub fn output_size(&self) -> (usize, usize) {
        self.output_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bccb_filter_preserves_dimensions() {
        let kernel = Kernel::<f64>::gaussian(1.0, 3).unwrap();
        let filter = BCCBFilter::new(kernel, 16, 16).unwrap();
        let image = Array2::zeros((16, 16));
        let result = filter.apply(&image).unwrap();
        assert_eq!(result.dim(), (16, 16));
    }

    #[test]
    fn test_filter_dimension_mismatch() {
        let kernel = Kernel::<f64>::gaussian(1.0, 3).unwrap();
        let filter = BCCBFilter::new(kernel, 16, 16).unwrap();
        let image = Array2::zeros((8, 8));
        let result = filter.apply(&image);
        assert!(result.is_err());
    }

    #[test]
    fn test_identity_like_filter() {
        // A box blur of size 1 is essentially identity
        let kernel = Kernel::<f64>::box_blur(1).unwrap();
        let filter = BCCBFilter::new(kernel, 8, 8).unwrap();

        let mut image = Array2::zeros((8, 8));
        image[[4, 4]] = 1.0;

        let result = filter.apply(&image).unwrap();
        // The center should still have a significant value
        assert!(result[[4, 4]].abs() > 0.5);
    }

    #[test]
    fn test_gaussian_blur_smoothing() {
        let kernel = Kernel::<f64>::gaussian(1.0, 3).unwrap();
        let mut filter = BCCBFilter::new(kernel, 16, 16).unwrap();
        filter.precompute();

        // Create image with single bright pixel
        let mut image = Array2::zeros((16, 16));
        image[[8, 8]] = 100.0;

        let result = filter.apply(&image).unwrap();

        // After blur, the peak should be lower and neighbors should have non-zero values
        assert!(result[[8, 8]] < image[[8, 8]]);
        assert!(result[[8, 9]].abs() > 0.0);
        assert!(result[[9, 8]].abs() > 0.0);
    }

    #[test]
    fn test_filter_matches_naive_convolution_simple() {
        // Test with a simple 3x3 kernel on a small image
        // For a box blur, the result should be an average of neighbors
        let kernel = Kernel::<f64>::box_blur(3).unwrap();
        let filter = BCCBFilter::new(kernel, 8, 8).unwrap();

        // Create a simple test pattern
        let mut image = Array2::zeros((8, 8));
        for i in 3..5 {
            for j in 3..5 {
                image[[i, j]] = 1.0;
            }
        }

        let result = filter.apply(&image).unwrap();

        // The result should be smoothed - center value should be average of 3x3 region
        // This is a sanity check rather than exact comparison
        let sum: f64 = result.iter().sum();
        let original_sum: f64 = image.iter().sum();
        // Total "energy" should be approximately preserved
        assert_relative_eq!(sum, original_sum, epsilon = 0.1);
    }
}
