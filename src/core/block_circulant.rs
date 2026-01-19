//! Block Circulant with Circulant Blocks (BCCB) matrix implementation.

use crate::error::{CirculantError, Result};
use crate::fft::{FftBackend, RustFftBackend};
use crate::traits::{BlockOps, Scalar};
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex;
use std::sync::Arc;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A Block Circulant with Circulant Blocks (BCCB) matrix.
///
/// A BCCB matrix is a block circulant matrix where each block is itself circulant.
/// This structure arises naturally in 2D convolution with periodic boundary conditions.
///
/// For an M×N BCCB matrix with block size P×Q:
/// - The outer structure is an M×M circulant matrix of blocks
/// - Each P×Q block is itself circulant
///
/// The key property is that BCCB matrices can be diagonalized by the 2D DFT,
/// enabling O(MN log(MN)) matrix-vector multiplication.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BlockCirculant<T: Scalar + rustfft::FftNum> {
    /// The generator: first block-row of the matrix.
    /// Shape: [num_block_cols, block_rows, block_cols]
    generator: Array3<Complex<T>>,

    /// Number of block rows/cols (outer circulant dimension).
    num_blocks: usize,

    /// Size of each block (inner circulant dimension).
    block_size: (usize, usize),

    /// Whether this is a true BCCB (both levels circulant).
    is_bccb: bool,

    /// Cached 2D eigenvalues for repeated operations.
    #[cfg_attr(feature = "serde", serde(skip))]
    cached_spectrum: Option<Array2<Complex<T>>>,

    /// FFT backend for rows.
    #[cfg_attr(feature = "serde", serde(skip))]
    fft_rows: Option<Arc<RustFftBackend<T>>>,

    /// FFT backend for columns.
    #[cfg_attr(feature = "serde", serde(skip))]
    fft_cols: Option<Arc<RustFftBackend<T>>>,
}

impl<T: Scalar + rustfft::FftNum> BlockCirculant<T> {
    /// Create a BCCB matrix from a 2D kernel (first block).
    ///
    /// The kernel defines the first block of the first block-row.
    /// The full matrix is constructed by circulant extension in both dimensions.
    ///
    /// # Arguments
    ///
    /// * `kernel` - The 2D kernel (first block)
    /// * `output_rows` - Number of rows in the output (defines outer circulant size)
    /// * `output_cols` - Number of columns in the output (defines inner circulant size)
    pub fn from_kernel(
        kernel: Array2<Complex<T>>,
        output_rows: usize,
        output_cols: usize,
    ) -> Result<Self> {
        let (kernel_rows, kernel_cols) = kernel.dim();

        if kernel_rows == 0 || kernel_cols == 0 {
            return Err(CirculantError::EmptyGenerator);
        }

        if output_rows == 0 || output_cols == 0 {
            return Err(CirculantError::InvalidBlockStructure(
                "output dimensions must be positive".to_string(),
            ));
        }

        // Embed kernel in output-sized array (zero-padding)
        let mut padded = Array2::zeros((output_rows, output_cols));
        for i in 0..kernel_rows.min(output_rows) {
            for j in 0..kernel_cols.min(output_cols) {
                padded[(i, j)] = kernel[(i, j)];
            }
        }

        // For BCCB, the generator is just the padded kernel
        // We store it as a 3D array with shape [1, output_rows, output_cols]
        let generator = padded.insert_axis(ndarray::Axis(0));

        let fft_rows = Arc::new(RustFftBackend::new(output_rows));
        let fft_cols = Arc::new(RustFftBackend::new(output_cols));

        Ok(Self {
            generator,
            num_blocks: 1,
            block_size: (output_rows, output_cols),
            is_bccb: true,
            cached_spectrum: None,
            fft_rows: Some(fft_rows),
            fft_cols: Some(fft_cols),
        })
    }

    /// Create a BCCB matrix directly from a 2D generator.
    pub fn new(generator: Array2<Complex<T>>) -> Result<Self> {
        let (rows, cols) = generator.dim();

        if rows == 0 || cols == 0 {
            return Err(CirculantError::EmptyGenerator);
        }

        let gen_3d = generator.insert_axis(ndarray::Axis(0));
        let fft_rows = Arc::new(RustFftBackend::new(rows));
        let fft_cols = Arc::new(RustFftBackend::new(cols));

        Ok(Self {
            generator: gen_3d,
            num_blocks: 1,
            block_size: (rows, cols),
            is_bccb: true,
            cached_spectrum: None,
            fft_rows: Some(fft_rows),
            fft_cols: Some(fft_cols),
        })
    }

    /// Create a BCCB matrix from a real 2D generator.
    pub fn from_real(generator: Array2<T>) -> Result<Self> {
        let complex_gen = generator.mapv(|x| Complex::new(x, T::zero()));
        Self::new(complex_gen)
    }

    /// Get the generator.
    pub fn generator(&self) -> &Array3<Complex<T>> {
        &self.generator
    }

    /// Precompute and cache the 2D spectrum.
    pub fn precompute(&mut self) {
        if self.cached_spectrum.is_some() {
            return;
        }

        self.cached_spectrum = Some(self.compute_spectrum());
    }

    /// Compute the 2D spectrum for multiplication.
    fn compute_spectrum(&self) -> Array2<Complex<T>> {
        let (rows, cols) = self.block_size;
        let fft_rows = self.fft_rows.as_ref().unwrap();
        let fft_cols = self.fft_cols.as_ref().unwrap();

        // Get the 2D generator (first slice of 3D array)
        let gen_2d = self.generator.slice(ndarray::s![0, .., ..]);

        // For 2D BCCB convolution: spectrum = FFT2(generator)
        // The naive implementation uses convolution indexing: g[(i-k), (j-l)]
        // For convolution: y = IFFT2(FFT2(g) * FFT2(x))
        let mut spectrum = gen_2d.to_owned();

        // Apply 2D FFT
        // FFT along rows
        for i in 0..rows {
            let mut row: Vec<Complex<T>> = spectrum.row(i).to_vec();
            fft_cols.fft_forward(&mut row);
            for (j, val) in row.into_iter().enumerate() {
                spectrum[(i, j)] = val;
            }
        }

        // FFT along columns
        for j in 0..cols {
            let mut col: Vec<Complex<T>> = spectrum.column(j).to_vec();
            fft_rows.fft_forward(&mut col);
            for (i, val) in col.into_iter().enumerate() {
                spectrum[(i, j)] = val;
            }
        }

        spectrum
    }

    /// Clear the cached spectrum.
    pub fn clear_cache(&mut self) {
        self.cached_spectrum = None;
    }

    /// Check if the spectrum is cached.
    pub fn is_precomputed(&self) -> bool {
        self.cached_spectrum.is_some()
    }

    /// Ensure FFT backends are initialized (useful after deserialization).
    #[allow(dead_code)]
    pub fn ensure_fft(&mut self) {
        if self.fft_rows.is_none() {
            self.fft_rows = Some(Arc::new(RustFftBackend::new(self.block_size.0)));
        }
        if self.fft_cols.is_none() {
            self.fft_cols = Some(Arc::new(RustFftBackend::new(self.block_size.1)));
        }
    }
}

impl<T: Scalar + rustfft::FftNum> BlockOps<T> for BlockCirculant<T> {
    fn mul_array(&self, x: &Array2<Complex<T>>) -> Result<Array2<Complex<T>>> {
        let (rows, cols) = self.block_size;

        if x.dim() != (rows, cols) {
            return Err(CirculantError::DimensionMismatch {
                expected: rows * cols,
                got: x.nrows() * x.ncols(),
            });
        }

        let fft_rows = self.fft_rows.as_ref().unwrap();
        let fft_cols = self.fft_cols.as_ref().unwrap();

        // Get eigenvalues
        let spectrum = if let Some(ref cached) = self.cached_spectrum {
            cached.clone()
        } else {
            self.compute_spectrum()
        };

        // 2D FFT of input
        let mut x_fft = x.clone();

        // FFT along rows
        for i in 0..rows {
            let mut row: Vec<Complex<T>> = x_fft.row(i).to_vec();
            fft_cols.fft_forward(&mut row);
            for (j, val) in row.into_iter().enumerate() {
                x_fft[(i, j)] = val;
            }
        }

        // FFT along columns
        for j in 0..cols {
            let mut col: Vec<Complex<T>> = x_fft.column(j).to_vec();
            fft_rows.fft_forward(&mut col);
            for (i, val) in col.into_iter().enumerate() {
                x_fft[(i, j)] = val;
            }
        }

        // Element-wise multiply
        let mut result_fft = &x_fft * &spectrum;

        // 2D inverse FFT
        // IFFT along columns
        for j in 0..cols {
            let mut col: Vec<Complex<T>> = result_fft.column(j).to_vec();
            fft_rows.fft_inverse(&mut col);
            for (i, val) in col.into_iter().enumerate() {
                result_fft[(i, j)] = val;
            }
        }

        // IFFT along rows
        for i in 0..rows {
            let mut row: Vec<Complex<T>> = result_fft.row(i).to_vec();
            fft_cols.fft_inverse(&mut row);
            for (j, val) in row.into_iter().enumerate() {
                result_fft[(i, j)] = val;
            }
        }

        Ok(result_fft)
    }

    fn mul_vec(&self, x: &Array1<Complex<T>>) -> Result<Array1<Complex<T>>> {
        let (rows, cols) = self.block_size;
        let expected = rows * cols;

        if x.len() != expected {
            return Err(CirculantError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        // Reshape to 2D
        let x_2d = x
            .to_shape((rows, cols))
            .map_err(|_| CirculantError::InvalidBlockStructure("reshape failed".to_string()))?
            .to_owned();

        // Multiply
        let result_2d = self.mul_array(&x_2d)?;

        // Flatten back
        Ok(result_2d.into_shape_with_order(expected).unwrap())
    }

    fn eigenvalues_2d(&self) -> Array2<Complex<T>> {
        if let Some(ref cached) = self.cached_spectrum {
            cached.clone()
        } else {
            self.compute_spectrum()
        }
    }

    fn dimensions(&self) -> (usize, usize, usize) {
        (self.num_blocks, self.block_size.0, self.block_size.1)
    }

    fn is_bccb(&self) -> bool {
        self.is_bccb
    }
}

/// Naive O(N²) 2D circulant convolution (for testing).
pub fn naive_bccb_mul<T: Scalar>(
    generator: &Array2<Complex<T>>,
    x: &Array2<Complex<T>>,
) -> Array2<Complex<T>> {
    let (rows, cols) = generator.dim();
    let mut result = Array2::zeros((rows, cols));

    for i in 0..rows {
        for j in 0..cols {
            let mut sum = Complex::new(T::zero(), T::zero());
            for k in 0..rows {
                for l in 0..cols {
                    // Circular indices
                    let gi = (i + rows - k) % rows;
                    let gj = (j + cols - l) % cols;
                    sum += generator[(gi, gj)] * x[(k, l)];
                }
            }
            result[(i, j)] = sum;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_block_circulant_creation() {
        let gen = Array2::from_shape_vec(
            (4, 4),
            (0..16)
                .map(|x| Complex::new(x as f64, 0.0))
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let bc = BlockCirculant::new(gen).unwrap();
        assert_eq!(bc.dimensions(), (1, 4, 4));
        assert!(bc.is_bccb());
    }

    #[test]
    fn test_block_circulant_from_real() {
        let gen = Array2::from_shape_vec((3, 3), (1..=9).map(|x| x as f64).collect::<Vec<_>>())
            .unwrap();

        let bc = BlockCirculant::from_real(gen).unwrap();
        assert_eq!(bc.dimensions(), (1, 3, 3));
    }

    #[test]
    fn test_bccb_from_kernel() {
        let kernel = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
            ],
        )
        .unwrap();

        let bc = BlockCirculant::from_kernel(kernel, 4, 4).unwrap();
        assert_eq!(bc.dimensions(), (1, 4, 4));
    }

    #[test]
    fn test_block_multiply_matches_naive_2d() {
        let gen = Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
            ],
        )
        .unwrap();

        let x = Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
            ],
        )
        .unwrap();

        let bc = BlockCirculant::new(gen.clone()).unwrap();
        let fft_result = bc.mul_array(&x).unwrap();
        let naive_result = naive_bccb_mul(&gen, &x);

        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(fft_result[(i, j)].re, naive_result[(i, j)].re, epsilon = 1e-10);
                assert_relative_eq!(fft_result[(i, j)].im, naive_result[(i, j)].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_block_multiply_random() {
        // Test with more varied data
        let gen = Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex::new(1.0, 0.5),
                Complex::new(-0.5, 1.0),
                Complex::new(2.0, -1.0),
                Complex::new(0.0, 0.5),
                Complex::new(-1.0, -1.0),
                Complex::new(0.5, 0.0),
                Complex::new(1.0, 1.0),
                Complex::new(-0.5, -0.5),
                Complex::new(0.5, 0.5),
                Complex::new(1.0, 0.0),
                Complex::new(-1.0, 0.5),
                Complex::new(0.0, -1.0),
                Complex::new(2.0, 0.0),
                Complex::new(-0.5, 0.5),
                Complex::new(1.0, -1.0),
                Complex::new(0.0, 0.0),
            ],
        )
        .unwrap();

        let x = Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.5, 0.5),
                Complex::new(-1.0, 1.0),
                Complex::new(0.0, -1.0),
                Complex::new(2.0, 0.0),
                Complex::new(-0.5, 0.5),
                Complex::new(1.0, -1.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.5, 0.5),
                Complex::new(1.0, 0.0),
                Complex::new(-1.0, 0.5),
                Complex::new(0.0, -1.0),
                Complex::new(1.0, 1.0),
                Complex::new(-0.5, 0.5),
                Complex::new(0.0, -1.0),
                Complex::new(0.5, 0.0),
            ],
        )
        .unwrap();

        let bc = BlockCirculant::new(gen.clone()).unwrap();
        let fft_result = bc.mul_array(&x).unwrap();
        let naive_result = naive_bccb_mul(&gen, &x);

        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(fft_result[(i, j)].re, naive_result[(i, j)].re, epsilon = 1e-9);
                assert_relative_eq!(fft_result[(i, j)].im, naive_result[(i, j)].im, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_identity_bccb() {
        // Identity: generator with 1 at (0,0), rest zeros
        let mut gen = Array2::zeros((4, 4));
        gen[(0, 0)] = Complex::new(1.0, 0.0);

        let x = Array2::from_shape_vec(
            (4, 4),
            (0..16)
                .map(|i| Complex::new(i as f64, (i as f64) * 0.5))
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let bc = BlockCirculant::new(gen).unwrap();
        let result = bc.mul_array(&x).unwrap();

        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(result[(i, j)].re, x[(i, j)].re, epsilon = 1e-10);
                assert_relative_eq!(result[(i, j)].im, x[(i, j)].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_mul_vec() {
        let gen = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
            ],
        )
        .unwrap();

        let x = Array1::from_vec(vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        ]);

        let bc = BlockCirculant::new(gen.clone()).unwrap();
        let result = bc.mul_vec(&x).unwrap();

        // Compare with 2D version
        let x_2d = x.to_shape((2, 2)).unwrap().to_owned();
        let result_2d = bc.mul_array(&x_2d).unwrap();
        let result_2d_flat = result_2d.into_shape_with_order(4).unwrap();

        for i in 0..4 {
            assert_relative_eq!(result[i].re, result_2d_flat[i].re, epsilon = 1e-10);
            assert_relative_eq!(result[i].im, result_2d_flat[i].im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_precompute() {
        let gen = Array2::from_shape_vec(
            (4, 4),
            (0..16)
                .map(|x| Complex::new(x as f64, 0.0))
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let mut bc = BlockCirculant::new(gen).unwrap();
        assert!(!bc.is_precomputed());

        bc.precompute();
        assert!(bc.is_precomputed());

        bc.clear_cache();
        assert!(!bc.is_precomputed());
    }

    #[test]
    fn test_eigenvalues_2d() {
        let gen = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
            ],
        )
        .unwrap();

        let bc = BlockCirculant::new(gen).unwrap();
        let spectrum = bc.eigenvalues_2d();

        assert_eq!(spectrum.dim(), (2, 2));
    }

    #[test]
    fn test_dimension_mismatch() {
        let gen = Array2::from_shape_vec(
            (4, 4),
            (0..16)
                .map(|x| Complex::new(x as f64, 0.0))
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let bc = BlockCirculant::new(gen).unwrap();

        let x = Array2::zeros((3, 3)); // Wrong size
        let result = bc.mul_array(&x);
        assert!(matches!(result, Err(CirculantError::DimensionMismatch { .. })));
    }
}
