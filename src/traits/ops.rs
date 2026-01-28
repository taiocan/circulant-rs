// @module: crate::traits::ops
// @status: stable
// @owner: code_expert
// @feature: none
// @depends: [crate::error, crate::traits::numeric, ndarray, num_complex]
// @tests: [none]
// @version: 1.0.0
// @event: DE-2026-002

//! Operation traits for circulant matrices and tensors.

use crate::error::Result;
use crate::traits::numeric::Scalar;
use ndarray::{Array1, Array2, ArrayD};
use num_complex::Complex;

/// Operations for 1D circulant matrices.
pub trait CirculantOps<T: Scalar> {
    /// Multiply the circulant matrix by a vector.
    ///
    /// Uses FFT for O(N log N) complexity.
    fn mul_vec(&self, x: &[Complex<T>]) -> Result<Vec<Complex<T>>>;

    /// Multiply the circulant matrix by a real vector.
    ///
    /// Converts to complex, performs multiplication, returns complex result.
    fn mul_vec_real(&self, x: &[T]) -> Result<Vec<Complex<T>>>;

    /// Get the eigenvalues (DFT of the generator).
    fn eigenvalues(&self) -> Vec<Complex<T>>;

    /// Get the size of the circulant matrix.
    fn size(&self) -> usize;
}

/// Operations for block circulant matrices.
pub trait BlockOps<T: Scalar> {
    /// Multiply the block circulant matrix by a 2D array.
    ///
    /// Uses 2D FFT for O(N log N) complexity.
    fn mul_array(&self, x: &Array2<Complex<T>>) -> Result<Array2<Complex<T>>>;

    /// Multiply the block circulant matrix by a 1D array (flattened 2D).
    fn mul_vec(&self, x: &Array1<Complex<T>>) -> Result<Array1<Complex<T>>>;

    /// Get the 2D eigenvalues (2D DFT of the generator block).
    fn eigenvalues_2d(&self) -> Array2<Complex<T>>;

    /// Get the dimensions (num_blocks, block_rows, block_cols).
    fn dimensions(&self) -> (usize, usize, usize);

    /// Check if this is a BCCB (Block Circulant with Circulant Blocks) matrix.
    fn is_bccb(&self) -> bool;
}

/// Operations for N-dimensional circulant tensors.
///
/// This trait defines the core operations for circulant tensors of arbitrary
/// dimension D. The const generic D provides compile-time dimension checking.
pub trait TensorOps<T: Scalar, const D: usize> {
    /// Multiply the circulant tensor by an N-D array.
    ///
    /// Uses N-D FFT for O(N log N) complexity where N = total elements.
    ///
    /// # Errors
    ///
    /// Returns `InvalidTensorShape` if input dimensions don't match the tensor shape.
    fn mul_tensor(&self, x: &ArrayD<Complex<T>>) -> Result<ArrayD<Complex<T>>>;

    /// Multiply the circulant tensor by a flattened vector.
    ///
    /// The vector is reshaped to the tensor shape, multiplied, and flattened back.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if vector length doesn't match total tensor size.
    fn mul_vec(&self, x: &[Complex<T>]) -> Result<Vec<Complex<T>>>;

    /// Get the N-D eigenvalues (N-D DFT of the generator tensor).
    ///
    /// The eigenvalues have the same shape as the generator tensor.
    fn eigenvalues_nd(&self) -> ArrayD<Complex<T>>;

    /// Get the shape of the tensor as a fixed-size array.
    ///
    /// Returns [n₁, n₂, ..., n_D] where each nᵢ is the size along axis i.
    fn shape(&self) -> [usize; D];

    /// Get the total number of elements (product of all dimensions).
    fn total_size(&self) -> usize;
}

#[cfg(test)]
mod tests {
    // Trait tests will be added when implementations exist
}
