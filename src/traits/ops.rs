//! Operation traits for circulant matrices.

use crate::error::Result;
use crate::traits::numeric::Scalar;
use ndarray::{Array1, Array2};
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

#[cfg(test)]
mod tests {
    // Trait tests will be added when implementations exist
}
