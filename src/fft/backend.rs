// @module: crate::fft::backend
// @status: stable
// @owner: code_expert
// @feature: none
// @depends: [crate::traits, num_complex]
// @tests: [none]

//! FFT backend trait definition.

use crate::traits::Scalar;
use num_complex::Complex;

/// Trait for FFT backend implementations.
///
/// This trait abstracts the FFT implementation, allowing for different backends
/// (e.g., rustfft, FFTW, custom implementations).
pub trait FftBackend<T: Scalar>: Send + Sync {
    /// Perform a forward FFT in-place.
    ///
    /// The input buffer is transformed to frequency domain.
    fn fft_forward(&self, buffer: &mut [Complex<T>]);

    /// Perform an inverse FFT in-place.
    ///
    /// The input buffer is transformed back to time domain.
    /// The result is normalized by 1/N.
    fn fft_inverse(&self, buffer: &mut [Complex<T>]);

    /// Get the size of the FFT.
    fn size(&self) -> usize;

    /// Create a new scratch buffer for this FFT size.
    fn make_scratch(&self) -> Vec<Complex<T>>;
}
