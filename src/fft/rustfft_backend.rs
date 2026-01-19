//! RustFFT backend implementation.

use crate::fft::backend::FftBackend;
use crate::traits::Scalar;
use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// FFT backend using the rustfft crate.
///
/// This is the default FFT backend for circulant-rs.
pub struct RustFftBackend<T: Scalar + rustfft::FftNum> {
    forward: Arc<dyn Fft<T>>,
    inverse: Arc<dyn Fft<T>>,
    size: usize,
    scratch_len: usize,
}

impl<T: Scalar + rustfft::FftNum> RustFftBackend<T> {
    /// Create a new RustFFT backend for the given size.
    ///
    /// # Panics
    ///
    /// Panics if size is 0.
    pub fn new(size: usize) -> Self {
        assert!(size > 0, "FFT size must be positive");

        let mut planner = FftPlanner::new();
        let forward = planner.plan_fft_forward(size);
        let inverse = planner.plan_fft_inverse(size);

        let scratch_len = forward.get_inplace_scratch_len().max(inverse.get_inplace_scratch_len());

        Self {
            forward,
            inverse,
            size,
            scratch_len,
        }
    }
}

impl<T: Scalar + rustfft::FftNum> FftBackend<T> for RustFftBackend<T> {
    fn fft_forward(&self, buffer: &mut [Complex<T>]) {
        let mut scratch = vec![Complex::new(T::zero(), T::zero()); self.scratch_len];
        self.forward.process_with_scratch(buffer, &mut scratch);
    }

    fn fft_inverse(&self, buffer: &mut [Complex<T>]) {
        let mut scratch = vec![Complex::new(T::zero(), T::zero()); self.scratch_len];
        self.inverse.process_with_scratch(buffer, &mut scratch);

        // Normalize by 1/N
        let scale = T::one() / T::from(self.size).unwrap();
        for val in buffer.iter_mut() {
            *val = Complex::new(val.re * scale, val.im * scale);
        }
    }

    fn size(&self) -> usize {
        self.size
    }

    fn make_scratch(&self) -> Vec<Complex<T>> {
        vec![Complex::new(T::zero(), T::zero()); self.scratch_len]
    }
}

// Implement Send + Sync since Arc<dyn Fft<T>> is Send + Sync
unsafe impl<T: Scalar + rustfft::FftNum> Send for RustFftBackend<T> {}
unsafe impl<T: Scalar + rustfft::FftNum> Sync for RustFftBackend<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_fft_forward_inverse_roundtrip() {
        let backend = RustFftBackend::<f64>::new(8);

        // Original signal: [1, 2, 3, 4, 5, 6, 7, 8]
        let original: Vec<Complex<f64>> = (1..=8)
            .map(|x| Complex::new(x as f64, 0.0))
            .collect();

        let mut buffer = original.clone();

        // Forward FFT
        backend.fft_forward(&mut buffer);

        // Inverse FFT
        backend.fft_inverse(&mut buffer);

        // Should match original
        for (orig, result) in original.iter().zip(buffer.iter()) {
            assert_relative_eq!(orig.re, result.re, epsilon = 1e-10);
            assert_relative_eq!(orig.im, result.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fft_known_values() {
        // DFT of [1, 0, 0, 0] should be [1, 1, 1, 1]
        let backend = RustFftBackend::<f64>::new(4);

        let mut buffer = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];

        backend.fft_forward(&mut buffer);

        for val in &buffer {
            assert_relative_eq!(val.re, 1.0, epsilon = 1e-10);
            assert_relative_eq!(val.im, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fft_known_values_sinusoid() {
        // DFT of a pure sinusoid should have peaks at the frequency bins
        let n = 8;
        let backend = RustFftBackend::<f64>::new(n);

        // cos(2*pi*k/N) for k=0..N-1, with frequency 1
        let mut buffer: Vec<Complex<f64>> = (0..n)
            .map(|k| {
                let theta = 2.0 * PI * (k as f64) / (n as f64);
                Complex::new(theta.cos(), 0.0)
            })
            .collect();

        backend.fft_forward(&mut buffer);

        // For cos, expect peaks at bins 1 and N-1 (which is 7)
        // Bin 0 should be ~0 (DC component)
        // Bin 1 and 7 should be ~N/2 = 4
        assert_relative_eq!(buffer[0].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(buffer[1].re, 4.0, epsilon = 1e-10);
        assert_relative_eq!(buffer[7].re, 4.0, epsilon = 1e-10);

        // Other bins should be ~0
        for i in [2, 3, 4, 5, 6] {
            assert_relative_eq!(buffer[i].re.abs(), 0.0, epsilon = 1e-10);
            assert_relative_eq!(buffer[i].im.abs(), 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fft_linearity() {
        let backend = RustFftBackend::<f64>::new(4);

        let x: Vec<Complex<f64>> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let y: Vec<Complex<f64>> = vec![
            Complex::new(5.0, 0.0),
            Complex::new(6.0, 0.0),
            Complex::new(7.0, 0.0),
            Complex::new(8.0, 0.0),
        ];

        // Compute FFT(x) and FFT(y)
        let mut fft_x = x.clone();
        let mut fft_y = y.clone();
        backend.fft_forward(&mut fft_x);
        backend.fft_forward(&mut fft_y);

        // Compute FFT(x + y)
        let mut sum: Vec<Complex<f64>> = x.iter().zip(y.iter()).map(|(a, b)| a + b).collect();
        backend.fft_forward(&mut sum);

        // FFT(x + y) should equal FFT(x) + FFT(y)
        for i in 0..4 {
            let expected = fft_x[i] + fft_y[i];
            assert_relative_eq!(sum[i].re, expected.re, epsilon = 1e-10);
            assert_relative_eq!(sum[i].im, expected.im, epsilon = 1e-10);
        }

        // Test scaling: FFT(alpha * x) = alpha * FFT(x)
        let alpha = Complex::new(2.5, 0.0);
        let mut scaled_x: Vec<Complex<f64>> = x.iter().map(|v| alpha * v).collect();
        backend.fft_forward(&mut scaled_x);

        let mut original_fft = x.clone();
        backend.fft_forward(&mut original_fft);

        for i in 0..4 {
            let expected = alpha * original_fft[i];
            assert_relative_eq!(scaled_x[i].re, expected.re, epsilon = 1e-10);
            assert_relative_eq!(scaled_x[i].im, expected.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fft_size() {
        let backend = RustFftBackend::<f64>::new(16);
        assert_eq!(backend.size(), 16);
    }

    #[test]
    #[should_panic(expected = "FFT size must be positive")]
    fn test_fft_zero_size_panics() {
        let _ = RustFftBackend::<f64>::new(0);
    }

    #[test]
    fn test_fft_f32() {
        let backend = RustFftBackend::<f32>::new(4);

        let original: Vec<Complex<f32>> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];

        let mut buffer = original.clone();
        backend.fft_forward(&mut buffer);
        backend.fft_inverse(&mut buffer);

        for (orig, result) in original.iter().zip(buffer.iter()) {
            assert_relative_eq!(orig.re, result.re, epsilon = 1e-5);
            assert_relative_eq!(orig.im, result.im, epsilon = 1e-5);
        }
    }
}
