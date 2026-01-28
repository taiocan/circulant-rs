// @module: crate::core::circulant
// @status: stable
// @owner: code_expert
// @feature: none
// @depends: [crate::error, crate::fft, crate::traits]
// @tests: [unit, property]

//! 1D Circulant matrix implementation.
//!
//! **Deprecation Note**: This module provides the legacy `Circulant<T>` type.
//! For new code, prefer using [`CirculantTensor<T, 1>`](crate::core::CirculantTensor).

use crate::error::{CirculantError, Result};
use crate::fft::{FftBackend, RustFftBackend};
use crate::traits::{CirculantOps, Scalar};
use num_complex::Complex;
use std::sync::Arc;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A circulant matrix defined by its first row (generator).
///
/// A circulant matrix is a special Toeplitz matrix where each row is a cyclic
/// shift of the previous row. For example, a 4x4 circulant matrix with
/// generator [c0, c1, c2, c3] looks like:
///
/// ```text
/// | c0  c1  c2  c3 |
/// | c3  c0  c1  c2 |
/// | c2  c3  c0  c1 |
/// | c1  c2  c3  c0 |
/// ```
///
/// The key property exploited by this implementation is that any circulant
/// matrix can be diagonalized by the DFT matrix, enabling O(N log N)
/// matrix-vector multiplication instead of O(N²).
///
/// # Deprecation Note
///
/// This type is deprecated in favor of [`CirculantTensor<T, 1>`](crate::core::CirculantTensor).
/// For new code, prefer using `CirculantTensor` which provides a unified API for all dimensions.
#[deprecated(
    since = "1.0.0",
    note = "Use CirculantTensor<T, 1> or Circulant1D<T> instead for a unified N-D API"
)]
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Circulant<T: Scalar + rustfft::FftNum> {
    /// The generator (first row) of the circulant matrix.
    generator: Vec<Complex<T>>,

    /// Cached eigenvalues (FFT of generator) for repeated operations.
    #[cfg_attr(feature = "serde", serde(skip))]
    cached_spectrum: Option<Vec<Complex<T>>>,

    /// FFT backend.
    #[cfg_attr(feature = "serde", serde(skip))]
    fft: Option<Arc<RustFftBackend<T>>>,
}

#[allow(deprecated)]
impl<T: Scalar + rustfft::FftNum> Circulant<T> {
    /// Create a new circulant matrix from a complex generator.
    ///
    /// # Errors
    ///
    /// Returns `CirculantError::EmptyGenerator` if the generator is empty.
    pub fn new(generator: Vec<Complex<T>>) -> Result<Self> {
        if generator.is_empty() {
            return Err(CirculantError::EmptyGenerator);
        }

        let fft = Arc::new(RustFftBackend::new(generator.len())?);

        Ok(Self {
            generator,
            cached_spectrum: None,
            fft: Some(fft),
        })
    }

    /// Create a new circulant matrix from a real generator.
    ///
    /// # Errors
    ///
    /// Returns `CirculantError::EmptyGenerator` if the generator is empty.
    pub fn from_real(generator: Vec<T>) -> Result<Self> {
        let complex_gen: Vec<Complex<T>> = generator
            .into_iter()
            .map(|x| Complex::new(x, T::zero()))
            .collect();
        Self::new(complex_gen)
    }

    /// Get the generator (first row) of the circulant matrix.
    pub fn generator(&self) -> &[Complex<T>] {
        &self.generator
    }

    /// Precompute and cache the eigenvalues (spectrum) for faster repeated multiplications.
    pub fn precompute(&mut self) {
        if self.cached_spectrum.is_some() {
            return;
        }

        if let Some(ref fft) = self.fft {
            // For circulant matrix multiply Cx = c' * x (convolution with reversed c)
            // where c'[0] = c[0], c'[k] = c[n-k] for k > 0
            // FFT(c') = conj(FFT(conj(c)))
            // So we: 1) conjugate generator, 2) FFT, 3) conjugate result
            let mut spectrum: Vec<Complex<T>> = self
                .generator
                .iter()
                .map(|v| Complex::new(v.re, -v.im))
                .collect();
            fft.fft_forward(&mut spectrum);
            for val in spectrum.iter_mut() {
                *val = Complex::new(val.re, -val.im);
            }
            self.cached_spectrum = Some(spectrum);
        }
    }

    /// Clear the cached spectrum.
    pub fn clear_cache(&mut self) {
        self.cached_spectrum = None;
    }

    /// Check if the spectrum is cached.
    pub fn is_precomputed(&self) -> bool {
        self.cached_spectrum.is_some()
    }

    /// Get the element at row i, column j.
    ///
    /// Uses the circulant property: `C[i,j] = c[(j-i) mod n]`
    pub fn get(&self, i: usize, j: usize) -> Complex<T> {
        let n = self.generator.len();
        let idx = (j + n - i) % n;
        self.generator[idx]
    }

    /// Convert to a dense matrix representation.
    #[allow(clippy::needless_range_loop)]
    pub fn to_dense(&self) -> Vec<Vec<Complex<T>>> {
        let n = self.generator.len();
        let mut matrix = vec![vec![Complex::new(T::zero(), T::zero()); n]; n];
        for i in 0..n {
            for j in 0..n {
                matrix[i][j] = self.get(i, j);
            }
        }
        matrix
    }

    /// Ensure FFT backend is initialized (useful after deserialization).
    ///
    /// # Errors
    ///
    /// Returns `InvalidFftSize` if initialization fails (should not happen for valid circulants).
    #[allow(dead_code)]
    pub fn ensure_fft(&mut self) -> Result<()> {
        if self.fft.is_none() {
            self.fft = Some(Arc::new(RustFftBackend::new(self.generator.len())?));
        }
        Ok(())
    }

    /// Compute eigenvalues using naive O(N²) DFT (fallback method).
    fn compute_eigenvalues_naive(&self) -> Vec<Complex<T>> {
        let n = self.generator.len();
        let mut spectrum = vec![Complex::new(T::zero(), T::zero()); n];

        for (k, spec_k) in spectrum.iter_mut().enumerate() {
            let mut sum = Complex::new(T::zero(), T::zero());
            for j in 0..n {
                // Eigenvalue: λ_k = Σ_j c[j] * e^(2πijk/n) with positive exponent
                let theta = T::from(2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64)
                    .unwrap_or_else(T::zero);
                let omega = Complex::new(theta.cos(), theta.sin());
                sum += self.generator[j] * omega;
            }
            *spec_k = sum;
        }

        spectrum
    }
}

#[allow(deprecated)]
impl<T: Scalar + rustfft::FftNum> CirculantOps<T> for Circulant<T> {
    fn mul_vec(&self, x: &[Complex<T>]) -> Result<Vec<Complex<T>>> {
        let n = self.generator.len();

        if x.len() != n {
            return Err(CirculantError::DimensionMismatch {
                expected: n,
                got: x.len(),
            });
        }

        // Get FFT backend (create if needed - size is valid since generator is non-empty)
        let fft = match self.fft.as_ref() {
            Some(f) => f.clone(),
            None => Arc::new(RustFftBackend::new(n)?),
        };

        // Get spectrum for multiplication
        // Circulant multiply Cx = c' * x where c' is c with reversed tail
        // FFT(c') = conj(FFT(conj(c)))
        let eigenvalues = if let Some(ref cached) = self.cached_spectrum {
            cached.clone()
        } else {
            // Conjugate generator, FFT, conjugate result
            let mut spectrum: Vec<Complex<T>> = self
                .generator
                .iter()
                .map(|v| Complex::new(v.re, -v.im))
                .collect();
            fft.fft_forward(&mut spectrum);
            for val in spectrum.iter_mut() {
                *val = Complex::new(val.re, -val.im);
            }
            spectrum
        };

        // FFT of input
        let mut x_fft = x.to_vec();
        fft.fft_forward(&mut x_fft);

        // Element-wise multiply
        for (y, lambda) in x_fft.iter_mut().zip(eigenvalues.iter()) {
            *y *= *lambda;
        }

        // Inverse FFT
        fft.fft_inverse(&mut x_fft);

        Ok(x_fft)
    }

    fn mul_vec_real(&self, x: &[T]) -> Result<Vec<Complex<T>>> {
        let complex_x: Vec<Complex<T>> = x.iter().map(|&v| Complex::new(v, T::zero())).collect();
        self.mul_vec(&complex_x)
    }

    fn eigenvalues(&self) -> Vec<Complex<T>> {
        // The eigenvalues of a circulant matrix with first row c are:
        // λ_k = Σ_j c[j] * e^(2πijk/n) = conj(FFT(conj(c))[k])
        // The cached spectrum already stores this value.
        if let Some(ref cached) = self.cached_spectrum {
            return cached.clone();
        }

        let n = self.generator.len();

        // Get FFT backend (create if needed - size is valid since generator is non-empty)
        let fft = match self.fft.as_ref() {
            Some(f) => f.clone(),
            None => match RustFftBackend::new(n) {
                Ok(f) => Arc::new(f),
                Err(_) => {
                    // Fallback: compute DFT manually (shouldn't happen for valid circulants)
                    return self.compute_eigenvalues_naive();
                }
            },
        };

        // Conjugate generator, FFT, conjugate result
        let mut spectrum: Vec<Complex<T>> = self
            .generator
            .iter()
            .map(|v| Complex::new(v.re, -v.im))
            .collect();
        fft.fft_forward(&mut spectrum);
        for val in spectrum.iter_mut() {
            *val = Complex::new(val.re, -val.im);
        }
        spectrum
    }

    fn size(&self) -> usize {
        self.generator.len()
    }
}

/// Naive O(N²) circulant matrix-vector multiplication (for testing).
#[allow(clippy::needless_range_loop)]
pub fn naive_circulant_mul<T: Scalar>(
    generator: &[Complex<T>],
    x: &[Complex<T>],
) -> Vec<Complex<T>> {
    let n = generator.len();
    let mut result = vec![Complex::new(T::zero(), T::zero()); n];

    for i in 0..n {
        for j in 0..n {
            let idx = (j + n - i) % n;
            result[i] += generator[idx] * x[j];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_circulant_creation() {
        let gen = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let c = Circulant::new(gen.clone()).unwrap();
        assert_eq!(c.size(), 4);
        assert_eq!(c.generator(), &gen);
    }

    #[test]
    fn test_circulant_from_real() {
        let c = Circulant::from_real(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(c.size(), 4);
        assert_eq!(c.generator()[0], Complex::new(1.0, 0.0));
    }

    #[test]
    fn test_circulant_empty_generator() {
        let result = Circulant::<f64>::new(vec![]);
        assert!(matches!(result, Err(CirculantError::EmptyGenerator)));
    }

    #[test]
    fn test_circulant_indexing() {
        // Generator: [1, 2, 3, 4]
        // Matrix:
        // | 1 2 3 4 |
        // | 4 1 2 3 |
        // | 3 4 1 2 |
        // | 2 3 4 1 |
        let c = Circulant::from_real(vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // First row
        assert_eq!(c.get(0, 0).re, 1.0);
        assert_eq!(c.get(0, 1).re, 2.0);
        assert_eq!(c.get(0, 2).re, 3.0);
        assert_eq!(c.get(0, 3).re, 4.0);

        // Second row
        assert_eq!(c.get(1, 0).re, 4.0);
        assert_eq!(c.get(1, 1).re, 1.0);
        assert_eq!(c.get(1, 2).re, 2.0);
        assert_eq!(c.get(1, 3).re, 3.0);

        // Diagonal
        assert_eq!(c.get(2, 2).re, 1.0);
        assert_eq!(c.get(3, 3).re, 1.0);
    }

    #[test]
    fn test_fft_multiply_matches_naive() {
        // This is the critical correctness test
        let c = Circulant::from_real(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let x: Vec<Complex<f64>> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
        ];

        let fft_result = c.mul_vec(&x).unwrap();
        let naive_result = naive_circulant_mul(c.generator(), &x);

        for (fft, naive) in fft_result.iter().zip(naive_result.iter()) {
            assert_relative_eq!(fft.re, naive.re, epsilon = 1e-10);
            assert_relative_eq!(fft.im, naive.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fft_multiply_matches_naive_random() {
        // Test with more complex values
        let gen: Vec<Complex<f64>> = vec![
            Complex::new(1.5, 0.5),
            Complex::new(-0.5, 1.0),
            Complex::new(2.0, -1.0),
            Complex::new(0.0, 0.5),
            Complex::new(-1.0, -1.0),
            Complex::new(0.5, 0.0),
            Complex::new(1.0, 1.0),
            Complex::new(-0.5, -0.5),
        ];
        let c = Circulant::new(gen).unwrap();

        let x: Vec<Complex<f64>> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.5, 0.5),
            Complex::new(-1.0, 1.0),
            Complex::new(0.0, -1.0),
            Complex::new(2.0, 0.0),
            Complex::new(-0.5, 0.5),
            Complex::new(1.0, -1.0),
            Complex::new(0.0, 0.0),
        ];

        let fft_result = c.mul_vec(&x).unwrap();
        let naive_result = naive_circulant_mul(c.generator(), &x);

        for (fft, naive) in fft_result.iter().zip(naive_result.iter()) {
            assert_relative_eq!(fft.re, naive.re, epsilon = 1e-10);
            assert_relative_eq!(fft.im, naive.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_circulant_eigenvalues_are_dft() {
        // Eigenvalues of a circulant matrix with first row c are:
        // λ_k = Σ_j c[j] * e^(2πijk/n) (note: POSITIVE exponent)
        let gen = vec![1.0, 2.0, 3.0, 4.0];
        let c = Circulant::from_real(gen.clone()).unwrap();
        let eigenvalues = c.eigenvalues();

        // Manually compute eigenvalues with correct formula
        let n = gen.len();
        for k in 0..n {
            let mut lambda = Complex::new(0.0, 0.0);
            for j in 0..n {
                // Positive exponent for true eigenvalues
                let theta = 2.0 * PI * (k as f64) * (j as f64) / (n as f64);
                let omega = Complex::new(theta.cos(), theta.sin());
                lambda = lambda + Complex::new(gen[j], 0.0) * omega;
            }
            assert_relative_eq!(eigenvalues[k].re, lambda.re, epsilon = 1e-10);
            assert_relative_eq!(eigenvalues[k].im, lambda.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_multiply_linearity() {
        let c = Circulant::from_real(vec![1.0, 2.0, 3.0, 4.0]).unwrap();

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
        let alpha = Complex::new(2.5, 0.0);

        // C(x + y) = C(x) + C(y)
        let sum: Vec<Complex<f64>> = x.iter().zip(y.iter()).map(|(a, b)| a + b).collect();
        let c_sum = c.mul_vec(&sum).unwrap();
        let c_x = c.mul_vec(&x).unwrap();
        let c_y = c.mul_vec(&y).unwrap();

        for i in 0..4 {
            let expected = c_x[i] + c_y[i];
            assert_relative_eq!(c_sum[i].re, expected.re, epsilon = 1e-10);
            assert_relative_eq!(c_sum[i].im, expected.im, epsilon = 1e-10);
        }

        // C(alpha * x) = alpha * C(x)
        let scaled_x: Vec<Complex<f64>> = x.iter().map(|v| alpha * v).collect();
        let c_scaled = c.mul_vec(&scaled_x).unwrap();

        for i in 0..4 {
            let expected = alpha * c_x[i];
            assert_relative_eq!(c_scaled[i].re, expected.re, epsilon = 1e-10);
            assert_relative_eq!(c_scaled[i].im, expected.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_precompute_caches_spectrum() {
        let mut c = Circulant::from_real(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(!c.is_precomputed());

        c.precompute();
        assert!(c.is_precomputed());

        // Multiple calls should be idempotent
        c.precompute();
        assert!(c.is_precomputed());

        c.clear_cache();
        assert!(!c.is_precomputed());
    }

    #[test]
    fn test_dimension_mismatch() {
        let c = Circulant::from_real(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let x = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]; // Wrong size

        let result = c.mul_vec(&x);
        assert!(matches!(
            result,
            Err(CirculantError::DimensionMismatch {
                expected: 4,
                got: 2
            })
        ));
    }

    #[test]
    fn test_mul_vec_real() {
        let c = Circulant::from_real(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let x = vec![1.0, 0.0, 1.0, 0.0];

        let result = c.mul_vec_real(&x).unwrap();
        let expected = c
            .mul_vec(&x.iter().map(|&v| Complex::new(v, 0.0)).collect::<Vec<_>>())
            .unwrap();

        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(r.re, e.re, epsilon = 1e-10);
            assert_relative_eq!(r.im, e.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_to_dense() {
        let c = Circulant::from_real(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let dense = c.to_dense();

        // Expected:
        // | 1 2 3 4 |
        // | 4 1 2 3 |
        // | 3 4 1 2 |
        // | 2 3 4 1 |
        assert_eq!(dense[0][0].re, 1.0);
        assert_eq!(dense[0][1].re, 2.0);
        assert_eq!(dense[1][0].re, 4.0);
        assert_eq!(dense[1][1].re, 1.0);
        assert_eq!(dense[2][2].re, 1.0);
        assert_eq!(dense[3][3].re, 1.0);
    }

    #[test]
    fn test_identity_circulant() {
        // Identity circulant: [1, 0, 0, 0] should act as identity
        let c = Circulant::from_real(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let x: Vec<Complex<f64>> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];

        let result = c.mul_vec(&x).unwrap();

        for (r, xi) in result.iter().zip(x.iter()) {
            assert_relative_eq!(r.re, xi.re, epsilon = 1e-10);
            assert_relative_eq!(r.im, xi.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_shift_circulant() {
        // Shift circulant: [0, 1, 0, 0] should shift elements
        let c = Circulant::from_real(vec![0.0, 1.0, 0.0, 0.0]).unwrap();
        let x: Vec<Complex<f64>> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];

        let result = c.mul_vec(&x).unwrap();

        // Result should be [2, 3, 4, 1] (circular shift right)
        assert_relative_eq!(result[0].re, 2.0, epsilon = 1e-10);
        assert_relative_eq!(result[1].re, 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[2].re, 4.0, epsilon = 1e-10);
        assert_relative_eq!(result[3].re, 1.0, epsilon = 1e-10);
    }
}
