// @module: crate::python::circulant
// @status: stable
// @owner: code_expert
// @feature: python
// @depends: [crate::traits, crate::Circulant, pyo3]
// @tests: [none]

//! Python wrapper for Circulant matrix.

#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;

use crate::traits::CirculantOps;
use crate::Circulant;

/// A circulant matrix for FFT-accelerated operations.
#[pyclass(name = "Circulant")]
pub struct PyCirculant {
    inner: Circulant<f64>,
}

#[pymethods]
impl PyCirculant {
    /// Create a circulant matrix from a real-valued generator.
    ///
    /// Args:
    ///     generator: The first row of the matrix (generator vector).
    ///
    /// Returns:
    ///     A new Circulant matrix.
    ///
    /// Raises:
    ///     ValueError: If the generator is empty.
    #[staticmethod]
    fn from_real(generator: Vec<f64>) -> PyResult<Self> {
        let inner = Circulant::from_real(generator)?;
        Ok(Self { inner })
    }

    /// Multiply the circulant matrix by a real vector.
    ///
    /// Args:
    ///     x: The vector to multiply.
    ///
    /// Returns:
    ///     The result as a list of (real, imag) tuples.
    ///
    /// Raises:
    ///     ValueError: If dimensions don't match.
    fn mul_vec(&self, x: Vec<f64>) -> PyResult<Vec<(f64, f64)>> {
        let result = self.inner.mul_vec_real(&x)?;
        Ok(result.into_iter().map(|c| (c.re, c.im)).collect())
    }

    /// Get the eigenvalues of the matrix.
    ///
    /// Returns:
    ///     The eigenvalues as a list of (real, imag) tuples.
    fn eigenvalues(&self) -> Vec<(f64, f64)> {
        self.inner
            .eigenvalues()
            .into_iter()
            .map(|c| (c.re, c.im))
            .collect()
    }

    /// Get the size of the matrix.
    fn size(&self) -> usize {
        self.inner.size()
    }

    /// Precompute FFT spectra for faster repeated multiplications.
    fn precompute(&mut self) {
        self.inner.precompute();
    }

    /// Get the generator (first row) of the matrix.
    fn generator(&self) -> Vec<(f64, f64)> {
        self.inner
            .generator()
            .iter()
            .map(|c| (c.re, c.im))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("Circulant(size={})", self.inner.size())
    }
}
