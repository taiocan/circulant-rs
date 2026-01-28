// @module: crate::core::tensor
// @status: stable
// @owner: code_expert
// @feature: none
// @depends: [crate::error, crate::fft, crate::traits, ndarray]
// @tests: [unit, property]
// @version: 1.0.0
// @event: DE-2026-002

//! N-dimensional circulant tensor implementation.
//!
//! This module provides `CirculantTensor<T, D>`, a generic circulant tensor type
//! that supports arbitrary dimension D with compile-time dimension checking.
//!
//! # Mathematical Background
//!
//! An N-D circulant tensor C of shape [n₁, n₂, ..., n_D] is defined by a generator
//! tensor G of the same shape, where:
//!
//! ```text
//! C[i₁, i₂, ..., i_D, j₁, j₂, ..., j_D] = G[(i₁-j₁) mod n₁, ..., (i_D-j_D) mod n_D]
//! ```
//!
//! The key property is that N-D circulant tensors can be diagonalized by the N-D DFT,
//! enabling O(N log N) tensor-vector multiplication where N = ∏nᵢ.
//!
//! # Example
//!
//! ```
//! use circulant_rs::core::CirculantTensor;
//! use circulant_rs::TensorOps;
//! use ndarray::ArrayD;
//! use num_complex::Complex;
//!
//! // Create a 3D circulant tensor
//! let shape = vec![4, 4, 4];
//! let generator = ArrayD::from_elem(shape.clone(), Complex::new(0.0, 0.0));
//! let tensor = CirculantTensor::<f64, 3>::new(generator).unwrap();
//! assert_eq!(tensor.total_size(), 64);
//! ```

use crate::error::{CirculantError, Result};
use crate::fft::{FftBackend, RustFftBackend};
use crate::traits::{Scalar, TensorOps};
use ndarray::{ArrayD, Axis, IxDyn};
use num_complex::Complex;
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// An N-dimensional circulant tensor with compile-time dimension checking.
///
/// A circulant tensor is defined by its generator tensor G, where the full
/// tensor C has the property:
///
/// ```text
/// C[i, j] = G[(i - j) mod shape]  (multi-index notation)
/// ```
///
/// This structure arises in N-D convolution with periodic boundary conditions
/// and can be efficiently multiplied using the N-D FFT in O(N log N) time.
///
/// # Type Parameters
///
/// - `T`: The scalar type (f32 or f64)
/// - `D`: The number of dimensions (const generic)
///
/// # Examples
///
/// ```
/// use circulant_rs::core::CirculantTensor;
/// use circulant_rs::TensorOps;
/// use ndarray::ArrayD;
/// use num_complex::Complex;
///
/// // Create a 2D circulant tensor (equivalent to BlockCirculant)
/// let gen = ArrayD::from_shape_vec(
///     vec![3, 3],
///     (0..9).map(|x| Complex::new(x as f64, 0.0)).collect()
/// ).unwrap();
///
/// let tensor = CirculantTensor::<f64, 2>::new(gen).unwrap();
/// assert_eq!(tensor.shape(), [3, 3]);
/// ```
#[derive(Clone)]
pub struct CirculantTensor<T: Scalar + rustfft::FftNum, const D: usize> {
    /// The generator tensor (defines the circulant structure).
    generator: ArrayD<Complex<T>>,

    /// Shape as a fixed-size array for compile-time dimension checking.
    shape: [usize; D],

    /// Cached N-D spectrum (FFT of generator) for repeated operations.
    cached_spectrum: Option<ArrayD<Complex<T>>>,

    /// FFT backends for each axis (one per unique dimension size).
    fft_backends: Vec<Arc<RustFftBackend<T>>>,
}

impl<T: Scalar + rustfft::FftNum, const D: usize> CirculantTensor<T, D> {
    /// Create a new circulant tensor from a complex generator.
    ///
    /// # Arguments
    ///
    /// * `generator` - The generator tensor defining the circulant structure.
    ///   Shape must have exactly D dimensions.
    ///
    /// # Errors
    ///
    /// Returns `InvalidTensorDimension` if generator has wrong number of dimensions.
    /// Returns `EmptyGenerator` if any dimension is zero.
    /// Returns `InvalidFftSize` if FFT backend creation fails.
    ///
    /// # Example
    ///
    /// ```
    /// use circulant_rs::core::CirculantTensor;
    /// use ndarray::ArrayD;
    /// use num_complex::Complex;
    ///
    /// let gen = ArrayD::from_shape_vec(
    ///     vec![4, 4, 4],
    ///     (0..64).map(|x| Complex::new(x as f64, 0.0)).collect()
    /// ).unwrap();
    ///
    /// let tensor = CirculantTensor::<f64, 3>::new(gen).unwrap();
    /// ```
    pub fn new(generator: ArrayD<Complex<T>>) -> Result<Self> {
        let gen_shape = generator.shape();

        // Verify dimension count matches const generic D
        if gen_shape.len() != D {
            return Err(CirculantError::InvalidTensorDimension {
                expected: D,
                got: gen_shape.len(),
            });
        }

        // Verify no dimension is zero
        for &dim in gen_shape.iter() {
            if dim == 0 {
                return Err(CirculantError::InvalidTensorShape {
                    expected: vec![1; D], // Non-zero dimensions expected
                    got: gen_shape.to_vec(),
                });
            }
            // Also verify dimension is reasonable (prevent overflow in FFT)
            if dim > 1 << 30 {
                return Err(CirculantError::InvalidTensorShape {
                    expected: vec![1 << 30; D],
                    got: gen_shape.to_vec(),
                });
            }
        }

        // Convert to fixed-size array
        let mut shape = [0usize; D];
        for (i, &dim) in gen_shape.iter().enumerate() {
            shape[i] = dim;
        }

        // Create FFT backends for each axis
        // Collect unique sizes and create one backend per unique size
        let mut fft_backends = Vec::with_capacity(D);
        for &dim in &shape {
            // Check if we already have a backend for this size
            let existing = fft_backends
                .iter()
                .find(|b: &&Arc<RustFftBackend<T>>| b.size() == dim);
            if existing.is_some() {
                // Reuse existing backend
                let backend = fft_backends
                    .iter()
                    .find(|b| b.size() == dim)
                    .cloned()
                    .expect("Backend should exist");
                fft_backends.push(backend);
            } else {
                // Create new backend for this size
                let backend = Arc::new(RustFftBackend::new(dim)?);
                fft_backends.push(backend);
            }
        }
        // We need exactly D backends (one per axis), maintaining order
        // Let's rebuild this more carefully
        fft_backends.clear();
        let mut backend_cache: std::collections::HashMap<usize, Arc<RustFftBackend<T>>> =
            std::collections::HashMap::new();
        for &dim in &shape {
            if let Some(backend) = backend_cache.get(&dim) {
                fft_backends.push(backend.clone());
            } else {
                let backend = Arc::new(RustFftBackend::new(dim)?);
                backend_cache.insert(dim, backend.clone());
                fft_backends.push(backend);
            }
        }

        Ok(Self {
            generator,
            shape,
            cached_spectrum: None,
            fft_backends,
        })
    }

    /// Create a new circulant tensor from a real generator.
    ///
    /// Converts the real tensor to complex with zero imaginary parts.
    ///
    /// # Arguments
    ///
    /// * `generator` - The real-valued generator tensor.
    ///
    /// # Errors
    ///
    /// Same as `new()`.
    pub fn from_real(generator: ArrayD<T>) -> Result<Self> {
        let complex_gen = generator.mapv(|x| Complex::new(x, T::zero()));
        Self::new(complex_gen)
    }

    /// Create a circulant tensor from a kernel with zero-padding.
    ///
    /// This is useful for convolution where the kernel is smaller than the
    /// output size. The kernel is embedded in a larger tensor with zeros.
    ///
    /// # Arguments
    ///
    /// * `kernel` - The convolution kernel (can be smaller than output shape).
    /// * `output_shape` - The desired output shape (must have D dimensions).
    ///
    /// # Errors
    ///
    /// Returns `InvalidTensorDimension` if shapes don't have D dimensions.
    /// Returns `InvalidTensorShape` if kernel is larger than output shape in any dimension.
    pub fn from_kernel(kernel: ArrayD<Complex<T>>, output_shape: [usize; D]) -> Result<Self> {
        let kernel_shape = kernel.shape();

        // Verify dimension counts
        if kernel_shape.len() != D {
            return Err(CirculantError::InvalidTensorDimension {
                expected: D,
                got: kernel_shape.len(),
            });
        }

        // Verify kernel fits in output shape
        for (axis, (&k_dim, &o_dim)) in kernel_shape.iter().zip(output_shape.iter()).enumerate() {
            if k_dim > o_dim {
                return Err(CirculantError::InvalidKernel(format!(
                    "kernel dimension {} ({}) exceeds output dimension ({})",
                    axis, k_dim, o_dim
                )));
            }
        }

        // Create zero-padded generator
        let mut padded = ArrayD::from_elem(IxDyn(&output_shape), Complex::new(T::zero(), T::zero()));

        // Copy kernel into padded array
        // Use ndarray's slice assignment
        let mut slice_info: Vec<ndarray::SliceInfoElem> = Vec::with_capacity(D);
        for &k_dim in kernel_shape {
            slice_info.push(ndarray::SliceInfoElem::Slice {
                start: 0,
                end: Some(k_dim as isize),
                step: 1,
            });
        }

        // Manual copy for safety (ndarray slice API can be tricky)
        Self::copy_kernel_to_padded(&kernel, &mut padded, kernel_shape);

        Self::new(padded)
    }

    /// Helper to copy kernel into padded array.
    fn copy_kernel_to_padded(
        kernel: &ArrayD<Complex<T>>,
        padded: &mut ArrayD<Complex<T>>,
        kernel_shape: &[usize],
    ) {
        // Recursive copy using indices
        let total_kernel_elements: usize = kernel_shape.iter().product();
        for flat_idx in 0..total_kernel_elements {
            // Convert flat index to multi-index for kernel
            let mut multi_idx: Vec<usize> = vec![0; kernel_shape.len()];
            let mut remaining = flat_idx;
            for (i, &dim) in kernel_shape.iter().enumerate().rev() {
                multi_idx[i] = remaining % dim;
                remaining /= dim;
            }

            // Copy element
            let kernel_idx = IxDyn(&multi_idx);
            let padded_idx = IxDyn(&multi_idx);
            padded[padded_idx.clone()] = kernel[kernel_idx];
        }
    }

    /// Get a reference to the generator tensor.
    pub fn generator(&self) -> &ArrayD<Complex<T>> {
        &self.generator
    }

    /// Precompute and cache the N-D spectrum for faster repeated multiplications.
    ///
    /// This computes the N-D FFT of the generator and caches the result.
    /// Subsequent multiplications will use the cached spectrum instead of
    /// recomputing it each time.
    pub fn precompute(&mut self) {
        if self.cached_spectrum.is_some() {
            return;
        }

        self.cached_spectrum = Some(self.compute_spectrum());
    }

    /// Clear the cached spectrum.
    pub fn clear_cache(&mut self) {
        self.cached_spectrum = None;
    }

    /// Check if the spectrum is cached.
    pub fn is_precomputed(&self) -> bool {
        self.cached_spectrum.is_some()
    }

    /// Compute the N-D spectrum (N-D FFT of generator).
    ///
    /// For circulant tensor multiplication: spectrum = FFT_ND(generator)
    /// Uses the convolution semantics: y = IFFT(FFT(g) * FFT(x))
    fn compute_spectrum(&self) -> ArrayD<Complex<T>> {
        let mut spectrum = self.generator.clone();

        // Apply FFT along each axis sequentially (separable N-D FFT)
        for axis in 0..D {
            self.fft_axis_forward(&mut spectrum, axis);
        }

        spectrum
    }

    /// Apply 1D FFT along a specific axis.
    fn fft_axis_forward(&self, data: &mut ArrayD<Complex<T>>, axis: usize) {
        let backend = &self.fft_backends[axis];

        // Process each "lane" along the specified axis
        for mut lane in data.lanes_mut(Axis(axis)) {
            let mut buffer: Vec<Complex<T>> = lane.iter().copied().collect();
            backend.fft_forward(&mut buffer);
            for (dest, src) in lane.iter_mut().zip(buffer.into_iter()) {
                *dest = src;
            }
        }
    }

    /// Apply 1D inverse FFT along a specific axis.
    fn fft_axis_inverse(&self, data: &mut ArrayD<Complex<T>>, axis: usize) {
        let backend = &self.fft_backends[axis];

        // Process each "lane" along the specified axis
        for mut lane in data.lanes_mut(Axis(axis)) {
            let mut buffer: Vec<Complex<T>> = lane.iter().copied().collect();
            backend.fft_inverse(&mut buffer);
            for (dest, src) in lane.iter_mut().zip(buffer.into_iter()) {
                *dest = src;
            }
        }
    }

    /// Apply N-D FFT to data.
    fn fft_nd_forward(&self, data: &mut ArrayD<Complex<T>>) {
        for axis in 0..D {
            self.fft_axis_forward(data, axis);
        }
    }

    /// Apply N-D inverse FFT to data.
    fn fft_nd_inverse(&self, data: &mut ArrayD<Complex<T>>) {
        for axis in 0..D {
            self.fft_axis_inverse(data, axis);
        }
    }

    /// Ensure FFT backends are initialized (useful after deserialization).
    ///
    /// # Errors
    ///
    /// Returns `InvalidFftSize` if initialization fails.
    #[allow(dead_code)]
    pub fn ensure_fft(&mut self) -> Result<()> {
        if self.fft_backends.is_empty() {
            let mut backend_cache: std::collections::HashMap<usize, Arc<RustFftBackend<T>>> =
                std::collections::HashMap::new();
            for &dim in &self.shape {
                if let Some(backend) = backend_cache.get(&dim) {
                    self.fft_backends.push(backend.clone());
                } else {
                    let backend = Arc::new(RustFftBackend::new(dim)?);
                    backend_cache.insert(dim, backend.clone());
                    self.fft_backends.push(backend);
                }
            }
        }
        Ok(())
    }

    /// Get element at multi-index (i, j) where both are D-dimensional.
    ///
    /// Uses the circulant property: C[i,j] = G[(i-j) mod shape]
    pub fn get(&self, i: &[usize; D], j: &[usize; D]) -> Complex<T> {
        let mut idx = [0usize; D];
        for k in 0..D {
            idx[k] = (i[k] + self.shape[k] - j[k]) % self.shape[k];
        }
        self.generator[IxDyn(&idx)]
    }
}

impl<T: Scalar + rustfft::FftNum, const D: usize> TensorOps<T, D> for CirculantTensor<T, D> {
    fn mul_tensor(&self, x: &ArrayD<Complex<T>>) -> Result<ArrayD<Complex<T>>> {
        let x_shape = x.shape();

        // Verify shape matches
        if x_shape.len() != D {
            return Err(CirculantError::InvalidTensorDimension {
                expected: D,
                got: x_shape.len(),
            });
        }

        for (&x_dim, &g_dim) in x_shape.iter().zip(self.shape.iter()) {
            if x_dim != g_dim {
                return Err(CirculantError::InvalidTensorShape {
                    expected: self.shape.to_vec(),
                    got: x_shape.to_vec(),
                });
            }
        }

        // Get spectrum (compute or use cached)
        let spectrum = if let Some(ref cached) = self.cached_spectrum {
            cached.clone()
        } else {
            self.compute_spectrum()
        };

        // Forward N-D FFT of input
        let mut x_fft = x.clone();
        self.fft_nd_forward(&mut x_fft);

        // Element-wise multiplication in frequency domain
        let mut result_fft = &x_fft * &spectrum;

        // Inverse N-D FFT
        self.fft_nd_inverse(&mut result_fft);

        Ok(result_fft)
    }

    fn mul_vec(&self, x: &[Complex<T>]) -> Result<Vec<Complex<T>>> {
        let total = self.total_size();

        if x.len() != total {
            return Err(CirculantError::DimensionMismatch {
                expected: total,
                got: x.len(),
            });
        }

        // Reshape to N-D array
        let x_nd = ArrayD::from_shape_vec(IxDyn(&self.shape), x.to_vec()).map_err(|_| {
            CirculantError::ReshapeFailed(format!(
                "failed to reshape vector of length {} to shape {:?}",
                x.len(),
                self.shape
            ))
        })?;

        // Multiply
        let result_nd = self.mul_tensor(&x_nd)?;

        // Flatten back to vector
        let (vec, _offset) = result_nd.into_raw_vec_and_offset();
        Ok(vec)
    }

    fn eigenvalues_nd(&self) -> ArrayD<Complex<T>> {
        if let Some(ref cached) = self.cached_spectrum {
            cached.clone()
        } else {
            self.compute_spectrum()
        }
    }

    fn shape(&self) -> [usize; D] {
        self.shape
    }

    fn total_size(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Parallel N-D FFT implementation for large tensors.
#[cfg(feature = "parallel")]
impl<T: Scalar + rustfft::FftNum + Send + Sync, const D: usize> CirculantTensor<T, D> {
    /// Parallelism threshold: use parallel FFT for tensors with more elements than this.
    const PARALLEL_THRESHOLD: usize = 32 * 32 * 32; // ~32K elements

    /// Multiply using parallel N-D FFT for large tensors.
    ///
    /// Automatically chooses between sequential and parallel based on tensor size.
    pub fn mul_tensor_auto(&self, x: &ArrayD<Complex<T>>) -> Result<ArrayD<Complex<T>>> {
        if self.total_size() > Self::PARALLEL_THRESHOLD {
            self.mul_tensor_parallel(x)
        } else {
            self.mul_tensor(x)
        }
    }

    /// Multiply using parallel N-D FFT.
    ///
    /// Each axis FFT is parallelized using rayon.
    pub fn mul_tensor_parallel(&self, x: &ArrayD<Complex<T>>) -> Result<ArrayD<Complex<T>>> {
        let x_shape = x.shape();

        // Verify shape matches
        if x_shape.len() != D {
            return Err(CirculantError::InvalidTensorDimension {
                expected: D,
                got: x_shape.len(),
            });
        }

        for (&x_dim, &g_dim) in x_shape.iter().zip(self.shape.iter()) {
            if x_dim != g_dim {
                return Err(CirculantError::InvalidTensorShape {
                    expected: self.shape.to_vec(),
                    got: x_shape.to_vec(),
                });
            }
        }

        // Get spectrum (compute or use cached)
        let spectrum = if let Some(ref cached) = self.cached_spectrum {
            cached.clone()
        } else {
            self.compute_spectrum_parallel()
        };

        // Forward N-D FFT of input (parallel)
        let mut x_fft = x.clone();
        self.fft_nd_forward_parallel(&mut x_fft);

        // Element-wise multiplication in frequency domain
        let mut result_fft = &x_fft * &spectrum;

        // Inverse N-D FFT (parallel)
        self.fft_nd_inverse_parallel(&mut result_fft);

        Ok(result_fft)
    }

    /// Compute spectrum using parallel FFT.
    fn compute_spectrum_parallel(&self) -> ArrayD<Complex<T>> {
        let mut spectrum = self.generator.clone();
        self.fft_nd_forward_parallel(&mut spectrum);
        spectrum
    }

    /// Apply N-D FFT with parallel axis processing.
    fn fft_nd_forward_parallel(&self, data: &mut ArrayD<Complex<T>>) {
        for axis in 0..D {
            self.fft_axis_forward_parallel(data, axis);
        }
    }

    /// Apply N-D inverse FFT with parallel axis processing.
    fn fft_nd_inverse_parallel(&self, data: &mut ArrayD<Complex<T>>) {
        for axis in 0..D {
            self.fft_axis_inverse_parallel(data, axis);
        }
    }

    /// Apply 1D FFT along axis with parallel lane processing.
    ///
    /// Uses a gather-scatter approach: extract lanes, process in parallel, scatter back.
    fn fft_axis_forward_parallel(&self, data: &mut ArrayD<Complex<T>>, axis: usize) {
        let backend = &self.fft_backends[axis];
        let size = self.shape[axis];
        let n_lanes = data.len() / size;

        // Gather: extract all lanes into separate vectors
        let mut lanes: Vec<Vec<Complex<T>>> = Vec::with_capacity(n_lanes);
        for lane in data.lanes(Axis(axis)) {
            lanes.push(lane.iter().copied().collect());
        }

        // Process lanes in parallel
        lanes.par_iter_mut().for_each(|lane| {
            backend.fft_forward(lane);
        });

        // Scatter: write results back to data
        for (lane_data, mut dest_lane) in lanes.into_iter().zip(data.lanes_mut(Axis(axis))) {
            for (src, dest) in lane_data.into_iter().zip(dest_lane.iter_mut()) {
                *dest = src;
            }
        }
    }

    /// Apply 1D inverse FFT along axis with parallel lane processing.
    fn fft_axis_inverse_parallel(&self, data: &mut ArrayD<Complex<T>>, axis: usize) {
        let backend = &self.fft_backends[axis];
        let size = self.shape[axis];
        let n_lanes = data.len() / size;

        // Gather
        let mut lanes: Vec<Vec<Complex<T>>> = Vec::with_capacity(n_lanes);
        for lane in data.lanes(Axis(axis)) {
            lanes.push(lane.iter().copied().collect());
        }

        // Process lanes in parallel
        lanes.par_iter_mut().for_each(|lane| {
            backend.fft_inverse(lane);
        });

        // Scatter
        for (lane_data, mut dest_lane) in lanes.into_iter().zip(data.lanes_mut(Axis(axis))) {
            for (src, dest) in lane_data.into_iter().zip(dest_lane.iter_mut()) {
                *dest = src;
            }
        }
    }
}

/// Naive O(N^D) N-D circulant tensor-vector multiplication (for testing).
///
/// This function computes the exact matrix-vector product using the circulant
/// property directly, without FFT. It's used as a reference for testing the
/// FFT-based implementation.
///
/// # Complexity
///
/// O(N²) where N = total elements = ∏nᵢ
///
/// # Example
///
/// ```
/// use circulant_rs::core::naive_tensor_mul;
/// use ndarray::ArrayD;
/// use num_complex::Complex;
///
/// let gen = ArrayD::from_shape_vec(
///     vec![2, 2],
///     vec![
///         Complex::new(1.0, 0.0),
///         Complex::new(2.0, 0.0),
///         Complex::new(3.0, 0.0),
///         Complex::new(4.0, 0.0),
///     ]
/// ).unwrap();
///
/// let x = ArrayD::from_shape_vec(
///     vec![2, 2],
///     vec![
///         Complex::new(1.0, 0.0),
///         Complex::new(0.0, 0.0),
///         Complex::new(0.0, 0.0),
///         Complex::new(0.0, 0.0),
///     ]
/// ).unwrap();
///
/// let result = naive_tensor_mul(&gen, &x);
/// ```
pub fn naive_tensor_mul<T: Scalar>(
    generator: &ArrayD<Complex<T>>,
    x: &ArrayD<Complex<T>>,
) -> ArrayD<Complex<T>> {
    let shape = generator.shape();
    let ndim = shape.len();
    let total: usize = shape.iter().product();

    let mut result = ArrayD::from_elem(shape, Complex::new(T::zero(), T::zero()));

    // Iterate over all output positions
    for out_flat in 0..total {
        // Convert flat index to multi-index
        let out_idx = flat_to_multi(out_flat, shape);

        let mut sum = Complex::new(T::zero(), T::zero());

        // Sum over all input positions
        for in_flat in 0..total {
            let in_idx = flat_to_multi(in_flat, shape);

            // Compute generator index: g_idx = (out_idx - in_idx) mod shape
            let mut g_idx: Vec<usize> = Vec::with_capacity(ndim);
            for k in 0..ndim {
                g_idx.push((out_idx[k] + shape[k] - in_idx[k]) % shape[k]);
            }

            sum += generator[IxDyn(&g_idx)] * x[IxDyn(&in_idx)];
        }

        result[IxDyn(&out_idx)] = sum;
    }

    result
}

/// Convert flat index to multi-index.
fn flat_to_multi(flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut multi = vec![0; shape.len()];
    let mut remaining = flat;
    for (i, &dim) in shape.iter().enumerate().rev() {
        multi[i] = remaining % dim;
        remaining /= dim;
    }
    multi
}

// Type aliases for common dimensions
/// 1D circulant tensor (equivalent to Circulant).
pub type Circulant1D<T> = CirculantTensor<T, 1>;
/// 2D circulant tensor (equivalent to BlockCirculant/BCCB).
pub type Circulant2D<T> = CirculantTensor<T, 2>;
/// 3D circulant tensor.
pub type Circulant3D<T> = CirculantTensor<T, 3>;
/// 4D circulant tensor.
pub type Circulant4D<T> = CirculantTensor<T, 4>;

#[cfg(test)]
mod tests {
    // @test_for: [crate::core::tensor]
    // @math_verified: false
    // @verified_by: math_expert
    // @properties_checked: [fft_roundtrip, naive_comparison, identity, linearity]
    // @version: 1.0.0
    // @event: DE-2026-002

    use super::*;
    use approx::assert_relative_eq;

    fn complex(re: f64, im: f64) -> Complex<f64> {
        Complex::new(re, im)
    }

    #[test]
    fn test_tensor_creation_2d() {
        let gen = ArrayD::from_shape_vec(
            vec![3, 3],
            (0..9).map(|x| complex(x as f64, 0.0)).collect(),
        )
        .expect("Failed to create array");

        let tensor = CirculantTensor::<f64, 2>::new(gen).expect("Failed to create tensor");
        assert_eq!(tensor.shape(), [3, 3]);
        assert_eq!(tensor.total_size(), 9);
    }

    #[test]
    fn test_tensor_creation_3d() {
        let gen = ArrayD::from_shape_vec(
            vec![4, 4, 4],
            (0..64).map(|x| complex(x as f64, 0.0)).collect(),
        )
        .expect("Failed to create array");

        let tensor = CirculantTensor::<f64, 3>::new(gen).expect("Failed to create tensor");
        assert_eq!(tensor.shape(), [4, 4, 4]);
        assert_eq!(tensor.total_size(), 64);
    }

    #[test]
    fn test_tensor_from_real() {
        let gen = ArrayD::from_shape_vec(vec![2, 2, 2], (0..8).map(|x| x as f64).collect())
            .expect("Failed to create array");

        let tensor = CirculantTensor::<f64, 3>::from_real(gen).expect("Failed to create tensor");
        assert_eq!(tensor.shape(), [2, 2, 2]);

        // Verify values are complex with zero imaginary
        let g = tensor.generator();
        assert_eq!(g[IxDyn(&[0, 0, 0])], complex(0.0, 0.0));
        assert_eq!(g[IxDyn(&[1, 1, 1])], complex(7.0, 0.0));
    }

    #[test]
    fn test_tensor_dimension_mismatch() {
        let gen = ArrayD::from_shape_vec(vec![3, 3], (0..9).map(|x| complex(x as f64, 0.0)).collect())
            .expect("Failed to create array");

        // Try to create 3D tensor from 2D generator
        let result = CirculantTensor::<f64, 3>::new(gen);
        assert!(matches!(
            result,
            Err(CirculantError::InvalidTensorDimension {
                expected: 3,
                got: 2
            })
        ));
    }

    #[test]
    fn test_tensor_empty_dimension() {
        let gen = ArrayD::from_shape_vec(vec![3, 0], vec![]).expect("Failed to create array");

        let result = CirculantTensor::<f64, 2>::new(gen);
        assert!(matches!(result, Err(CirculantError::InvalidTensorShape { .. })));
    }

    #[test]
    fn test_tensor_from_kernel() {
        let kernel = ArrayD::from_shape_vec(
            vec![2, 2],
            vec![complex(1.0, 0.0), complex(2.0, 0.0), complex(3.0, 0.0), complex(4.0, 0.0)],
        )
        .expect("Failed to create array");

        let tensor = CirculantTensor::<f64, 2>::from_kernel(kernel, [4, 4])
            .expect("Failed to create tensor");
        assert_eq!(tensor.shape(), [4, 4]);

        // Verify kernel is embedded correctly
        let g = tensor.generator();
        assert_eq!(g[IxDyn(&[0, 0])], complex(1.0, 0.0));
        assert_eq!(g[IxDyn(&[0, 1])], complex(2.0, 0.0));
        assert_eq!(g[IxDyn(&[1, 0])], complex(3.0, 0.0));
        assert_eq!(g[IxDyn(&[1, 1])], complex(4.0, 0.0));
        // Rest should be zeros
        assert_eq!(g[IxDyn(&[2, 2])], complex(0.0, 0.0));
        assert_eq!(g[IxDyn(&[3, 3])], complex(0.0, 0.0));
    }

    #[test]
    fn test_2d_multiply_matches_naive() {
        let gen = ArrayD::from_shape_vec(
            vec![4, 4],
            vec![
                complex(1.0, 0.0), complex(2.0, 0.0), complex(0.0, 0.0), complex(0.0, 0.0),
                complex(3.0, 0.0), complex(4.0, 0.0), complex(0.0, 0.0), complex(0.0, 0.0),
                complex(0.0, 0.0), complex(0.0, 0.0), complex(0.0, 0.0), complex(0.0, 0.0),
                complex(0.0, 0.0), complex(0.0, 0.0), complex(0.0, 0.0), complex(0.0, 0.0),
            ],
        )
        .expect("Failed to create array");

        let x = ArrayD::from_shape_vec(
            vec![4, 4],
            vec![
                complex(1.0, 0.0), complex(0.0, 0.0), complex(0.0, 0.0), complex(0.0, 0.0),
                complex(0.0, 0.0), complex(1.0, 0.0), complex(0.0, 0.0), complex(0.0, 0.0),
                complex(0.0, 0.0), complex(0.0, 0.0), complex(0.0, 0.0), complex(0.0, 0.0),
                complex(0.0, 0.0), complex(0.0, 0.0), complex(0.0, 0.0), complex(0.0, 0.0),
            ],
        )
        .expect("Failed to create array");

        let tensor = CirculantTensor::<f64, 2>::new(gen.clone()).expect("Failed to create tensor");
        let fft_result = tensor.mul_tensor(&x).expect("FFT multiply failed");
        let naive_result = naive_tensor_mul(&gen, &x);

        for i in 0..4 {
            for j in 0..4 {
                let idx = IxDyn(&[i, j]);
                assert_relative_eq!(fft_result[idx.clone()].re, naive_result[idx.clone()].re, epsilon = 1e-10);
                assert_relative_eq!(fft_result[idx.clone()].im, naive_result[idx].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_3d_multiply_matches_naive() {
        // Small 3D tensor for tractable naive computation
        let gen = ArrayD::from_shape_vec(
            vec![3, 3, 3],
            (0..27)
                .map(|x| complex((x as f64) * 0.1, (x as f64) * 0.05))
                .collect(),
        )
        .expect("Failed to create array");

        let x = ArrayD::from_shape_vec(
            vec![3, 3, 3],
            (0..27).map(|x| complex(1.0 / (x + 1) as f64, 0.0)).collect(),
        )
        .expect("Failed to create array");

        let tensor = CirculantTensor::<f64, 3>::new(gen.clone()).expect("Failed to create tensor");
        let fft_result = tensor.mul_tensor(&x).expect("FFT multiply failed");
        let naive_result = naive_tensor_mul(&gen, &x);

        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let idx = IxDyn(&[i, j, k]);
                    assert_relative_eq!(
                        fft_result[idx.clone()].re,
                        naive_result[idx.clone()].re,
                        epsilon = 1e-9
                    );
                    assert_relative_eq!(
                        fft_result[idx.clone()].im,
                        naive_result[idx].im,
                        epsilon = 1e-9
                    );
                }
            }
        }
    }

    #[test]
    fn test_identity_tensor_2d() {
        // Identity: generator with 1 at origin, rest zeros
        let mut gen = ArrayD::from_elem(vec![4, 4], complex(0.0, 0.0));
        gen[IxDyn(&[0, 0])] = complex(1.0, 0.0);

        let x = ArrayD::from_shape_vec(
            vec![4, 4],
            (0..16).map(|i| complex(i as f64, (i as f64) * 0.5)).collect(),
        )
        .expect("Failed to create array");

        let tensor = CirculantTensor::<f64, 2>::new(gen).expect("Failed to create tensor");
        let result = tensor.mul_tensor(&x).expect("Multiply failed");

        for i in 0..4 {
            for j in 0..4 {
                let idx = IxDyn(&[i, j]);
                assert_relative_eq!(result[idx.clone()].re, x[idx.clone()].re, epsilon = 1e-10);
                assert_relative_eq!(result[idx.clone()].im, x[idx].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_identity_tensor_3d() {
        // Identity for 3D
        let mut gen = ArrayD::from_elem(vec![4, 4, 4], complex(0.0, 0.0));
        gen[IxDyn(&[0, 0, 0])] = complex(1.0, 0.0);

        let x = ArrayD::from_shape_vec(
            vec![4, 4, 4],
            (0..64).map(|i| complex(i as f64, 0.0)).collect(),
        )
        .expect("Failed to create array");

        let tensor = CirculantTensor::<f64, 3>::new(gen).expect("Failed to create tensor");
        let result = tensor.mul_tensor(&x).expect("Multiply failed");

        for flat in 0..64 {
            let idx: Vec<usize> = flat_to_multi(flat, &[4, 4, 4]);
            let idx = IxDyn(&idx);
            assert_relative_eq!(result[idx.clone()].re, x[idx.clone()].re, epsilon = 1e-10);
            assert_relative_eq!(result[idx.clone()].im, x[idx].im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mul_vec_2d() {
        let gen = ArrayD::from_shape_vec(
            vec![2, 2],
            vec![complex(1.0, 0.0), complex(2.0, 0.0), complex(3.0, 0.0), complex(4.0, 0.0)],
        )
        .expect("Failed to create array");

        let x_vec = vec![
            complex(1.0, 0.0),
            complex(0.0, 0.0),
            complex(0.0, 0.0),
            complex(1.0, 0.0),
        ];

        let tensor = CirculantTensor::<f64, 2>::new(gen.clone()).expect("Failed to create tensor");
        let result_vec = tensor.mul_vec(&x_vec).expect("mul_vec failed");

        // Compare with tensor multiply
        let x_tensor = ArrayD::from_shape_vec(vec![2, 2], x_vec.clone()).expect("reshape failed");
        let result_tensor = tensor.mul_tensor(&x_tensor).expect("mul_tensor failed");

        for (v, t) in result_vec.iter().zip(result_tensor.iter()) {
            assert_relative_eq!(v.re, t.re, epsilon = 1e-10);
            assert_relative_eq!(v.im, t.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_linearity_2d() {
        let gen = ArrayD::from_shape_vec(
            vec![3, 3],
            (0..9).map(|x| complex(x as f64, 0.0)).collect(),
        )
        .expect("Failed to create array");

        let tensor = CirculantTensor::<f64, 2>::new(gen).expect("Failed to create tensor");

        let x = ArrayD::from_shape_vec(vec![3, 3], (0..9).map(|i| complex(i as f64, 0.0)).collect())
            .expect("Failed to create array");
        let y = ArrayD::from_shape_vec(
            vec![3, 3],
            (0..9).map(|i| complex((i + 1) as f64, 0.5)).collect(),
        )
        .expect("Failed to create array");

        let alpha = complex(2.5, 0.0);
        let beta = complex(1.5, 0.0);

        // C(αx + βy)
        let ax: ArrayD<Complex<f64>> = x.mapv(|v| alpha * v);
        let by: ArrayD<Complex<f64>> = y.mapv(|v| beta * v);
        let sum: ArrayD<Complex<f64>> = &ax + &by;
        let c_sum = tensor.mul_tensor(&sum).expect("multiply failed");

        // αC(x) + βC(y)
        let c_x = tensor.mul_tensor(&x).expect("multiply failed");
        let c_y = tensor.mul_tensor(&y).expect("multiply failed");
        let ac_x: ArrayD<Complex<f64>> = c_x.mapv(|v| alpha * v);
        let bc_y: ArrayD<Complex<f64>> = c_y.mapv(|v| beta * v);
        let expected: ArrayD<Complex<f64>> = &ac_x + &bc_y;

        for i in 0..3 {
            for j in 0..3 {
                let idx = IxDyn(&[i, j]);
                assert_relative_eq!(c_sum[idx.clone()].re, expected[idx.clone()].re, epsilon = 1e-9);
                assert_relative_eq!(c_sum[idx.clone()].im, expected[idx].im, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_precompute() {
        let gen = ArrayD::from_shape_vec(
            vec![4, 4],
            (0..16).map(|x| complex(x as f64, 0.0)).collect(),
        )
        .expect("Failed to create array");

        let mut tensor = CirculantTensor::<f64, 2>::new(gen).expect("Failed to create tensor");
        assert!(!tensor.is_precomputed());

        tensor.precompute();
        assert!(tensor.is_precomputed());

        // Multiple precompute calls should be idempotent
        tensor.precompute();
        assert!(tensor.is_precomputed());

        tensor.clear_cache();
        assert!(!tensor.is_precomputed());
    }

    #[test]
    fn test_eigenvalues_count() {
        let gen = ArrayD::from_shape_vec(
            vec![4, 5, 6],
            (0..120).map(|x| complex(x as f64, 0.0)).collect(),
        )
        .expect("Failed to create array");

        let tensor =
            CirculantTensor::<f64, 3>::new(gen).expect("Failed to create tensor");
        let eigenvalues = tensor.eigenvalues_nd();

        assert_eq!(eigenvalues.shape(), &[4, 5, 6]);
        assert_eq!(eigenvalues.len(), 120);
    }

    #[test]
    fn test_get_element() {
        let gen = ArrayD::from_shape_vec(
            vec![3, 3],
            vec![
                complex(0.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0),
                complex(3.0, 0.0), complex(4.0, 0.0), complex(5.0, 0.0),
                complex(6.0, 0.0), complex(7.0, 0.0), complex(8.0, 0.0),
            ],
        )
        .expect("Failed to create array");

        let tensor = CirculantTensor::<f64, 2>::new(gen).expect("Failed to create tensor");

        // C[0,0] = G[0,0] = 0
        assert_eq!(tensor.get(&[0, 0], &[0, 0]).re, 0.0);

        // C[0,1,0,0] = G[(0-0)%3, (1-0)%3] = G[0,1] = 1
        assert_eq!(tensor.get(&[0, 1], &[0, 0]).re, 1.0);

        // C[1,0,0,0] = G[(1-0)%3, (0-0)%3] = G[1,0] = 3
        assert_eq!(tensor.get(&[1, 0], &[0, 0]).re, 3.0);

        // C[0,0,1,1] = G[(0-1+3)%3, (0-1+3)%3] = G[2,2] = 8
        assert_eq!(tensor.get(&[0, 0], &[1, 1]).re, 8.0);
    }

    #[test]
    fn test_dimension_mismatch_mul_tensor() {
        let gen = ArrayD::from_shape_vec(
            vec![4, 4],
            (0..16).map(|x| complex(x as f64, 0.0)).collect(),
        )
        .expect("Failed to create array");

        let tensor = CirculantTensor::<f64, 2>::new(gen).expect("Failed to create tensor");

        let x = ArrayD::from_shape_vec(vec![3, 3], (0..9).map(|x| complex(x as f64, 0.0)).collect())
            .expect("Failed to create array");

        let result = tensor.mul_tensor(&x);
        assert!(matches!(
            result,
            Err(CirculantError::InvalidTensorShape { .. })
        ));
    }

    #[test]
    fn test_dimension_mismatch_mul_vec() {
        let gen = ArrayD::from_shape_vec(
            vec![4, 4],
            (0..16).map(|x| complex(x as f64, 0.0)).collect(),
        )
        .expect("Failed to create array");

        let tensor = CirculantTensor::<f64, 2>::new(gen).expect("Failed to create tensor");

        let x_wrong_size: Vec<Complex<f64>> = (0..10).map(|x| complex(x as f64, 0.0)).collect();

        let result = tensor.mul_vec(&x_wrong_size);
        assert!(matches!(
            result,
            Err(CirculantError::DimensionMismatch {
                expected: 16,
                got: 10
            })
        ));
    }

    #[test]
    fn test_type_aliases() {
        let gen_1d = ArrayD::from_shape_vec(vec![4], (0..4).map(|x| complex(x as f64, 0.0)).collect())
            .expect("Failed to create array");
        let _tensor_1d: Circulant1D<f64> =
            CirculantTensor::new(gen_1d).expect("Failed to create 1D tensor");

        let gen_2d = ArrayD::from_shape_vec(
            vec![4, 4],
            (0..16).map(|x| complex(x as f64, 0.0)).collect(),
        )
        .expect("Failed to create array");
        let _tensor_2d: Circulant2D<f64> =
            CirculantTensor::new(gen_2d).expect("Failed to create 2D tensor");

        let gen_3d = ArrayD::from_shape_vec(
            vec![4, 4, 4],
            (0..64).map(|x| complex(x as f64, 0.0)).collect(),
        )
        .expect("Failed to create array");
        let _tensor_3d: Circulant3D<f64> =
            CirculantTensor::new(gen_3d).expect("Failed to create 3D tensor");

        let gen_4d = ArrayD::from_shape_vec(
            vec![4, 4, 4, 4],
            (0..256).map(|x| complex(x as f64, 0.0)).collect(),
        )
        .expect("Failed to create array");
        let _tensor_4d: Circulant4D<f64> =
            CirculantTensor::new(gen_4d).expect("Failed to create 4D tensor");
    }

    #[cfg(feature = "parallel")]
    mod parallel_tests {
        use super::*;

        #[test]
        fn test_parallel_matches_sequential() {
            let gen = ArrayD::from_shape_vec(
                vec![8, 8, 8],
                (0..512).map(|x| complex(x as f64 * 0.01, 0.0)).collect(),
            )
            .expect("Failed to create array");

            let x = ArrayD::from_shape_vec(
                vec![8, 8, 8],
                (0..512).map(|x| complex(1.0 / (x + 1) as f64, 0.0)).collect(),
            )
            .expect("Failed to create array");

            let tensor =
                CirculantTensor::<f64, 3>::new(gen).expect("Failed to create tensor");

            let sequential_result = tensor.mul_tensor(&x).expect("Sequential multiply failed");
            let parallel_result = tensor
                .mul_tensor_parallel(&x)
                .expect("Parallel multiply failed");

            for (s, p) in sequential_result.iter().zip(parallel_result.iter()) {
                assert_relative_eq!(s.re, p.re, epsilon = 1e-10);
                assert_relative_eq!(s.im, p.im, epsilon = 1e-10);
            }
        }
    }
}
