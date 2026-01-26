//! High-performance block-circulant matrix operations using FFT.
//!
//! This library exploits the fact that circulant matrices can be diagonalized
//! by the DFT matrix, enabling O(N log N) matrix-vector multiplication instead
//! of the naive O(N²) approach.
//!
//! # Overview
//!
//! A circulant matrix is a special type of Toeplitz matrix where each row is
//! a cyclic shift of the previous row. This structure allows for efficient
//! computation using the Fast Fourier Transform (FFT).
//!
//! # Features
//!
//! - `std` (default): Enable standard library features
//! - `physics` (default): Enable quantum physics module (quantum walks, coins, Hamiltonians)
//! - `parallel` (default): Enable parallel computation with rayon
//! - `serde` (default): Enable serialization with serde/bincode
//! - `vision`: Enable image processing with BCCB filters
//! - `visualize`: Enable visualization with plotters (base feature)
//! - `visualize-svg`: Enable SVG output for visualization
//! - `visualize-bitmap`: Enable bitmap output for visualization
//! - `python`: Enable Python bindings via PyO3
//!
//! # Example
//!
//! ```
//! use circulant_rs::prelude::*;
//! use num_complex::Complex;
//!
//! // Create a circulant matrix from a real generator
//! let c = Circulant::from_real(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
//!
//! // Multiply by a vector using FFT (O(N log N))
//! let x: Vec<Complex<f64>> = vec![
//!     Complex::new(1.0, 0.0),
//!     Complex::new(0.0, 0.0),
//!     Complex::new(1.0, 0.0),
//!     Complex::new(0.0, 0.0),
//! ];
//! let result = c.mul_vec(&x).unwrap();
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

pub mod core;
pub mod error;
pub mod fft;
pub mod traits;

#[cfg(feature = "physics")]
pub mod physics;

#[cfg(feature = "vision")]
pub mod vision;

#[cfg(feature = "visualize")]
pub mod visualize;

#[cfg(feature = "python")]
mod python;

pub mod prelude;

// Re-export commonly used types at the crate root
pub use crate::core::{BlockCirculant, Circulant};
pub use crate::error::{CirculantError, Result};
pub use crate::fft::{FftBackend, RustFftBackend};
pub use crate::traits::{BlockOps, CirculantOps, Scalar};
