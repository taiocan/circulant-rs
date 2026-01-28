// @module: crate::fft
// @status: stable
// @owner: code_expert
// @feature: none
// @depends: [crate::fft::backend, crate::fft::rustfft_backend]
// @tests: [none]

//! FFT backend implementations.
//!
//! This module provides the FFT functionality used for efficient circulant matrix operations.

mod backend;
mod rustfft_backend;

pub use backend::FftBackend;
pub use rustfft_backend::RustFftBackend;
