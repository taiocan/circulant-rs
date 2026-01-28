// @module: crate::vision
// @status: stable
// @owner: code_expert
// @feature: vision
// @depends: [crate::vision::filter, crate::vision::kernel]
// @tests: [none]

//! Image processing with block-circulant matrices.
//!
//! This module provides FFT-accelerated image filtering using BCCB (Block Circulant
//! with Circulant Blocks) matrices. Convolution operations that would normally be
//! O(N²M²) become O(NM log NM) using FFT.
//!
//! # Features
//!
//! Enable the `vision` feature to use this module:
//! ```toml
//! [dependencies]
//! circulant-rs = { version = "0.2", features = ["vision"] }
//! ```

mod filter;
mod kernel;

pub use filter::BCCBFilter;
pub use kernel::Kernel;
