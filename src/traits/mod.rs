// @module: crate::traits
// @status: stable
// @owner: code_expert
// @feature: none
// @depends: [crate::traits::numeric, crate::traits::ops]
// @tests: [none]

//! Trait definitions for circulant-rs.
//!
//! This module provides the trait bounds and operation traits used throughout the library.

mod numeric;
mod ops;

pub use numeric::{ComplexScalar, Scalar};
pub use ops::{BlockOps, CirculantOps};
