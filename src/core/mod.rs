// @module: crate::core
// @status: stable
// @owner: code_expert
// @feature: none
// @depends: [crate::core::circulant, crate::core::block_circulant, crate::core::indexing, crate::core::tensor]
// @tests: [none]
// @version: 1.0.0
// @event: DE-2026-002

//! Core circulant matrix and tensor types.
//!
//! This module provides the fundamental circulant matrix types:
//!
//! - [`Circulant`] - 1D circulant matrices (legacy, use `CirculantTensor<T, 1>`)
//! - [`BlockCirculant`] - 2D BCCB matrices (legacy, use `CirculantTensor<T, 2>`)
//! - [`CirculantTensor`] - N-D circulant tensors (unified API)

mod block_circulant;
mod circulant;
mod indexing;
mod tensor;

#[allow(deprecated)]
pub use block_circulant::{naive_bccb_mul, BlockCirculant};
#[allow(deprecated)]
pub use circulant::{naive_circulant_mul, Circulant};
pub use indexing::{circular_get, circular_index};
pub use tensor::{
    naive_tensor_mul, Circulant1D, Circulant2D, Circulant3D, Circulant4D, CirculantTensor,
};
