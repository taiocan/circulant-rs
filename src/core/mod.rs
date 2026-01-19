//! Core circulant matrix types.
//!
//! This module provides the fundamental circulant matrix types.

mod block_circulant;
mod circulant;
mod indexing;

pub use block_circulant::{naive_bccb_mul, BlockCirculant};
pub use circulant::{naive_circulant_mul, Circulant};
pub use indexing::{circular_get, circular_index};
