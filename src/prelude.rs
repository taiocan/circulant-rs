//! Convenient re-exports for common usage.
//!
//! This module provides a convenient way to import commonly used types:
//!
//! ```
//! use circulant_rs::prelude::*;
//! ```

pub use crate::core::{BlockCirculant, Circulant};
pub use crate::error::{CirculantError, Result};
pub use crate::fft::{FftBackend, RustFftBackend};
pub use crate::traits::{BlockOps, CirculantOps, ComplexScalar, Scalar};

#[cfg(feature = "physics")]
pub use crate::physics::{
    CirculantHamiltonian, Coin, CoinedWalk1D, CoinedWalk2D, Hamiltonian, QuantumState, QuantumWalk,
};
