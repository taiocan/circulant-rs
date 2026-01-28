// @module: crate::prelude
// @status: stable
// @owner: code_expert
// @feature: none
// @depends: [crate::core, crate::error, crate::fft, crate::traits, crate::physics]
// @tests: [none]

//! Convenient re-exports for common usage.
//!
//! This module provides a convenient way to import commonly used types:
//!
//! ```
//! use circulant_rs::prelude::*;
//! ```

#[allow(deprecated)]
pub use crate::core::{BlockCirculant, Circulant};
pub use crate::core::{Circulant1D, Circulant2D, Circulant3D, Circulant4D, CirculantTensor};
pub use crate::error::{CirculantError, Result};
pub use crate::fft::{FftBackend, RustFftBackend};
pub use crate::traits::{BlockOps, CirculantOps, ComplexScalar, Scalar, TensorOps};

#[cfg(feature = "physics")]
pub use crate::physics::{
    CirculantHamiltonian, Coin, CoinedWalk1D, CoinedWalk2D, Hamiltonian, QuantumState, QuantumWalk,
};
