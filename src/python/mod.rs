//! Python bindings for circulant-rs via PyO3.
//!
//! This module is not meant to be used from Rust code directly.
//! It provides a Python extension module.

mod circulant;
mod error;

#[cfg(feature = "physics")]
mod physics;

use pyo3::prelude::*;

/// The circulant-rs Python module.
#[pymodule]
fn circulant_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<circulant::PyCirculant>()?;

    #[cfg(feature = "physics")]
    {
        m.add_class::<physics::PyQuantumState>()?;
        m.add_class::<physics::PyCoin>()?;
        m.add_class::<physics::PyCoinedWalk1D>()?;
        m.add_class::<physics::PyCoinedWalk2D>()?;
        m.add_class::<physics::PyCirculantHamiltonian>()?;
    }

    Ok(())
}
