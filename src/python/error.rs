// @module: crate::python::error
// @status: stable
// @owner: code_expert
// @feature: python
// @depends: [crate::CirculantError, pyo3]
// @tests: [none]

//! Error conversion for Python bindings.

use pyo3::exceptions::PyValueError;
use pyo3::PyErr;

use crate::CirculantError;

impl From<CirculantError> for PyErr {
    fn from(err: CirculantError) -> Self {
        PyValueError::new_err(err.to_string())
    }
}
