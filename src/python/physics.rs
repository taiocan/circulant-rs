//! Python wrappers for physics module.

#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;

use crate::physics::{
    CirculantHamiltonian, Coin, CoinedWalk1D, CoinedWalk2D, Hamiltonian, QuantumState, QuantumWalk,
};

/// A quantum state for quantum walk simulations.
#[pyclass(name = "QuantumState")]
#[derive(Clone)]
pub struct PyQuantumState {
    inner: QuantumState<f64>,
}

#[pymethods]
impl PyQuantumState {
    /// Create a localized quantum state at a specific position.
    ///
    /// Args:
    ///     position: The position to localize at.
    ///     num_positions: Total number of positions.
    ///     coin_dim: Dimension of the coin space (typically 2).
    ///
    /// Returns:
    ///     A new localized quantum state.
    #[staticmethod]
    fn localized(position: usize, num_positions: usize, coin_dim: usize) -> PyResult<Self> {
        let inner = QuantumState::localized(position, num_positions, coin_dim)?;
        Ok(Self { inner })
    }

    /// Get the position probabilities.
    ///
    /// Returns:
    ///     A list of probabilities for each position.
    fn position_probabilities(&self) -> Vec<f64> {
        self.inner.position_probabilities()
    }

    /// Get the squared norm of the state (should be 1 for normalized states).
    fn norm_squared(&self) -> f64 {
        self.inner.norm_squared()
    }

    /// Get the number of positions.
    fn num_positions(&self) -> usize {
        self.inner.num_positions()
    }

    /// Get the coin dimension.
    fn coin_dim(&self) -> usize {
        self.inner.coin_dim()
    }

    /// Create a localized 2D quantum state.
    ///
    /// Args:
    ///     row: Row position.
    ///     col: Column position.
    ///     rows: Number of rows in the lattice.
    ///     cols: Number of columns in the lattice.
    ///     coin_dim: Dimension of coin space (typically 4 for 2D walks).
    #[staticmethod]
    fn localized_2d(
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
        coin_dim: usize,
    ) -> PyResult<Self> {
        let inner = QuantumState::localized_2d(row, col, rows, cols, coin_dim)?;
        Ok(Self { inner })
    }

    /// Create a 2D superposition state.
    #[staticmethod]
    fn superposition_2d(
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
        coin_dim: usize,
    ) -> PyResult<Self> {
        let inner = QuantumState::superposition_2d(row, col, rows, cols, coin_dim)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantumState(positions={}, coin_dim={})",
            self.inner.num_positions(),
            self.inner.coin_dim()
        )
    }
}

/// A coin operator for quantum walks.
#[pyclass(name = "Coin")]
#[derive(Clone)]
pub struct PyCoin {
    inner: Coin,
}

#[pymethods]
impl PyCoin {
    /// Create a Hadamard coin (2D).
    #[staticmethod]
    fn hadamard() -> Self {
        Self {
            inner: Coin::Hadamard,
        }
    }

    /// Create a Grover coin of given dimension.
    #[staticmethod]
    fn grover(dim: usize) -> Self {
        Self {
            inner: Coin::Grover(dim),
        }
    }

    /// Create a DFT coin of given dimension.
    #[staticmethod]
    fn dft(dim: usize) -> Self {
        Self {
            inner: Coin::Dft(dim),
        }
    }

    /// Create an identity coin of given dimension.
    #[staticmethod]
    fn identity(dim: usize) -> Self {
        Self {
            inner: Coin::Identity(dim),
        }
    }

    /// Create a 4D Grover coin for 2D walks.
    #[staticmethod]
    fn grover_4d() -> Self {
        Self {
            inner: Coin::grover_4d(),
        }
    }

    /// Create a 4D DFT coin for 2D walks.
    #[staticmethod]
    fn dft_4d() -> Self {
        Self {
            inner: Coin::dft_4d(),
        }
    }

    /// Create a 4D Hadamard coin (H⊗H) for 2D walks.
    #[staticmethod]
    fn hadamard_4d() -> Self {
        Self {
            inner: Coin::hadamard_4d(),
        }
    }

    /// Get the dimension of the coin.
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// Check if the coin is unitary.
    fn is_unitary(&self, tolerance: f64) -> bool {
        self.inner.is_unitary::<f64>(tolerance)
    }

    fn __repr__(&self) -> String {
        format!("Coin(dim={})", self.inner.dimension())
    }
}

/// A coined quantum walk on a 1D cycle.
#[pyclass(name = "CoinedWalk1D")]
pub struct PyCoinedWalk1D {
    inner: CoinedWalk1D<f64>,
}

#[pymethods]
impl PyCoinedWalk1D {
    /// Create a new 1D quantum walk.
    ///
    /// Args:
    ///     num_positions: Number of positions in the cycle.
    ///     coin: The coin operator to use.
    #[new]
    fn new(num_positions: usize, coin: PyCoin) -> PyResult<Self> {
        let inner = CoinedWalk1D::new(num_positions, coin.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Perform a single step of the walk.
    fn step(&self, state: &mut PyQuantumState) {
        self.inner.step(&mut state.inner);
    }

    /// Simulate the walk for multiple steps.
    ///
    /// Args:
    ///     initial: The initial quantum state.
    ///     steps: Number of steps to simulate.
    ///
    /// Returns:
    ///     The final quantum state.
    fn simulate(&self, initial: PyQuantumState, steps: usize) -> PyQuantumState {
        let final_state = self.inner.simulate(initial.inner, steps);
        PyQuantumState { inner: final_state }
    }

    /// Get the number of positions.
    fn num_positions(&self) -> usize {
        self.inner.num_positions()
    }

    fn __repr__(&self) -> String {
        format!("CoinedWalk1D(positions={})", self.inner.num_positions())
    }
}

/// A coined quantum walk on a 2D torus.
#[pyclass(name = "CoinedWalk2D")]
pub struct PyCoinedWalk2D {
    inner: CoinedWalk2D<f64>,
}

#[pymethods]
impl PyCoinedWalk2D {
    /// Create a new 2D quantum walk.
    ///
    /// Args:
    ///     rows: Number of rows in the lattice.
    ///     cols: Number of columns in the lattice.
    ///     coin: The coin operator to use (must be 4D).
    #[new]
    fn new(rows: usize, cols: usize, coin: PyCoin) -> PyResult<Self> {
        let inner = CoinedWalk2D::new(rows, cols, coin.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Perform a single step of the walk.
    fn step(&self, state: &mut PyQuantumState) {
        self.inner.step(&mut state.inner);
    }

    /// Simulate the walk for multiple steps.
    fn simulate(&self, initial: PyQuantumState, steps: usize) -> PyQuantumState {
        let final_state = self.inner.simulate(initial.inner, steps);
        PyQuantumState { inner: final_state }
    }

    /// Get the number of rows.
    fn rows(&self) -> usize {
        self.inner.rows()
    }

    /// Get the number of columns.
    fn cols(&self) -> usize {
        self.inner.cols()
    }

    /// Get the total number of positions.
    fn num_positions(&self) -> usize {
        self.inner.num_positions()
    }

    fn __repr__(&self) -> String {
        format!(
            "CoinedWalk2D(rows={}, cols={})",
            self.inner.rows(),
            self.inner.cols()
        )
    }
}

/// A circulant Hamiltonian for continuous-time quantum walks.
#[pyclass(name = "CirculantHamiltonian")]
pub struct PyCirculantHamiltonian {
    inner: CirculantHamiltonian<f64>,
}

#[pymethods]
impl PyCirculantHamiltonian {
    /// Create a cycle graph Hamiltonian.
    ///
    /// Args:
    ///     n: Number of vertices in the cycle.
    ///
    /// Returns:
    ///     A Hamiltonian representing the cycle graph Laplacian.
    #[staticmethod]
    fn cycle_graph(n: usize) -> PyResult<Self> {
        let inner = CirculantHamiltonian::cycle_graph(n)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Create from a real generator vector.
    ///
    /// Args:
    ///     generator: The first row of the circulant matrix (must be symmetric).
    #[staticmethod]
    fn from_real_generator(generator: Vec<f64>) -> PyResult<Self> {
        let inner = CirculantHamiltonian::from_real_generator(generator)?;
        Ok(Self { inner })
    }

    /// Propagate the state by time t.
    ///
    /// Args:
    ///     state: The quantum state to evolve.
    ///     time: The evolution time.
    fn propagate(&self, state: &mut PyQuantumState, time: f64) {
        self.inner.propagate(&mut state.inner, time);
    }

    /// Get the eigenvalues of the Hamiltonian.
    fn eigenvalues(&self) -> Vec<(f64, f64)> {
        self.inner
            .eigenvalues()
            .into_iter()
            .map(|c| (c.re, c.im))
            .collect()
    }

    /// Get the dimension of the Hamiltonian.
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn __repr__(&self) -> String {
        format!("CirculantHamiltonian(dim={})", self.inner.dimension())
    }
}
