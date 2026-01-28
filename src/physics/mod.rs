// @module: crate::physics
// @status: stable
// @owner: math_expert
// @feature: physics
// @depends: [crate::physics::coin, crate::physics::hamiltonian, crate::physics::state, crate::physics::walk, crate::physics::walk_1d, crate::physics::walk_2d]
// @tests: [none]

//! Quantum physics module for quantum walks and simulations.
//!
//! This module provides types for simulating quantum walks on circulant graphs.
//!
//! # Discrete-Time Quantum Walks
//!
//! - [`CoinedWalk1D`]: 1D walks on a cycle (ring) with 2D coin space
//! - [`CoinedWalk2D`]: 2D walks on a torus (periodic lattice) with 4D coin space
//!
//! # Continuous-Time Quantum Walks
//!
//! - [`CirculantHamiltonian`]: O(N log N) time evolution via FFT-diagonalized Hamiltonians
//! - [`Hamiltonian`]: Trait for general quantum Hamiltonians

mod coin;
mod hamiltonian;
mod state;
mod walk;
mod walk_1d;
mod walk_2d;

pub use coin::Coin;
pub use hamiltonian::{CirculantHamiltonian, Hamiltonian};
pub use state::QuantumState;
pub use walk::QuantumWalk;
pub use walk_1d::CoinedWalk1D;
pub use walk_2d::CoinedWalk2D;
