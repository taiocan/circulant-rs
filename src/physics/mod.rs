//! Quantum physics module for quantum walks and simulations.
//!
//! This module provides types for simulating quantum walks on circulant graphs.

mod coin;
mod state;
mod walk;
mod walk_1d;

pub use coin::Coin;
pub use state::QuantumState;
pub use walk::QuantumWalk;
pub use walk_1d::CoinedWalk1D;
