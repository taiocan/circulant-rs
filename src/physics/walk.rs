// @module: crate::physics::walk
// @status: stable
// @owner: math_expert
// @feature: physics
// @depends: [crate::physics::state, crate::traits, ndarray, num_complex]
// @tests: [none]

//! Quantum walk trait definition.

use crate::physics::state::QuantumState;
use crate::traits::Scalar;
use ndarray::Array2;
use num_complex::Complex;

/// Trait for discrete-time quantum walks.
///
/// A quantum walk consists of:
/// 1. A coin operator that mixes internal (coin) states
/// 2. A shift operator that moves the walker based on coin state
pub trait QuantumWalk<T: Scalar> {
    /// Get the coin operator as a matrix.
    fn coin_operator(&self) -> Array2<Complex<T>>;

    /// Perform a single step of the quantum walk.
    ///
    /// A step consists of:
    /// 1. Apply the coin operator to each position
    /// 2. Apply the shift operator
    fn step(&self, state: &mut QuantumState<T>);

    /// Simulate the quantum walk for a given number of steps.
    ///
    /// Returns the final state after all steps.
    fn simulate(&self, initial: QuantumState<T>, steps: usize) -> QuantumState<T> {
        let mut state = initial;
        for _ in 0..steps {
            self.step(&mut state);
        }
        state
    }

    /// Get the number of positions in the walk.
    fn num_positions(&self) -> usize;

    /// Get the coin dimension.
    fn coin_dim(&self) -> usize;
}
