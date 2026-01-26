//! Quantum state representation for quantum walks.

use crate::error::{CirculantError, Result};
use crate::traits::Scalar;
use ndarray::{Array1, Array2};
use num_complex::Complex;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A quantum state for discrete-time quantum walks.
///
/// The state represents a superposition over position and coin degrees of freedom.
/// For a 1D walk with coin dimension 2 and N positions, the state has 2N amplitudes.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct QuantumState<T: Scalar> {
    /// Complex amplitudes (position ⊗ coin space).
    amplitudes: Array1<Complex<T>>,

    /// Number of positions in the walk.
    num_positions: usize,

    /// Dimension of the coin space.
    coin_dim: usize,
}

impl<T: Scalar> QuantumState<T> {
    /// Create a new quantum state from amplitudes.
    ///
    /// # Errors
    ///
    /// Returns an error if the amplitudes length doesn't match num_positions * coin_dim.
    pub fn new(
        amplitudes: Array1<Complex<T>>,
        num_positions: usize,
        coin_dim: usize,
    ) -> Result<Self> {
        let expected_len = num_positions * coin_dim;
        if amplitudes.len() != expected_len {
            return Err(CirculantError::DimensionMismatch {
                expected: expected_len,
                got: amplitudes.len(),
            });
        }

        Ok(Self {
            amplitudes,
            num_positions,
            coin_dim,
        })
    }

    /// Create a localized state at a single position with coin state |0⟩.
    ///
    /// # Errors
    ///
    /// Returns an error if position >= num_positions.
    pub fn localized(position: usize, num_positions: usize, coin_dim: usize) -> Result<Self> {
        if position >= num_positions {
            return Err(CirculantError::PositionOutOfBounds {
                position,
                size: num_positions,
            });
        }

        let total_dim = num_positions * coin_dim;
        let mut amplitudes = Array1::zeros(total_dim);

        // State |position, 0⟩
        amplitudes[position * coin_dim] = Complex::new(T::one(), T::zero());

        Ok(Self {
            amplitudes,
            num_positions,
            coin_dim,
        })
    }

    /// Create a localized state with a specific coin state.
    ///
    /// # Errors
    ///
    /// Returns an error if position >= num_positions or coin_state >= coin_dim.
    pub fn localized_with_coin(
        position: usize,
        coin_state: usize,
        num_positions: usize,
        coin_dim: usize,
    ) -> Result<Self> {
        if position >= num_positions {
            return Err(CirculantError::PositionOutOfBounds {
                position,
                size: num_positions,
            });
        }
        if coin_state >= coin_dim {
            return Err(CirculantError::InvalidCoinDimension {
                expected: coin_dim,
                got: coin_state + 1,
            });
        }

        let total_dim = num_positions * coin_dim;
        let mut amplitudes = Array1::zeros(total_dim);

        // State |position, coin_state⟩
        amplitudes[position * coin_dim + coin_state] = Complex::new(T::one(), T::zero());

        Ok(Self {
            amplitudes,
            num_positions,
            coin_dim,
        })
    }

    /// Create a state that's an equal superposition of coin states at one position.
    pub fn superposition_at(
        position: usize,
        num_positions: usize,
        coin_dim: usize,
    ) -> Result<Self> {
        if position >= num_positions {
            return Err(CirculantError::PositionOutOfBounds {
                position,
                size: num_positions,
            });
        }

        let total_dim = num_positions * coin_dim;
        let mut amplitudes = Array1::zeros(total_dim);

        // Equal superposition over all coin states at this position
        let norm = T::one() / T::from(coin_dim).unwrap_or_else(T::one).sqrt();
        for c in 0..coin_dim {
            amplitudes[position * coin_dim + c] = Complex::new(norm, T::zero());
        }

        Ok(Self {
            amplitudes,
            num_positions,
            coin_dim,
        })
    }

    /// Get the amplitudes.
    pub fn amplitudes(&self) -> &Array1<Complex<T>> {
        &self.amplitudes
    }

    /// Get mutable access to the amplitudes.
    pub fn amplitudes_mut(&mut self) -> &mut Array1<Complex<T>> {
        &mut self.amplitudes
    }

    /// Get the number of positions.
    pub fn num_positions(&self) -> usize {
        self.num_positions
    }

    /// Get the coin dimension.
    pub fn coin_dim(&self) -> usize {
        self.coin_dim
    }

    /// Compute the squared norm of the state.
    pub fn norm_squared(&self) -> T {
        self.amplitudes
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .fold(T::zero(), |acc, x| acc + x)
    }

    /// Normalize the state in place.
    pub fn normalize(&mut self) {
        let norm = self.norm_squared().sqrt();
        if norm > T::epsilon() {
            for amp in self.amplitudes.iter_mut() {
                *amp = Complex::new(amp.re / norm, amp.im / norm);
            }
        }
    }

    /// Check if the state is normalized (norm ≈ 1).
    pub fn is_normalized(&self, tolerance: T) -> bool {
        (self.norm_squared() - T::one()).abs() < tolerance
    }

    /// Compute the probability distribution over positions.
    ///
    /// Returns a vector of probabilities where `prob[i] = Σ_c |ψ(i,c)|²`
    #[allow(clippy::needless_range_loop)]
    pub fn position_probabilities(&self) -> Vec<T> {
        let mut probs = vec![T::zero(); self.num_positions];

        for pos in 0..self.num_positions {
            for c in 0..self.coin_dim {
                let idx = pos * self.coin_dim + c;
                let amp = self.amplitudes[idx];
                probs[pos] += amp.re * amp.re + amp.im * amp.im;
            }
        }

        probs
    }

    /// Get the amplitude at (position, coin_state).
    pub fn get(&self, position: usize, coin_state: usize) -> Complex<T> {
        self.amplitudes[position * self.coin_dim + coin_state]
    }

    /// Set the amplitude at (position, coin_state).
    pub fn set(&mut self, position: usize, coin_state: usize, value: Complex<T>) {
        self.amplitudes[position * self.coin_dim + coin_state] = value;
    }

    /// Convert 2D coordinates to a linear index.
    ///
    /// Uses row-major ordering: index = row * cols + col
    #[inline]
    pub fn index_2d(row: usize, col: usize, cols: usize) -> usize {
        row * cols + col
    }

    /// Create a localized state at a 2D position (row, col) with coin state |0⟩.
    ///
    /// For 2D walks, the position space is a rows × cols torus (periodic boundaries).
    /// The coin dimension is typically 4 for 4-direction walks.
    ///
    /// # Arguments
    ///
    /// * `row` - Row position (0 to rows-1)
    /// * `col` - Column position (0 to cols-1)
    /// * `rows` - Number of rows in the lattice
    /// * `cols` - Number of columns in the lattice
    /// * `coin_dim` - Dimension of coin space (typically 4 for 2D walks)
    ///
    /// # Errors
    ///
    /// Returns an error if row >= rows or col >= cols.
    pub fn localized_2d(
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
        coin_dim: usize,
    ) -> Result<Self> {
        if row >= rows || col >= cols {
            return Err(CirculantError::PositionOutOfBounds {
                position: row * cols + col,
                size: rows * cols,
            });
        }

        let num_positions = rows * cols;
        let total_dim = num_positions * coin_dim;
        let mut amplitudes = Array1::zeros(total_dim);

        let pos = Self::index_2d(row, col, cols);
        amplitudes[pos * coin_dim] = Complex::new(T::one(), T::zero());

        Ok(Self {
            amplitudes,
            num_positions,
            coin_dim,
        })
    }

    /// Create a 2D superposition state at a specific position.
    ///
    /// Creates an equal superposition over all coin states at the given position.
    pub fn superposition_2d(
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
        coin_dim: usize,
    ) -> Result<Self> {
        if row >= rows || col >= cols {
            return Err(CirculantError::PositionOutOfBounds {
                position: row * cols + col,
                size: rows * cols,
            });
        }

        let num_positions = rows * cols;
        let total_dim = num_positions * coin_dim;
        let mut amplitudes = Array1::zeros(total_dim);

        let pos = Self::index_2d(row, col, cols);
        let norm = T::one() / T::from(coin_dim).unwrap_or_else(T::one).sqrt();
        for c in 0..coin_dim {
            amplitudes[pos * coin_dim + c] = Complex::new(norm, T::zero());
        }

        Ok(Self {
            amplitudes,
            num_positions,
            coin_dim,
        })
    }

    /// Extract the 2D probability distribution over positions.
    ///
    /// Returns a 2D array of shape (rows, cols) where each element is
    /// the probability of finding the walker at that position (summed over coin states).
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows in the lattice
    /// * `cols` - Number of columns in the lattice
    ///
    /// # Errors
    ///
    /// Returns an error if rows * cols doesn't match num_positions.
    pub fn position_probabilities_2d(&self, rows: usize, cols: usize) -> Result<Array2<T>> {
        if rows * cols != self.num_positions {
            return Err(CirculantError::DimensionMismatch {
                expected: self.num_positions,
                got: rows * cols,
            });
        }

        let mut probs = Array2::zeros((rows, cols));

        for row in 0..rows {
            for col in 0..cols {
                let pos = Self::index_2d(row, col, cols);
                let mut prob = T::zero();
                for c in 0..self.coin_dim {
                    let amp = self.amplitudes[pos * self.coin_dim + c];
                    prob = prob + amp.re * amp.re + amp.im * amp.im;
                }
                probs[[row, col]] = prob;
            }
        }

        Ok(probs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_localized_state_single_position() {
        let state = QuantumState::<f64>::localized(5, 10, 2).unwrap();

        // Should have amplitude 1 at position 5, coin 0
        assert_eq!(state.get(5, 0), Complex::new(1.0, 0.0));
        assert_eq!(state.get(5, 1), Complex::new(0.0, 0.0));

        // All other positions should be zero
        for pos in 0..10 {
            if pos != 5 {
                assert_eq!(state.get(pos, 0), Complex::new(0.0, 0.0));
                assert_eq!(state.get(pos, 1), Complex::new(0.0, 0.0));
            }
        }
    }

    #[test]
    fn test_localized_with_coin() {
        let state = QuantumState::<f64>::localized_with_coin(3, 1, 10, 2).unwrap();

        assert_eq!(state.get(3, 0), Complex::new(0.0, 0.0));
        assert_eq!(state.get(3, 1), Complex::new(1.0, 0.0));
    }

    #[test]
    fn test_state_normalization() {
        let state = QuantumState::<f64>::localized(5, 10, 2).unwrap();
        assert_relative_eq!(state.norm_squared(), 1.0, epsilon = 1e-10);
        assert!(state.is_normalized(1e-10));
    }

    #[test]
    fn test_superposition_normalized() {
        let state = QuantumState::<f64>::superposition_at(5, 10, 2).unwrap();
        assert_relative_eq!(state.norm_squared(), 1.0, epsilon = 1e-10);

        // Each coin state should have equal probability
        let amp0 = state.get(5, 0).norm();
        let amp1 = state.get(5, 1).norm();
        assert_relative_eq!(amp0, amp1, epsilon = 1e-10);
    }

    #[test]
    fn test_probabilities_sum_to_one() {
        let state = QuantumState::<f64>::localized(5, 10, 2).unwrap();
        let probs = state.position_probabilities();

        let total: f64 = probs.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-10);

        // Only position 5 should have probability 1
        assert_relative_eq!(probs[5], 1.0, epsilon = 1e-10);
        for (i, &p) in probs.iter().enumerate() {
            if i != 5 {
                assert_relative_eq!(p, 0.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_normalize() {
        let mut amplitudes = Array1::zeros(20);
        amplitudes[0] = Complex::new(2.0, 0.0);
        amplitudes[1] = Complex::new(0.0, 2.0);

        let mut state = QuantumState::<f64>::new(amplitudes, 10, 2).unwrap();
        assert_relative_eq!(state.norm_squared(), 8.0, epsilon = 1e-10);

        state.normalize();
        assert_relative_eq!(state.norm_squared(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dimension_mismatch() {
        let amplitudes = Array1::zeros(10);
        let result = QuantumState::<f64>::new(amplitudes, 10, 2);
        assert!(matches!(
            result,
            Err(CirculantError::DimensionMismatch {
                expected: 20,
                got: 10
            })
        ));
    }

    #[test]
    fn test_position_out_of_bounds() {
        let result = QuantumState::<f64>::localized(15, 10, 2);
        assert!(matches!(
            result,
            Err(CirculantError::PositionOutOfBounds {
                position: 15,
                size: 10
            })
        ));
    }

    #[test]
    fn test_get_set() {
        let mut state = QuantumState::<f64>::localized(0, 4, 2).unwrap();

        state.set(1, 1, Complex::new(0.5, 0.5));
        assert_eq!(state.get(1, 1), Complex::new(0.5, 0.5));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_state_serialization_bincode() {
        let state = QuantumState::<f64>::localized(5, 10, 2).unwrap();

        let encoded = bincode::serialize(&state).unwrap();
        let decoded: QuantumState<f64> = bincode::deserialize(&encoded).unwrap();

        assert_eq!(decoded.num_positions(), 10);
        assert_eq!(decoded.coin_dim(), 2);
        assert_relative_eq!(decoded.norm_squared(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_index_2d() {
        assert_eq!(QuantumState::<f64>::index_2d(0, 0, 10), 0);
        assert_eq!(QuantumState::<f64>::index_2d(0, 5, 10), 5);
        assert_eq!(QuantumState::<f64>::index_2d(1, 0, 10), 10);
        assert_eq!(QuantumState::<f64>::index_2d(2, 3, 10), 23);
    }

    #[test]
    fn test_localized_2d_correct_indexing() {
        let state = QuantumState::<f64>::localized_2d(5, 5, 10, 10, 4).unwrap();
        assert_eq!(state.num_positions(), 100);
        assert_eq!(state.coin_dim(), 4);
        assert_relative_eq!(state.norm_squared(), 1.0, epsilon = 1e-10);

        // Position 5,5 = index 55. Amplitude at (55, coin 0) should be 1.
        let pos = QuantumState::<f64>::index_2d(5, 5, 10);
        assert_eq!(state.get(pos, 0), Complex::new(1.0, 0.0));
        assert_eq!(state.get(pos, 1), Complex::new(0.0, 0.0));
    }

    #[test]
    fn test_localized_2d_out_of_bounds() {
        let result = QuantumState::<f64>::localized_2d(10, 5, 10, 10, 4);
        assert!(result.is_err());

        let result = QuantumState::<f64>::localized_2d(5, 10, 10, 10, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_superposition_2d_normalized() {
        let state = QuantumState::<f64>::superposition_2d(5, 5, 10, 10, 4).unwrap();
        assert_relative_eq!(state.norm_squared(), 1.0, epsilon = 1e-10);

        let pos = QuantumState::<f64>::index_2d(5, 5, 10);
        // Each coin state should have equal amplitude
        let expected_amp = 0.5; // 1/sqrt(4) = 0.5
        for c in 0..4 {
            assert_relative_eq!(state.get(pos, c).re, expected_amp, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_position_probabilities_2d() {
        let state = QuantumState::<f64>::localized_2d(3, 4, 8, 8, 4).unwrap();
        let probs = state.position_probabilities_2d(8, 8).unwrap();

        assert_eq!(probs.dim(), (8, 8));

        // Only position (3, 4) should have probability 1
        assert_relative_eq!(probs[[3, 4]], 1.0, epsilon = 1e-10);

        // All other positions should be 0
        for i in 0..8 {
            for j in 0..8 {
                if i != 3 || j != 4 {
                    assert_relative_eq!(probs[[i, j]], 0.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_position_probabilities_2d_dimension_mismatch() {
        let state = QuantumState::<f64>::localized_2d(5, 5, 10, 10, 4).unwrap();
        let result = state.position_probabilities_2d(8, 8);
        assert!(result.is_err());
    }
}
