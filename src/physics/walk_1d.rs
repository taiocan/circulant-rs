//! 1D coined quantum walk implementation.

use crate::fft::{FftBackend, RustFftBackend};
use crate::physics::coin::Coin;
use crate::physics::state::QuantumState;
use crate::physics::walk::QuantumWalk;
use crate::traits::Scalar;
use ndarray::Array2;
use num_complex::Complex;
use std::sync::Arc;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A discrete-time coined quantum walk on a 1D cycle (ring).
///
/// The walk takes place on a cycle of N positions with periodic boundary conditions.
/// Each position has an internal coin degree of freedom (typically 2 for left/right).
///
/// The evolution is:
/// 1. Apply the coin operator C to each position: |x,c⟩ → |x, C|c⟩⟩
/// 2. Apply the shift operator S: |x,0⟩ → |x-1,0⟩, |x,1⟩ → |x+1,1⟩
///
/// The shift operator is implemented using FFT-based circulant multiplication
/// for O(N log N) efficiency.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CoinedWalk1D<T: Scalar + rustfft::FftNum> {
    /// Number of positions on the cycle.
    num_positions: usize,

    /// The coin operator.
    coin: Coin,

    /// Precomputed coin matrix.
    #[cfg_attr(feature = "serde", serde(skip))]
    coin_matrix: Option<Array2<Complex<T>>>,

    /// FFT backend for shift operations.
    #[cfg_attr(feature = "serde", serde(skip))]
    fft: Option<Arc<RustFftBackend<T>>>,

    /// Precomputed left shift eigenvalues.
    #[cfg_attr(feature = "serde", serde(skip))]
    left_shift_spectrum: Option<Vec<Complex<T>>>,

    /// Precomputed right shift eigenvalues.
    #[cfg_attr(feature = "serde", serde(skip))]
    right_shift_spectrum: Option<Vec<Complex<T>>>,
}

impl<T: Scalar + rustfft::FftNum> CoinedWalk1D<T> {
    /// Create a new 1D coined quantum walk.
    ///
    /// # Arguments
    ///
    /// * `num_positions` - Number of positions on the cycle (must be > 0)
    /// * `coin` - The coin operator (must have dimension 2 for standard walks)
    ///
    /// # Errors
    ///
    /// Returns an error if `num_positions` is 0 or if coin dimension is not 2.
    pub fn new(num_positions: usize, coin: Coin) -> crate::error::Result<Self> {
        if num_positions == 0 {
            return Err(crate::error::CirculantError::InvalidWalkParameters(
                "number of positions must be positive".to_string(),
            ));
        }
        if coin.dimension() != 2 {
            return Err(crate::error::CirculantError::InvalidCoinDimension {
                expected: 2,
                got: coin.dimension(),
            });
        }

        let fft = Arc::new(RustFftBackend::new(num_positions)?);
        let coin_matrix = Some(coin.to_matrix());

        // Precompute shift spectra
        let (left_spectrum, right_spectrum) = Self::compute_shift_spectra(num_positions, &fft);

        Ok(Self {
            num_positions,
            coin,
            coin_matrix,
            fft: Some(fft),
            left_shift_spectrum: Some(left_spectrum),
            right_shift_spectrum: Some(right_spectrum),
        })
    }

    /// Compute the FFT spectra for left and right shifts.
    fn compute_shift_spectra(
        n: usize,
        fft: &RustFftBackend<T>,
    ) -> (Vec<Complex<T>>, Vec<Complex<T>>) {
        // Left shift: generator [0, 0, ..., 0, 1] (shifts left by 1)
        let mut left_gen = vec![Complex::new(T::zero(), T::zero()); n];
        left_gen[n - 1] = Complex::new(T::one(), T::zero());
        fft.fft_forward(&mut left_gen);

        // Right shift: generator [0, 1, 0, ..., 0] (shifts right by 1)
        let mut right_gen = vec![Complex::new(T::zero(), T::zero()); n];
        right_gen[1] = Complex::new(T::one(), T::zero());
        fft.fft_forward(&mut right_gen);

        (left_gen, right_gen)
    }

    /// Ensure internal state is initialized (useful after deserialization).
    ///
    /// # Errors
    ///
    /// Returns an error if FFT initialization fails.
    pub fn ensure_initialized(&mut self) -> crate::error::Result<()> {
        if self.fft.is_none() {
            let fft = Arc::new(RustFftBackend::new(self.num_positions)?);
            let (left, right) = Self::compute_shift_spectra(self.num_positions, &fft);
            self.fft = Some(fft);
            self.left_shift_spectrum = Some(left);
            self.right_shift_spectrum = Some(right);
            self.coin_matrix = Some(self.coin.to_matrix());
        }
        Ok(())
    }

    /// Get the coin operator.
    pub fn coin(&self) -> &Coin {
        &self.coin
    }

    /// Apply shift to a single coin component using FFT.
    fn apply_shift_fft(
        &self,
        amplitudes: &[Complex<T>],
        spectrum: &[Complex<T>],
    ) -> Vec<Complex<T>> {
        let fft = match self.fft.as_ref() {
            Some(f) => f,
            None => return amplitudes.to_vec(),
        };

        // FFT of input
        let mut x_fft = amplitudes.to_vec();
        fft.fft_forward(&mut x_fft);

        // Multiply by shift spectrum
        for (x, s) in x_fft.iter_mut().zip(spectrum.iter()) {
            *x *= *s;
        }

        // Inverse FFT
        fft.fft_inverse(&mut x_fft);

        x_fft
    }
}

impl<T: Scalar + rustfft::FftNum> QuantumWalk<T> for CoinedWalk1D<T> {
    fn coin_operator(&self) -> Array2<Complex<T>> {
        self.coin.to_matrix()
    }

    fn step(&self, state: &mut QuantumState<T>) {
        let n = self.num_positions;
        let (coin_matrix, left_spectrum, right_spectrum) = match (
            self.coin_matrix.as_ref(),
            self.left_shift_spectrum.as_ref(),
            self.right_shift_spectrum.as_ref(),
        ) {
            (Some(c), Some(l), Some(r)) => (c, l, r),
            _ => return, // Not initialized - no-op
        };

        // Step 1: Apply coin operator at each position
        // For each position x: |x,c⟩ → Σ_c' C_{c',c} |x,c'⟩
        let amplitudes = state.amplitudes();
        let mut after_coin = amplitudes.to_vec();

        for pos in 0..n {
            let idx0 = pos * 2;
            let idx1 = pos * 2 + 1;

            let a0 = amplitudes[idx0];
            let a1 = amplitudes[idx1];

            // Apply 2x2 coin matrix
            after_coin[idx0] = coin_matrix[(0, 0)] * a0 + coin_matrix[(0, 1)] * a1;
            after_coin[idx1] = coin_matrix[(1, 0)] * a0 + coin_matrix[(1, 1)] * a1;
        }

        // Step 2: Apply shift operator using FFT
        // |x,0⟩ → |x-1,0⟩ (left shift for coin 0)
        // |x,1⟩ → |x+1,1⟩ (right shift for coin 1)

        // Extract coin-0 and coin-1 components
        let coin0: Vec<Complex<T>> = (0..n).map(|i| after_coin[i * 2]).collect();
        let coin1: Vec<Complex<T>> = (0..n).map(|i| after_coin[i * 2 + 1]).collect();

        // Apply shifts using FFT
        let shifted0 = self.apply_shift_fft(&coin0, left_spectrum);
        let shifted1 = self.apply_shift_fft(&coin1, right_spectrum);

        // Recombine
        let state_amps = state.amplitudes_mut();
        for pos in 0..n {
            state_amps[pos * 2] = shifted0[pos];
            state_amps[pos * 2 + 1] = shifted1[pos];
        }
    }

    fn num_positions(&self) -> usize {
        self.num_positions
    }

    fn coin_dim(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_single_step_preserves_norm() {
        let walk = CoinedWalk1D::<f64>::new(256, Coin::Hadamard).unwrap();
        let mut state = QuantumState::localized(128, 256, 2).unwrap();

        let initial_norm = state.norm_squared();
        walk.step(&mut state);
        let final_norm = state.norm_squared();

        assert_relative_eq!(initial_norm, 1.0, epsilon = 1e-10);
        assert_relative_eq!(final_norm, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_100_steps_preserves_norm() {
        let walk = CoinedWalk1D::<f64>::new(256, Coin::Hadamard).unwrap();
        let mut state = QuantumState::localized(128, 256, 2).unwrap();

        for _step in 0..100 {
            walk.step(&mut state);
            let norm = state.norm_squared();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_walk_spreading_pattern() {
        // Quantum walk should show ballistic spreading with two peaks
        let n = 101;
        let walk = CoinedWalk1D::<f64>::new(n, Coin::Hadamard).unwrap();

        // Start in the middle with superposition coin state
        let state = QuantumState::superposition_at(50, n, 2).unwrap();
        let final_state = walk.simulate(state, 30);

        let probs = final_state.position_probabilities();

        // The walk should have spread out - check that probability
        // is not concentrated at the center
        let center_prob = probs[50];
        let total_spread: f64 = probs.iter().sum();

        assert_relative_eq!(total_spread, 1.0, epsilon = 1e-10);

        // Center should have lost most probability after 30 steps
        assert!(
            center_prob < 0.1,
            "Center probability {} too high",
            center_prob
        );

        // There should be some probability away from center
        let edge_region: f64 =
            probs[20..30].iter().sum::<f64>() + probs[70..80].iter().sum::<f64>();
        assert!(edge_region > 0.1, "Not enough spreading to edges");
    }

    #[test]
    fn test_identity_coin_no_mixing() {
        let n = 16;
        let walk = CoinedWalk1D::<f64>::new(n, Coin::Identity(2)).unwrap();

        // Start at position 8, coin 0
        let state = QuantumState::localized(8, n, 2).unwrap();
        let final_state = walk.simulate(state, 5);

        let probs = final_state.position_probabilities();

        // With identity coin, coin-0 shifts left, coin-1 shifts right
        // Starting with coin-0, after 5 steps should be at position 3
        assert_relative_eq!(probs[3], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_walk_periodic_boundary() {
        let n = 16;
        let walk = CoinedWalk1D::<f64>::new(n, Coin::Identity(2)).unwrap();

        // Start at position 2, coin 0 (will shift left)
        let state = QuantumState::localized(2, n, 2).unwrap();

        // After 3 steps, should wrap around to position 15
        let final_state = walk.simulate(state, 3);
        let probs = final_state.position_probabilities();

        assert_relative_eq!(probs[15], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simulate_method() {
        let walk = CoinedWalk1D::<f64>::new(64, Coin::Hadamard).unwrap();
        let initial = QuantumState::localized(32, 64, 2).unwrap();

        let final_state = walk.simulate(initial.clone(), 10);

        // Verify the state evolved
        let initial_probs = initial.position_probabilities();
        let final_probs = final_state.position_probabilities();

        // Initial state should be localized
        assert_relative_eq!(initial_probs[32], 1.0, epsilon = 1e-10);

        // Final state should have spread
        assert!(final_probs[32] < 0.5, "Walk didn't spread enough");
    }

    #[test]
    fn test_coin_accessor() {
        let walk = CoinedWalk1D::<f64>::new(32, Coin::Hadamard).unwrap();
        assert_eq!(*walk.coin(), Coin::Hadamard);
        assert_eq!(walk.num_positions(), 32);
        assert_eq!(walk.coin_dim(), 2);
    }

    #[test]
    fn test_grover_coin_walk() {
        let n = 64;
        let walk = CoinedWalk1D::<f64>::new(n, Coin::Grover(2)).unwrap();
        let state = QuantumState::localized(32, n, 2).unwrap();

        let final_state = walk.simulate(state, 20);

        // Should still preserve norm
        assert_relative_eq!(final_state.norm_squared(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_walk_asymmetry_with_localized_initial() {
        // Starting with |coin=0⟩ gives asymmetric evolution due to Hadamard coin
        // H|0⟩ = (|0⟩+|1⟩)/√2, then |0⟩ shifts left, |1⟩ shifts right
        // This creates the characteristic quantum walk asymmetry
        let n = 101;
        let walk = CoinedWalk1D::<f64>::new(n, Coin::Hadamard).unwrap();

        let state = QuantumState::localized(50, n, 2).unwrap();
        let final_state = walk.simulate(state, 20);

        let probs = final_state.position_probabilities();

        // Quantum walk with Hadamard coin and |0⟩ initial state shows LEFT bias
        // (more probability on left side due to coin asymmetry)
        let left_sum: f64 = probs[30..50].iter().sum();
        let right_sum: f64 = probs[51..71].iter().sum();

        // Left side should have more probability
        assert!(left_sum > right_sum, "Expected left bias in Hadamard walk");

        // Total should still be 1
        let total: f64 = probs.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-10);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_walk_serialization() {
        let walk = CoinedWalk1D::<f64>::new(64, Coin::Hadamard).unwrap();

        let encoded = bincode::serialize(&walk).unwrap();
        let mut decoded: CoinedWalk1D<f64> = bincode::deserialize(&encoded).unwrap();

        // Need to reinitialize after deserialization
        decoded.ensure_initialized().unwrap();

        assert_eq!(decoded.num_positions(), 64);
        assert_eq!(*decoded.coin(), Coin::Hadamard);

        // Verify it works correctly after deserialization
        let state = QuantumState::localized(32, 64, 2).unwrap();
        let final_state = decoded.simulate(state, 10);
        assert_relative_eq!(final_state.norm_squared(), 1.0, epsilon = 1e-10);
    }
}
