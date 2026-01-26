//! 2D coined quantum walk implementation.

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

/// A discrete-time coined quantum walk on a 2D torus (rows × cols lattice).
///
/// The walk takes place on a periodic lattice with positions indexed by (row, col).
/// Each position has a 4-dimensional coin space corresponding to the four directions:
/// - Coin 0: Up (row - 1)
/// - Coin 1: Down (row + 1)
/// - Coin 2: Left (col - 1)
/// - Coin 3: Right (col + 1)
///
/// The evolution is:
/// 1. Apply the coin operator C to each position
/// 2. Apply the shift operator S based on coin state
///
/// All shifts are implemented using 1D FFT-based circulant multiplication for efficiency.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CoinedWalk2D<T: Scalar + rustfft::FftNum> {
    /// Number of rows in the lattice.
    rows: usize,

    /// Number of columns in the lattice.
    cols: usize,

    /// The coin operator (must be 4D for standard 2D walks).
    coin: Coin,

    /// Precomputed coin matrix.
    #[cfg_attr(feature = "serde", serde(skip))]
    coin_matrix: Option<Array2<Complex<T>>>,

    /// FFT backend for row shifts.
    #[cfg_attr(feature = "serde", serde(skip))]
    fft_rows: Option<Arc<RustFftBackend<T>>>,

    /// FFT backend for column shifts.
    #[cfg_attr(feature = "serde", serde(skip))]
    fft_cols: Option<Arc<RustFftBackend<T>>>,

    /// Precomputed shift spectra: [up, down, left, right].
    #[cfg_attr(feature = "serde", serde(skip))]
    shift_spectra: Option<ShiftSpectra<T>>,
}

/// Precomputed FFT spectra for the four shift directions.
#[derive(Clone)]
struct ShiftSpectra<T> {
    /// Shift up (row - 1) spectrum for row FFT.
    up: Vec<Complex<T>>,
    /// Shift down (row + 1) spectrum for row FFT.
    down: Vec<Complex<T>>,
    /// Shift left (col - 1) spectrum for column FFT.
    left: Vec<Complex<T>>,
    /// Shift right (col + 1) spectrum for column FFT.
    right: Vec<Complex<T>>,
}

impl<T: Scalar + rustfft::FftNum> CoinedWalk2D<T> {
    /// Create a new 2D coined quantum walk.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows in the lattice
    /// * `cols` - Number of columns in the lattice
    /// * `coin` - The coin operator (must have dimension 4)
    ///
    /// # Errors
    ///
    /// Returns an error if rows or cols is 0, or if coin dimension is not 4.
    pub fn new(rows: usize, cols: usize, coin: Coin) -> crate::error::Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(crate::error::CirculantError::InvalidWalkParameters(
                "lattice dimensions must be positive".to_string(),
            ));
        }
        if coin.dimension() != 4 {
            return Err(crate::error::CirculantError::InvalidCoinDimension {
                expected: 4,
                got: coin.dimension(),
            });
        }

        let fft_rows = Arc::new(RustFftBackend::new(rows)?);
        let fft_cols = Arc::new(RustFftBackend::new(cols)?);
        let coin_matrix = Some(coin.to_matrix());

        let shift_spectra = Some(Self::compute_shift_spectra(
            rows, cols, &fft_rows, &fft_cols,
        ));

        Ok(Self {
            rows,
            cols,
            coin,
            coin_matrix,
            fft_rows: Some(fft_rows),
            fft_cols: Some(fft_cols),
            shift_spectra,
        })
    }

    /// Compute FFT spectra for all four shift directions.
    fn compute_shift_spectra(
        rows: usize,
        cols: usize,
        fft_rows: &RustFftBackend<T>,
        fft_cols: &RustFftBackend<T>,
    ) -> ShiftSpectra<T> {
        // Up shift: row -> row - 1 (mod rows)
        // Generator: [0, 0, ..., 0, 1] (last element is 1)
        let mut up = vec![Complex::new(T::zero(), T::zero()); rows];
        up[rows - 1] = Complex::new(T::one(), T::zero());
        fft_rows.fft_forward(&mut up);

        // Down shift: row -> row + 1 (mod rows)
        // Generator: [0, 1, 0, ..., 0] (second element is 1)
        let mut down = vec![Complex::new(T::zero(), T::zero()); rows];
        down[1] = Complex::new(T::one(), T::zero());
        fft_rows.fft_forward(&mut down);

        // Left shift: col -> col - 1 (mod cols)
        let mut left = vec![Complex::new(T::zero(), T::zero()); cols];
        left[cols - 1] = Complex::new(T::one(), T::zero());
        fft_cols.fft_forward(&mut left);

        // Right shift: col -> col + 1 (mod cols)
        let mut right = vec![Complex::new(T::zero(), T::zero()); cols];
        right[1] = Complex::new(T::one(), T::zero());
        fft_cols.fft_forward(&mut right);

        ShiftSpectra {
            up,
            down,
            left,
            right,
        }
    }

    /// Get the number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get the coin operator.
    pub fn coin(&self) -> &Coin {
        &self.coin
    }

    /// Ensure internal state is initialized (useful after deserialization).
    ///
    /// # Errors
    ///
    /// Returns an error if FFT initialization fails.
    pub fn ensure_initialized(&mut self) -> crate::error::Result<()> {
        if self.fft_rows.is_none() {
            let fft_rows = Arc::new(RustFftBackend::new(self.rows)?);
            let fft_cols = Arc::new(RustFftBackend::new(self.cols)?);
            self.shift_spectra = Some(Self::compute_shift_spectra(
                self.rows, self.cols, &fft_rows, &fft_cols,
            ));
            self.fft_rows = Some(fft_rows);
            self.fft_cols = Some(fft_cols);
            self.coin_matrix = Some(self.coin.to_matrix());
        }
        Ok(())
    }

    /// Apply a row shift to a 2D array using FFT.
    ///
    /// This shifts all rows by applying the circulant multiplication to each column.
    /// Does nothing if FFT is not initialized.
    fn apply_row_shift(&self, data: &mut [Complex<T>], spectrum: &[Complex<T>]) {
        let fft = match self.fft_rows.as_ref() {
            Some(f) => f,
            None => return,
        };

        // For each column, apply the row shift
        for col in 0..self.cols {
            // Extract column
            let mut col_data: Vec<Complex<T>> = (0..self.rows)
                .map(|row| data[row * self.cols + col])
                .collect();

            // FFT
            fft.fft_forward(&mut col_data);

            // Multiply by spectrum
            for (x, s) in col_data.iter_mut().zip(spectrum.iter()) {
                *x *= *s;
            }

            // IFFT
            fft.fft_inverse(&mut col_data);

            // Write back
            for row in 0..self.rows {
                data[row * self.cols + col] = col_data[row];
            }
        }
    }

    /// Apply a column shift to a 2D array using FFT.
    ///
    /// This shifts all columns by applying the circulant multiplication to each row.
    /// Does nothing if FFT is not initialized.
    fn apply_col_shift(&self, data: &mut [Complex<T>], spectrum: &[Complex<T>]) {
        let fft = match self.fft_cols.as_ref() {
            Some(f) => f,
            None => return,
        };

        // For each row, apply the column shift
        for row in 0..self.rows {
            let start = row * self.cols;
            let mut row_data: Vec<Complex<T>> = data[start..start + self.cols].to_vec();

            // FFT
            fft.fft_forward(&mut row_data);

            // Multiply by spectrum
            for (x, s) in row_data.iter_mut().zip(spectrum.iter()) {
                *x *= *s;
            }

            // IFFT
            fft.fft_inverse(&mut row_data);

            // Write back
            data[start..start + self.cols].copy_from_slice(&row_data);
        }
    }
}

impl<T: Scalar + rustfft::FftNum> QuantumWalk<T> for CoinedWalk2D<T> {
    fn coin_operator(&self) -> Array2<Complex<T>> {
        self.coin.to_matrix()
    }

    fn step(&self, state: &mut QuantumState<T>) {
        let n = self.rows * self.cols;
        let coin_matrix = match self.coin_matrix.as_ref() {
            Some(m) => m,
            None => return,
        };
        let spectra = match self.shift_spectra.as_ref() {
            Some(s) => s,
            None => return,
        };

        // Step 1: Apply coin operator at each position
        let amplitudes = state.amplitudes();
        let mut after_coin = amplitudes.to_vec();

        for pos in 0..n {
            let base = pos * 4;
            let a = [
                amplitudes[base],
                amplitudes[base + 1],
                amplitudes[base + 2],
                amplitudes[base + 3],
            ];

            // Apply 4x4 coin matrix
            for c_out in 0..4 {
                let mut sum = Complex::new(T::zero(), T::zero());
                for c_in in 0..4 {
                    sum += coin_matrix[(c_out, c_in)] * a[c_in];
                }
                after_coin[base + c_out] = sum;
            }
        }

        // Step 2: Apply shift operator
        // Extract each coin component and shift it appropriately:
        // - Coin 0 (up): shift row - 1
        // - Coin 1 (down): shift row + 1
        // - Coin 2 (left): shift col - 1
        // - Coin 3 (right): shift col + 1

        let mut coin_components: [Vec<Complex<T>>; 4] = [
            (0..n).map(|i| after_coin[i * 4]).collect(),
            (0..n).map(|i| after_coin[i * 4 + 1]).collect(),
            (0..n).map(|i| after_coin[i * 4 + 2]).collect(),
            (0..n).map(|i| after_coin[i * 4 + 3]).collect(),
        ];

        // Apply shifts
        self.apply_row_shift(&mut coin_components[0], &spectra.up);
        self.apply_row_shift(&mut coin_components[1], &spectra.down);
        self.apply_col_shift(&mut coin_components[2], &spectra.left);
        self.apply_col_shift(&mut coin_components[3], &spectra.right);

        // Recombine
        let state_amps = state.amplitudes_mut();
        for pos in 0..n {
            state_amps[pos * 4] = coin_components[0][pos];
            state_amps[pos * 4 + 1] = coin_components[1][pos];
            state_amps[pos * 4 + 2] = coin_components[2][pos];
            state_amps[pos * 4 + 3] = coin_components[3][pos];
        }
    }

    fn num_positions(&self) -> usize {
        self.rows * self.cols
    }

    fn coin_dim(&self) -> usize {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_2d_walk_creation() {
        let walk = CoinedWalk2D::<f64>::new(16, 16, Coin::grover_4d()).unwrap();
        assert_eq!(walk.rows(), 16);
        assert_eq!(walk.cols(), 16);
        assert_eq!(walk.num_positions(), 256);
        assert_eq!(walk.coin_dim(), 4);
    }

    #[test]
    fn test_2d_walk_preserves_norm() {
        let walk = CoinedWalk2D::<f64>::new(16, 16, Coin::grover_4d()).unwrap();
        let initial = QuantumState::localized_2d(8, 8, 16, 16, 4).unwrap();

        let final_state = walk.simulate(initial, 10);
        assert_relative_eq!(final_state.norm_squared(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_2d_walk_100_steps_preserves_norm() {
        let walk = CoinedWalk2D::<f64>::new(16, 16, Coin::grover_4d()).unwrap();
        let mut state = QuantumState::superposition_2d(8, 8, 16, 16, 4).unwrap();

        for _step in 0..100 {
            walk.step(&mut state);
            let norm = state.norm_squared();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_2d_walk_spreading() {
        let walk = CoinedWalk2D::<f64>::new(32, 32, Coin::grover_4d()).unwrap();
        let initial = QuantumState::superposition_2d(16, 16, 32, 32, 4).unwrap();

        let final_state = walk.simulate(initial, 10);
        let probs = final_state.position_probabilities_2d(32, 32).unwrap();

        // Check that probability has spread from center
        let center_prob = probs[[16, 16]];
        assert!(center_prob < 0.5, "Walk should spread from center");

        // Total probability should be 1
        let total: f64 = probs.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_2d_walk_identity_coin_no_mixing() {
        let walk = CoinedWalk2D::<f64>::new(16, 16, Coin::Identity(4)).unwrap();

        // Start at position (8, 8), coin 0 (up)
        let state = QuantumState::localized_2d(8, 8, 16, 16, 4).unwrap();
        let final_state = walk.simulate(state, 3);

        let probs = final_state.position_probabilities_2d(16, 16).unwrap();

        // With identity coin and coin 0 (up), after 3 steps should be at row 5
        assert_relative_eq!(probs[[5, 8]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_2d_walk_periodic_boundary_row() {
        let walk = CoinedWalk2D::<f64>::new(8, 8, Coin::Identity(4)).unwrap();

        // Start at row 1, coin 0 (up), after 2 steps should wrap to row 7
        let state = QuantumState::localized_2d(1, 4, 8, 8, 4).unwrap();
        let final_state = walk.simulate(state, 2);

        let probs = final_state.position_probabilities_2d(8, 8).unwrap();
        assert_relative_eq!(probs[[7, 4]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_2d_walk_periodic_boundary_col() {
        let walk = CoinedWalk2D::<f64>::new(8, 8, Coin::Identity(4)).unwrap();

        // Start at col 1, coin 2 (left), after 2 steps should wrap to col 7
        let mut state = QuantumState::localized_2d(4, 1, 8, 8, 4).unwrap();
        // Set coin state to 2 (left)
        let pos = QuantumState::<f64>::index_2d(4, 1, 8);
        state.set(pos, 0, Complex::new(0.0, 0.0));
        state.set(pos, 2, Complex::new(1.0, 0.0));

        let final_state = walk.simulate(state, 2);
        let probs = final_state.position_probabilities_2d(8, 8).unwrap();
        assert_relative_eq!(probs[[4, 7]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_different_4d_coins() {
        let coins = [Coin::grover_4d(), Coin::dft_4d(), Coin::hadamard_4d()];

        for coin in coins {
            let walk = CoinedWalk2D::<f64>::new(16, 16, coin).unwrap();
            let initial = QuantumState::superposition_2d(8, 8, 16, 16, 4).unwrap();
            let final_state = walk.simulate(initial, 20);

            assert_relative_eq!(final_state.norm_squared(), 1.0, epsilon = 1e-9);
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_2d_walk_serialization() {
        let walk = CoinedWalk2D::<f64>::new(16, 16, Coin::grover_4d()).unwrap();

        let encoded = bincode::serialize(&walk).unwrap();
        let mut decoded: CoinedWalk2D<f64> = bincode::deserialize(&encoded).unwrap();

        decoded.ensure_initialized().unwrap();

        assert_eq!(decoded.rows(), 16);
        assert_eq!(decoded.cols(), 16);

        // Verify it works correctly after deserialization
        let state = QuantumState::localized_2d(8, 8, 16, 16, 4).unwrap();
        let final_state = decoded.simulate(state, 5);
        assert_relative_eq!(final_state.norm_squared(), 1.0, epsilon = 1e-10);
    }
}
