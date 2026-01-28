// @module: crate::physics::hamiltonian
// @status: stable
// @owner: math_expert
// @feature: physics
// @depends: [crate::error, crate::fft, crate::physics::state, crate::traits, crate::Circulant]
// @tests: [unit, property]

//! Hamiltonian operators for continuous-time quantum walks.
//!
//! This module provides traits and implementations for quantum Hamiltonians,
//! with a focus on circulant Hamiltonians that enable O(N log N) time evolution.

#[allow(deprecated)]
use crate::Circulant;
use crate::error::{CirculantError, Result};
use crate::fft::{FftBackend, RustFftBackend};
use crate::physics::state::QuantumState;
use crate::traits::Scalar;
use num_complex::Complex;
use std::sync::Arc;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Trait for quantum Hamiltonians.
///
/// A Hamiltonian H generates time evolution via U(t) = exp(-iHt).
/// For continuous-time quantum walks, the walker's state evolves as |ψ(t)⟩ = U(t)|ψ(0)⟩.
pub trait Hamiltonian<T: Scalar> {
    /// Propagate the state by time t: |ψ⟩ → exp(-iHt)|ψ⟩.
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state to evolve (modified in place)
    /// * `time` - The evolution time
    fn propagate(&self, state: &mut QuantumState<T>, time: T);

    /// Get the eigenvalues of the Hamiltonian.
    fn eigenvalues(&self) -> Vec<Complex<T>>;

    /// Get the dimension of the Hamiltonian.
    fn dimension(&self) -> usize;
}

/// A circulant Hamiltonian with O(N log N) time evolution.
///
/// Circulant Hamiltonians are diagonalized by the DFT, enabling efficient
/// time evolution using FFT:
///
/// 1. Compute eigenvalues λₖ = FFT(generator)
/// 2. FFT the state: |ψ̃⟩ = FFT|ψ⟩
/// 3. Apply phase evolution: |ψ̃(t)⟩ₖ = exp(-iλₖt)|ψ̃(0)⟩ₖ
/// 4. IFFT to get final state: |ψ(t)⟩ = IFFT|ψ̃(t)⟩
///
/// # Example: Cycle Graph Laplacian
///
/// The cycle graph Laplacian H with H\[i,i\] = 2 and H\[i,i±1\] = -1
/// has generator [2, -1, 0, ..., 0, -1], giving a continuous-time
/// quantum walk on a cycle.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CirculantHamiltonian<T: Scalar + rustfft::FftNum> {
    /// The generator of the circulant matrix (first row).
    generator: Vec<Complex<T>>,

    /// Cached eigenvalues (FFT of generator).
    #[cfg_attr(feature = "serde", serde(skip))]
    cached_eigenvalues: Option<Vec<Complex<T>>>,

    /// FFT backend.
    #[cfg_attr(feature = "serde", serde(skip))]
    fft: Option<Arc<RustFftBackend<T>>>,
}

impl<T: Scalar + rustfft::FftNum> CirculantHamiltonian<T> {
    /// Create a new circulant Hamiltonian from a generator vector.
    ///
    /// # Arguments
    ///
    /// * `generator` - The first row of the circulant matrix (must be Hermitian)
    ///
    /// # Errors
    ///
    /// Returns an error if the generator is empty or not Hermitian.
    pub fn new(generator: Vec<Complex<T>>) -> Result<Self> {
        if generator.is_empty() {
            return Err(CirculantError::EmptyGenerator);
        }

        // Check Hermitian: for circulant matrices, this means g[k] = conj(g[n-k])
        let n = generator.len();
        for k in 1..n {
            let conj_k = if k == 0 { 0 } else { n - k };
            let diff_re = (generator[k].re - generator[conj_k].re).abs();
            let diff_im = (generator[k].im + generator[conj_k].im).abs();
            let tol = T::from(1e-10).unwrap_or_else(T::zero);
            if diff_re > tol || diff_im > tol {
                return Err(CirculantError::NotHermitian);
            }
        }

        let fft = Arc::new(RustFftBackend::new(n)?);
        let mut eigenvalues = generator.clone();
        fft.fft_forward(&mut eigenvalues);

        Ok(Self {
            generator,
            cached_eigenvalues: Some(eigenvalues),
            fft: Some(fft),
        })
    }

    /// Create a circulant Hamiltonian from a real generator vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the generator is empty or not symmetric.
    pub fn from_real_generator(generator: Vec<T>) -> Result<Self> {
        let complex_gen: Vec<Complex<T>> = generator
            .into_iter()
            .map(|x| Complex::new(x, T::zero()))
            .collect();
        Self::new(complex_gen)
    }

    /// Create the Laplacian Hamiltonian for a cycle graph of size n.
    ///
    /// This is the standard continuous-time quantum walk Hamiltonian on a cycle:
    /// H\[i,i\] = 2 (diagonal), H\[i,i±1\] = -1 (neighbors), all others 0.
    ///
    /// The eigenvalues are λₖ = 2(1 - cos(2πk/n)) for k = 0, 1, ..., n-1.
    ///
    /// # Errors
    ///
    /// Returns an error if n < 3.
    pub fn cycle_graph(n: usize) -> Result<Self> {
        if n < 3 {
            return Err(CirculantError::InvalidWalkParameters(
                "cycle graph needs at least 3 vertices".to_string(),
            ));
        }

        // Generator: [2, -1, 0, ..., 0, -1]
        let mut generator = vec![Complex::new(T::zero(), T::zero()); n];
        generator[0] = Complex::new(T::from(2.0).unwrap_or_else(T::zero), T::zero());
        generator[1] = Complex::new(-T::one(), T::zero());
        generator[n - 1] = Complex::new(-T::one(), T::zero());

        Self::new(generator)
    }

    /// Create a custom circulant Hamiltonian from a coupling pattern.
    ///
    /// # Arguments
    ///
    /// * `n` - Size of the Hamiltonian
    /// * `couplings` - List of (distance, coupling_strength) pairs
    ///
    /// For example, `couplings = [(1, -1.0)]` gives nearest-neighbor hopping.
    pub fn from_couplings(n: usize, couplings: &[(usize, T)]) -> Result<Self> {
        if n == 0 {
            return Err(CirculantError::EmptyGenerator);
        }

        let mut generator = vec![Complex::new(T::zero(), T::zero()); n];

        // Diagonal element is negative sum of off-diagonal (for Laplacian normalization)
        let mut diag_sum = T::zero();

        for &(dist, coupling) in couplings {
            if dist > 0 && dist < n {
                generator[dist] = Complex::new(coupling, T::zero());
                generator[n - dist] = Complex::new(coupling, T::zero());
                diag_sum = diag_sum - coupling - coupling;
            }
        }

        generator[0] = Complex::new(diag_sum, T::zero());

        Self::new(generator)
    }

    /// Precompute eigenvalues for repeated time evolution.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub fn precompute(&mut self) -> Result<()> {
        if self.cached_eigenvalues.is_none() {
            self.ensure_initialized()?;
        }
        Ok(())
    }

    /// Ensure FFT backend and eigenvalues are initialized.
    ///
    /// # Errors
    ///
    /// Returns an error if FFT initialization fails.
    pub fn ensure_initialized(&mut self) -> Result<()> {
        if self.fft.is_none() {
            let n = self.generator.len();
            let fft = Arc::new(RustFftBackend::new(n)?);
            let mut eigenvalues = self.generator.clone();
            fft.fft_forward(&mut eigenvalues);
            self.fft = Some(fft);
            self.cached_eigenvalues = Some(eigenvalues);
        }
        Ok(())
    }

    /// Get the generator (first row) of the Hamiltonian.
    pub fn generator(&self) -> &[Complex<T>] {
        &self.generator
    }

    /// Get the underlying circulant matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the circulant cannot be created.
    #[allow(deprecated)]
    pub fn to_circulant(&self) -> Result<Circulant<T>> {
        Circulant::new(self.generator.clone())
    }
}

impl<T: Scalar + rustfft::FftNum> Hamiltonian<T> for CirculantHamiltonian<T> {
    fn propagate(&self, state: &mut QuantumState<T>, time: T) {
        let fft = match self.fft.as_ref() {
            Some(f) => f,
            None => return,
        };
        let eigenvalues = match self.cached_eigenvalues.as_ref() {
            Some(e) => e,
            None => return,
        };

        // For circulant Hamiltonians with coin_dim = 1, we can use the efficient method
        // For coin_dim > 1, we need to handle each position separately
        let n = self.generator.len();
        let coin_dim = state.coin_dim();

        if coin_dim == 1 {
            // Simple case: state is just position amplitudes
            let mut amplitudes = state.amplitudes().to_vec();

            // FFT
            fft.fft_forward(&mut amplitudes);

            // Apply phase evolution: exp(-i * eigenvalue * time)
            for (amp, eig) in amplitudes.iter_mut().zip(eigenvalues.iter()) {
                // eigenvalue should be real for Hermitian H
                let phase = -eig.re * time;
                let rotation = Complex::new(phase.cos(), phase.sin());
                *amp *= rotation;
            }

            // IFFT
            fft.fft_inverse(&mut amplitudes);

            // Write back
            let state_amps = state.amplitudes_mut();
            for (i, amp) in amplitudes.into_iter().enumerate() {
                state_amps[i] = amp;
            }
        } else {
            // General case: handle each coin state separately
            // The Hamiltonian only acts on position, not coin
            for c in 0..coin_dim {
                // Extract amplitudes for this coin state
                let mut coin_amps: Vec<Complex<T>> = (0..n)
                    .map(|pos| state.amplitudes()[pos * coin_dim + c])
                    .collect();

                // FFT
                fft.fft_forward(&mut coin_amps);

                // Apply phase evolution
                for (amp, eig) in coin_amps.iter_mut().zip(eigenvalues.iter()) {
                    let phase = -eig.re * time;
                    let rotation = Complex::new(phase.cos(), phase.sin());
                    *amp *= rotation;
                }

                // IFFT
                fft.fft_inverse(&mut coin_amps);

                // Write back
                let state_amps = state.amplitudes_mut();
                for pos in 0..n {
                    state_amps[pos * coin_dim + c] = coin_amps[pos];
                }
            }
        }
    }

    fn eigenvalues(&self) -> Vec<Complex<T>> {
        if let Some(ref cached) = self.cached_eigenvalues {
            return cached.clone();
        }

        // Compute eigenvalues via FFT
        match RustFftBackend::new(self.generator.len()) {
            Ok(fft) => {
                let mut eigs = self.generator.clone();
                fft.fft_forward(&mut eigs);
                eigs
            }
            Err(_) => {
                // Fallback: return zeros (shouldn't happen for valid Hamiltonians)
                vec![Complex::new(T::zero(), T::zero()); self.generator.len()]
            }
        }
    }

    fn dimension(&self) -> usize {
        self.generator.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cycle_graph_hamiltonian() {
        let h = CirculantHamiltonian::<f64>::cycle_graph(8).unwrap();
        assert_eq!(h.dimension(), 8);

        // Generator should be [2, -1, 0, 0, 0, 0, 0, -1]
        let gen = h.generator();
        assert_relative_eq!(gen[0].re, 2.0, epsilon = 1e-10);
        assert_relative_eq!(gen[1].re, -1.0, epsilon = 1e-10);
        assert_relative_eq!(gen[7].re, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hamiltonian_eigenvalues() {
        let h = CirculantHamiltonian::<f64>::cycle_graph(8).unwrap();
        let eigs = h.eigenvalues();

        assert_eq!(eigs.len(), 8);

        // For cycle graph: λₖ = 2(1 - cos(2πk/n))
        // λ₀ = 0, λ₄ = 4 (for n=8)
        assert_relative_eq!(eigs[0].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(eigs[4].re, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_propagation_preserves_norm() {
        let h = CirculantHamiltonian::<f64>::cycle_graph(64).unwrap();
        let mut state = QuantumState::localized(32, 64, 1).unwrap();

        h.propagate(&mut state, 1.0);

        assert_relative_eq!(state.norm_squared(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_propagation_preserves_norm_long_time() {
        let h = CirculantHamiltonian::<f64>::cycle_graph(64).unwrap();
        let mut state = QuantumState::localized(32, 64, 1).unwrap();

        // Propagate for many time steps
        for _ in 0..100 {
            h.propagate(&mut state, 0.1);
            assert_relative_eq!(state.norm_squared(), 1.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_zero_time_identity() {
        let h = CirculantHamiltonian::<f64>::cycle_graph(16).unwrap();
        let initial = QuantumState::localized(8, 16, 1).unwrap();
        let mut state = initial.clone();

        h.propagate(&mut state, 0.0);

        // State should be unchanged
        for i in 0..16 {
            assert_relative_eq!(
                state.amplitudes()[i].re,
                initial.amplitudes()[i].re,
                epsilon = 1e-10
            );
            assert_relative_eq!(
                state.amplitudes()[i].im,
                initial.amplitudes()[i].im,
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_continuous_walk_spreading() {
        let h = CirculantHamiltonian::<f64>::cycle_graph(101).unwrap();
        let mut state = QuantumState::localized(50, 101, 1).unwrap();

        // Propagate for some time
        h.propagate(&mut state, 5.0);

        let probs = state.position_probabilities();

        // Should have spread from center
        let center_prob = probs[50];
        assert!(center_prob < 0.5, "Probability should spread from center");

        // Total should still be 1
        let total: f64 = probs.iter().sum();
        assert_relative_eq!(total, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_from_couplings() {
        // Create a next-nearest-neighbor Hamiltonian
        let h = CirculantHamiltonian::<f64>::from_couplings(8, &[(1, -1.0), (2, -0.5)]).unwrap();
        assert_eq!(h.dimension(), 8);

        // Check structure
        let gen = h.generator();
        assert_relative_eq!(gen[1].re, -1.0, epsilon = 1e-10);
        assert_relative_eq!(gen[2].re, -0.5, epsilon = 1e-10);
        assert_relative_eq!(gen[7].re, -1.0, epsilon = 1e-10);
        assert_relative_eq!(gen[6].re, -0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_not_hermitian_error() {
        // Non-Hermitian generator: g[1] ≠ conj(g[n-1])
        let gen = vec![
            Complex::new(2.0, 0.0),
            Complex::new(-1.0, 1.0), // Has imaginary part
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(-1.0, 0.0), // But this doesn't have -i
        ];
        let result = CirculantHamiltonian::<f64>::new(gen);
        assert!(matches!(result, Err(CirculantError::NotHermitian)));
    }

    #[test]
    fn test_from_real_generator() {
        let h =
            CirculantHamiltonian::<f64>::from_real_generator(vec![2.0, -1.0, 0.0, -1.0]).unwrap();
        assert_eq!(h.dimension(), 4);

        let mut state = QuantumState::localized(0, 4, 1).unwrap();
        h.propagate(&mut state, 1.0);
        assert_relative_eq!(state.norm_squared(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hamiltonian_with_coin_space() {
        // Test that Hamiltonian works correctly when state has coin_dim > 1
        let h = CirculantHamiltonian::<f64>::cycle_graph(16).unwrap();
        let mut state = QuantumState::superposition_at(8, 16, 2).unwrap();

        h.propagate(&mut state, 1.0);

        assert_relative_eq!(state.norm_squared(), 1.0, epsilon = 1e-10);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_hamiltonian_serialization() {
        let h = CirculantHamiltonian::<f64>::cycle_graph(32).unwrap();

        let encoded = bincode::serialize(&h).unwrap();
        let mut decoded: CirculantHamiltonian<f64> = bincode::deserialize(&encoded).unwrap();

        decoded.ensure_initialized().unwrap();

        assert_eq!(decoded.dimension(), 32);

        // Verify it works correctly after deserialization
        let mut state = QuantumState::localized(16, 32, 1).unwrap();
        decoded.propagate(&mut state, 1.0);
        assert_relative_eq!(state.norm_squared(), 1.0, epsilon = 1e-10);
    }
}
