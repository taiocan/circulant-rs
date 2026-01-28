// @module: crate::physics::coin
// @status: stable
// @owner: math_expert
// @feature: physics
// @depends: [crate::traits, ndarray, num_complex]
// @tests: [unit]

//! Coin operators for quantum walks.

use crate::traits::Scalar;
use ndarray::Array2;
use num_complex::Complex;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Standard coin operators for quantum walks.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Coin {
    /// Hadamard coin: H = (1/√2) [[1, 1], [1, -1]]
    ///
    /// Creates equal superpositions - the most common choice for quantum walks.
    Hadamard,

    /// Grover coin: G = (2/d)J - I where J is the all-ones matrix
    ///
    /// For d=2: G = [[0, 1], [1, 0]] (NOT gate)
    /// For general d: G_ij = 2/d - δ_ij
    Grover(usize),

    /// DFT coin: F_jk = ω^(jk) / √d where ω = e^(2πi/d)
    ///
    /// The discrete Fourier transform as a coin operator.
    Dft(usize),

    /// Identity coin: no mixing of coin states.
    Identity(usize),

    /// Custom coin operator from a unitary matrix.
    Custom(Array2<Complex<f64>>),
}

impl Coin {
    /// Get the dimension of the coin space.
    pub fn dimension(&self) -> usize {
        match self {
            Coin::Hadamard => 2,
            Coin::Grover(d) => *d,
            Coin::Dft(d) => *d,
            Coin::Identity(d) => *d,
            Coin::Custom(m) => m.nrows(),
        }
    }

    /// Create a 4D Grover diffusion operator for 2D walks.
    ///
    /// The 4D Grover coin is the standard choice for 2D quantum walks,
    /// creating symmetric spreading in all four directions.
    pub fn grover_4d() -> Self {
        Coin::Grover(4)
    }

    /// Create a 4D DFT coin for 2D walks.
    ///
    /// The DFT coin creates different interference patterns compared to Grover.
    pub fn dft_4d() -> Self {
        Coin::Dft(4)
    }

    /// Create a 4D Hadamard coin (tensor product H ⊗ H).
    ///
    /// This is the tensor product of two 2D Hadamard matrices,
    /// useful for 2D walks that factorize into independent x and y components.
    pub fn hadamard_4d() -> Self {
        let h = FRAC_1_SQRT_2;
        // H ⊗ H = (1/2) [[1,1,1,1], [1,-1,1,-1], [1,1,-1,-1], [1,-1,-1,1]]
        let half = h * h; // 1/2
        let data = vec![
            Complex::new(half, 0.0),
            Complex::new(half, 0.0),
            Complex::new(half, 0.0),
            Complex::new(half, 0.0),
            Complex::new(half, 0.0),
            Complex::new(-half, 0.0),
            Complex::new(half, 0.0),
            Complex::new(-half, 0.0),
            Complex::new(half, 0.0),
            Complex::new(half, 0.0),
            Complex::new(-half, 0.0),
            Complex::new(-half, 0.0),
            Complex::new(half, 0.0),
            Complex::new(-half, 0.0),
            Complex::new(-half, 0.0),
            Complex::new(half, 0.0),
        ];
        // Shape is always valid for 16-element 4x4 matrix
        let matrix = Array2::from_shape_vec((4, 4), data).unwrap_or_else(|_| Array2::eye(4));
        Coin::Custom(matrix)
    }

    /// Convert the coin to a unitary matrix.
    pub fn to_matrix<T: Scalar>(&self) -> Array2<Complex<T>> {
        match self {
            Coin::Hadamard => self.hadamard_matrix(),
            Coin::Grover(d) => self.grover_matrix(*d),
            Coin::Dft(d) => self.dft_matrix(*d),
            Coin::Identity(d) => self.identity_matrix(*d),
            Coin::Custom(m) => m.mapv(|c| {
                Complex::new(
                    T::from(c.re).unwrap_or_else(T::zero),
                    T::from(c.im).unwrap_or_else(T::zero),
                )
            }),
        }
    }

    fn hadamard_matrix<T: Scalar>(&self) -> Array2<Complex<T>> {
        let h = T::from(FRAC_1_SQRT_2).unwrap_or_else(T::zero);
        Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(h, T::zero()),
                Complex::new(h, T::zero()),
                Complex::new(h, T::zero()),
                Complex::new(-h, T::zero()),
            ],
        )
        .unwrap_or_else(|_| Array2::eye(2))
    }

    fn grover_matrix<T: Scalar>(&self, d: usize) -> Array2<Complex<T>> {
        let two_over_d = T::from(2.0).unwrap_or_else(T::zero) / T::from(d).unwrap_or_else(T::one);
        let mut matrix = Array2::zeros((d, d));

        for i in 0..d {
            for j in 0..d {
                if i == j {
                    matrix[(i, j)] = Complex::new(two_over_d - T::one(), T::zero());
                } else {
                    matrix[(i, j)] = Complex::new(two_over_d, T::zero());
                }
            }
        }

        matrix
    }

    fn dft_matrix<T: Scalar>(&self, d: usize) -> Array2<Complex<T>> {
        let norm = T::one() / T::from(d).unwrap_or_else(T::one).sqrt();
        let mut matrix = Array2::zeros((d, d));

        for j in 0..d {
            for k in 0..d {
                let theta = T::from(2.0 * PI * (j * k) as f64 / d as f64).unwrap_or_else(T::zero);
                matrix[(j, k)] = Complex::new(norm * theta.cos(), norm * theta.sin());
            }
        }

        matrix
    }

    fn identity_matrix<T: Scalar>(&self, d: usize) -> Array2<Complex<T>> {
        let mut matrix = Array2::zeros((d, d));
        for i in 0..d {
            matrix[(i, i)] = Complex::new(T::one(), T::zero());
        }
        matrix
    }

    /// Check if the coin matrix is unitary.
    pub fn is_unitary<T: Scalar>(&self, tolerance: T) -> bool {
        let matrix = self.to_matrix::<T>();
        let d = matrix.nrows();

        // C† C should equal I
        for i in 0..d {
            for j in 0..d {
                let mut sum = Complex::new(T::zero(), T::zero());
                for k in 0..d {
                    // (C†)_ik = conj(C_ki)
                    let c_dag_ik = Complex::new(matrix[(k, i)].re, -matrix[(k, i)].im);
                    sum += c_dag_ik * matrix[(k, j)];
                }

                let expected = if i == j { T::one() } else { T::zero() };
                if (sum.re - expected).abs() > tolerance || sum.im.abs() > tolerance {
                    return false;
                }
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hadamard_is_unitary() {
        let coin = Coin::Hadamard;
        assert!(coin.is_unitary::<f64>(1e-10));
    }

    #[test]
    fn test_grover_is_unitary() {
        for d in 2..=5 {
            let coin = Coin::Grover(d);
            assert!(
                coin.is_unitary::<f64>(1e-10),
                "Grover({}) is not unitary",
                d
            );
        }
    }

    #[test]
    fn test_dft_is_unitary() {
        for d in 2..=5 {
            let coin = Coin::Dft(d);
            assert!(coin.is_unitary::<f64>(1e-10), "DFT({}) is not unitary", d);
        }
    }

    #[test]
    fn test_identity_is_unitary() {
        let coin = Coin::Identity(3);
        assert!(coin.is_unitary::<f64>(1e-10));
    }

    #[test]
    fn test_hadamard_equal_superposition() {
        let coin = Coin::Hadamard;
        let matrix = coin.to_matrix::<f64>();

        // H|0⟩ = (|0⟩ + |1⟩)/√2
        // First column should be [1/√2, 1/√2]
        let h = FRAC_1_SQRT_2;
        assert_relative_eq!(matrix[(0, 0)].re, h, epsilon = 1e-10);
        assert_relative_eq!(matrix[(1, 0)].re, h, epsilon = 1e-10);
    }

    #[test]
    fn test_hadamard_values() {
        let coin = Coin::Hadamard;
        let matrix = coin.to_matrix::<f64>();
        let h = FRAC_1_SQRT_2;

        assert_relative_eq!(matrix[(0, 0)].re, h, epsilon = 1e-10);
        assert_relative_eq!(matrix[(0, 1)].re, h, epsilon = 1e-10);
        assert_relative_eq!(matrix[(1, 0)].re, h, epsilon = 1e-10);
        assert_relative_eq!(matrix[(1, 1)].re, -h, epsilon = 1e-10);

        // Imaginary parts should be zero
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(matrix[(i, j)].im, 0.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_grover_2_is_not_gate() {
        let coin = Coin::Grover(2);
        let matrix = coin.to_matrix::<f64>();

        // For d=2: G = [[0, 1], [1, 0]]
        assert_relative_eq!(matrix[(0, 0)].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(matrix[(0, 1)].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(matrix[(1, 0)].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(matrix[(1, 1)].re, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_coin_dimension() {
        assert_eq!(Coin::Hadamard.dimension(), 2);
        assert_eq!(Coin::Grover(3).dimension(), 3);
        assert_eq!(Coin::Dft(4).dimension(), 4);
        assert_eq!(Coin::Identity(5).dimension(), 5);

        let custom = Array2::eye(3);
        assert_eq!(
            Coin::Custom(custom.mapv(|x| Complex::new(x, 0.0))).dimension(),
            3
        );
    }

    #[test]
    fn test_4d_coins_are_unitary() {
        let coins = [Coin::grover_4d(), Coin::dft_4d(), Coin::hadamard_4d()];
        for coin in coins {
            assert!(
                coin.is_unitary::<f64>(1e-10),
                "4D coin {} should be unitary",
                coin.dimension()
            );
        }
    }

    #[test]
    fn test_hadamard_4d_dimension() {
        let coin = Coin::hadamard_4d();
        assert_eq!(coin.dimension(), 4);
    }

    #[test]
    fn test_hadamard_4d_tensor_product_structure() {
        // H⊗H|00⟩ = |++⟩ = (|00⟩+|01⟩+|10⟩+|11⟩)/2
        let coin = Coin::hadamard_4d();
        let matrix = coin.to_matrix::<f64>();

        // First column should be [1/2, 1/2, 1/2, 1/2]
        let half = 0.5;
        for i in 0..4 {
            assert_relative_eq!(matrix[(i, 0)].re, half, epsilon = 1e-10);
            assert_relative_eq!(matrix[(i, 0)].im, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_custom_coin() {
        // Custom Hadamard
        let h = FRAC_1_SQRT_2;
        let custom_h = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(h, 0.0),
                Complex::new(h, 0.0),
                Complex::new(h, 0.0),
                Complex::new(-h, 0.0),
            ],
        )
        .unwrap();

        let coin = Coin::Custom(custom_h);
        assert!(coin.is_unitary::<f64>(1e-10));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_coin_serialization() {
        let coins = vec![
            Coin::Hadamard,
            Coin::Grover(3),
            Coin::Dft(4),
            Coin::Identity(2),
        ];

        for coin in coins {
            let encoded = bincode::serialize(&coin).unwrap();
            let decoded: Coin = bincode::deserialize(&encoded).unwrap();
            assert_eq!(coin, decoded);
        }
    }
}
