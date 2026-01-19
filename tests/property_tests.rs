//! Property-based tests using proptest.

use circulant_rs::core::Circulant;
use circulant_rs::traits::CirculantOps;
use num_complex::Complex;
use proptest::prelude::*;

// Generate a vector of complex numbers
fn complex_vec(size: usize) -> impl Strategy<Value = Vec<Complex<f64>>> {
    prop::collection::vec(
        (-10.0..10.0_f64, -10.0..10.0_f64).prop_map(|(re, im)| Complex::new(re, im)),
        size,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn test_linearity_addition(
        gen in complex_vec(8),
        x in complex_vec(8),
        y in complex_vec(8)
    ) {
        let c = Circulant::new(gen).unwrap();

        // C(x + y)
        let sum: Vec<Complex<f64>> = x.iter().zip(y.iter()).map(|(a, b)| a + b).collect();
        let c_sum = c.mul_vec(&sum).unwrap();

        // C(x) + C(y)
        let c_x = c.mul_vec(&x).unwrap();
        let c_y = c.mul_vec(&y).unwrap();
        let sum_c: Vec<Complex<f64>> = c_x.iter().zip(c_y.iter()).map(|(a, b)| a + b).collect();

        for (a, b) in c_sum.iter().zip(sum_c.iter()) {
            prop_assert!((a.re - b.re).abs() < 1e-8, "Linearity (addition) failed");
            prop_assert!((a.im - b.im).abs() < 1e-8, "Linearity (addition) failed");
        }
    }

    #[test]
    fn test_linearity_scalar(
        gen in complex_vec(8),
        x in complex_vec(8),
        alpha_re in -10.0..10.0_f64,
        alpha_im in -10.0..10.0_f64
    ) {
        let c = Circulant::new(gen).unwrap();
        let alpha = Complex::new(alpha_re, alpha_im);

        // C(alpha * x)
        let scaled_x: Vec<Complex<f64>> = x.iter().map(|v| alpha * v).collect();
        let c_scaled = c.mul_vec(&scaled_x).unwrap();

        // alpha * C(x)
        let c_x = c.mul_vec(&x).unwrap();
        let scaled_c: Vec<Complex<f64>> = c_x.iter().map(|v| alpha * v).collect();

        for (a, b) in c_scaled.iter().zip(scaled_c.iter()) {
            prop_assert!((a.re - b.re).abs() < 1e-8, "Linearity (scalar) failed");
            prop_assert!((a.im - b.im).abs() < 1e-8, "Linearity (scalar) failed");
        }
    }

    #[test]
    fn test_identity_multiplication(
        x in complex_vec(8)
    ) {
        // Identity circulant: [1, 0, 0, ...]
        let mut gen = vec![Complex::new(0.0, 0.0); 8];
        gen[0] = Complex::new(1.0, 0.0);

        let c = Circulant::new(gen).unwrap();
        let result = c.mul_vec(&x).unwrap();

        for (r, xi) in result.iter().zip(x.iter()) {
            prop_assert!((r.re - xi.re).abs() < 1e-10, "Identity failed");
            prop_assert!((r.im - xi.im).abs() < 1e-10, "Identity failed");
        }
    }

    #[test]
    fn test_eigenvalue_count(
        gen in complex_vec(16)
    ) {
        let c = Circulant::new(gen.clone()).unwrap();
        let eigenvalues = c.eigenvalues();

        prop_assert_eq!(eigenvalues.len(), gen.len(), "Wrong number of eigenvalues");
    }
}

#[cfg(feature = "physics")]
mod physics_props {
    use super::*;
    use circulant_rs::physics::{Coin, CoinedWalk1D, QuantumState, QuantumWalk};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn test_walk_preserves_norm(
            position in 1_usize..63,
            steps in 1_usize..50
        ) {
            let n = 64;
            let walk = CoinedWalk1D::<f64>::new(n, Coin::Hadamard);
            let state = QuantumState::localized(position, n, 2).unwrap();

            let final_state = walk.simulate(state, steps);

            let norm = final_state.norm_squared();
            prop_assert!((norm - 1.0).abs() < 1e-9, "Norm not preserved: {}", norm);
        }

        #[test]
        fn test_probability_sums_to_one(
            position in 1_usize..63,
            steps in 1_usize..30
        ) {
            let n = 64;
            let walk = CoinedWalk1D::<f64>::new(n, Coin::Hadamard);
            let state = QuantumState::localized(position, n, 2).unwrap();

            let final_state = walk.simulate(state, steps);
            let probs = final_state.position_probabilities();
            let total: f64 = probs.iter().sum();

            prop_assert!((total - 1.0).abs() < 1e-9, "Probabilities don't sum to 1: {}", total);
        }
    }
}
