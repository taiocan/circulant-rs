// @test_for: [crate::physics::walk, crate::physics::coin, crate::physics::state]
// @math_verified: true
// @verified_by: math_expert
// @properties_checked: [unitarity, probability_conservation, norm_preservation]
// @version: 0.2.2
// @event: initial

//! Integration tests for quantum physics module.

#![cfg(feature = "physics")]

use approx::assert_relative_eq;
use circulant_rs::physics::{Coin, CoinedWalk1D, QuantumState, QuantumWalk};

#[test]
fn test_quantum_walk_long_evolution() {
    let n = 512;
    let walk = CoinedWalk1D::<f64>::new(n, Coin::Hadamard).unwrap();

    let state = QuantumState::localized(n / 2, n, 2).unwrap();
    let final_state = walk.simulate(state, 200);

    // Norm must be preserved
    assert_relative_eq!(final_state.norm_squared(), 1.0, epsilon = 1e-9);

    // Probability should have spread significantly
    let probs = final_state.position_probabilities();
    let max_prob = probs.iter().cloned().fold(0.0_f64, f64::max);
    assert!(max_prob < 0.1, "Probability too concentrated: {}", max_prob);
}

#[test]
fn test_quantum_walk_different_coins() {
    let n = 64;

    for coin in [Coin::Hadamard, Coin::Grover(2), Coin::Dft(2)] {
        let walk = CoinedWalk1D::<f64>::new(n, coin.clone()).unwrap();
        let state = QuantumState::localized(n / 2, n, 2).unwrap();

        // Run 50 steps
        let final_state = walk.simulate(state, 50);

        // All coins should preserve normalization
        assert_relative_eq!(final_state.norm_squared(), 1.0, epsilon = 1e-10);
    }
}

#[test]
fn test_quantum_walk_ballistic_spreading() {
    // Quantum walks exhibit ballistic spreading (linear in time)
    // compared to classical random walks (sqrt of time)
    let n = 201;
    let walk = CoinedWalk1D::<f64>::new(n, Coin::Hadamard).unwrap();

    // Use localized initial state for clearer ballistic spreading
    // Starting with |coin=0⟩ shows asymmetric but ballistic spread
    let state = QuantumState::localized(100, n, 2).unwrap();

    // After t steps, the wave should have reached approximately t positions away
    let steps = 50;
    let final_state = walk.simulate(state, steps);
    let probs = final_state.position_probabilities();

    // Check that there's significant probability near the "light cone" edges
    // With localized |0⟩ start, probability is biased left but reaches both edges
    // Check regions that should have some probability after 50 steps
    let left_region: f64 = probs[50..80].iter().sum();
    let right_region: f64 = probs[120..150].iter().sum();

    // Both regions should have some probability (though left will have more)
    assert!(
        left_region > 0.1,
        "Left region probability {} too small",
        left_region
    );
    assert!(
        right_region > 0.01,
        "Right region probability {} too small",
        right_region
    );

    // Verify probability is not concentrated at start (ballistic not localized)
    let center_prob = probs[100];
    assert!(
        center_prob < 0.05,
        "Center probability {} too high - walk not spreading ballistically",
        center_prob
    );
}

#[test]
fn test_quantum_walk_localization_with_identity() {
    // Identity coin should produce deterministic evolution
    let n = 32;
    let walk = CoinedWalk1D::<f64>::new(n, Coin::Identity(2)).unwrap();

    // Start with coin 1 (will shift right)
    let state = QuantumState::localized_with_coin(16, 1, n, 2).unwrap();
    let final_state = walk.simulate(state, 5);

    let probs = final_state.position_probabilities();

    // Should be at position 21 (16 + 5)
    assert_relative_eq!(probs[21], 1.0, epsilon = 1e-10);
}

#[test]
fn test_quantum_state_manipulation() {
    let n = 16;
    let mut state = QuantumState::<f64>::localized(8, n, 2).unwrap();

    // Modify state
    state.set(0, 0, num_complex::Complex::new(0.5, 0.0));
    state.set(0, 1, num_complex::Complex::new(0.5, 0.0));

    // State is no longer normalized
    assert!(!state.is_normalized(1e-10));

    // Normalize
    state.normalize();
    assert!(state.is_normalized(1e-10));
}

#[test]
fn test_coin_properties() {
    // Test that all standard coins are unitary
    let coins = vec![
        Coin::Hadamard,
        Coin::Grover(2),
        Coin::Grover(3),
        Coin::Grover(4),
        Coin::Dft(2),
        Coin::Dft(3),
        Coin::Dft(4),
        Coin::Identity(2),
        Coin::Identity(5),
    ];

    for coin in coins {
        assert!(coin.is_unitary::<f64>(1e-10), "{:?} is not unitary", coin);
    }
}

#[test]
fn test_probability_conservation_many_steps() {
    let n = 128;
    let walk = CoinedWalk1D::<f64>::new(n, Coin::Hadamard).unwrap();
    let mut state = QuantumState::superposition_at(64, n, 2).unwrap();

    for _step in 0..100 {
        walk.step(&mut state);

        let probs = state.position_probabilities();
        let total: f64 = probs.iter().sum();

        assert_relative_eq!(total, 1.0, epsilon = 1e-10);
    }
}
