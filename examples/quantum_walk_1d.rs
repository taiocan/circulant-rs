//! Example: 1D Quantum Walk Simulation
//!
//! This example demonstrates a discrete-time coined quantum walk on a cycle.
//! The walker starts at the center and evolves using a Hadamard coin.
//!
//! Run with: `cargo run --example quantum_walk_1d`

use circulant_rs::physics::{Coin, CoinedWalk1D, QuantumState, QuantumWalk};

fn main() {
    println!("=== 1D Quantum Walk Simulation ===\n");

    // Parameters
    let num_positions = 101; // Odd number for symmetric display
    let start_position = 50; // Center
    let num_steps = 30;

    // Create the quantum walk with Hadamard coin
    let walk = CoinedWalk1D::<f64>::new(num_positions, Coin::Hadamard)
        .expect("Failed to create quantum walk");

    // Create initial state: superposition of coin states at center
    // This gives symmetric evolution
    let initial_state = QuantumState::superposition_at(start_position, num_positions, 2)
        .expect("Failed to create initial state");

    println!("Walk parameters:");
    println!("  Positions: {}", num_positions);
    println!("  Start: position {}", start_position);
    println!("  Coin: Hadamard");
    println!("  Steps: {}\n", num_steps);

    // Simulate the walk
    let final_state = walk.simulate(initial_state, num_steps);

    // Get probability distribution
    let probs = final_state.position_probabilities();

    // Verify normalization
    let total_prob: f64 = probs.iter().sum();
    println!("Total probability: {:.10} (should be 1.0)\n", total_prob);

    // Print probability distribution as ASCII histogram
    println!("Probability distribution after {} steps:", num_steps);
    println!("{}", "=".repeat(60));

    let max_prob = probs.iter().cloned().fold(0.0_f64, f64::max);
    let scale = 50.0 / max_prob;

    for (pos, &prob) in probs.iter().enumerate() {
        // Only show positions with non-negligible probability
        if prob > 0.001 {
            let bar_len = (prob * scale) as usize;
            let bar = "#".repeat(bar_len);
            println!("{:3}: {:6.4} |{}", pos, prob, bar);
        }
    }

    println!("{}", "=".repeat(60));

    // Statistics
    let mean: f64 = probs.iter().enumerate().map(|(i, &p)| i as f64 * p).sum();

    let variance: f64 = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let diff = i as f64 - mean;
            diff * diff * p
        })
        .sum();

    let std_dev = variance.sqrt();

    println!("\nStatistics:");
    println!("  Mean position: {:.2}", mean);
    println!("  Std deviation: {:.2}", std_dev);
    println!(
        "  Ballistic spread: {:.2} (≈ steps = {})",
        std_dev, num_steps
    );

    // Compare with classical random walk (diffusive: std ≈ sqrt(t))
    println!(
        "\nClassical RW would have std ≈ {:.2} (diffusive)",
        (num_steps as f64).sqrt()
    );
    println!("Quantum walk shows ballistic spreading!");
}
