//! Example: 2D Quantum Walk Simulation
//!
//! This example demonstrates a discrete-time coined quantum walk on a 2D torus.
//! The walker starts at the center and evolves using a Grover diffusion coin.
//!
//! Run with: `cargo run --example quantum_walk_2d --features physics`

use circulant_rs::physics::{Coin, CoinedWalk2D, QuantumState, QuantumWalk};

fn main() {
    println!("=== 2D Quantum Walk Simulation ===\n");

    // Parameters
    let rows = 21;
    let cols = 21;
    let start_row = 10;
    let start_col = 10;
    let num_steps = 15;

    // Create the quantum walk with 4D Grover coin
    let walk = CoinedWalk2D::<f64>::new(rows, cols, Coin::grover_4d())
        .expect("Failed to create quantum walk");

    // Create initial state: superposition at center
    let initial_state = QuantumState::superposition_2d(start_row, start_col, rows, cols, 4)
        .expect("Failed to create initial state");

    println!("Walk parameters:");
    println!("  Grid: {}x{} (torus)", rows, cols);
    println!("  Start: ({}, {})", start_row, start_col);
    println!("  Coin: 4D Grover diffusion");
    println!("  Steps: {}\n", num_steps);

    // Simulate the walk
    let final_state = walk.simulate(initial_state, num_steps);

    // Get 2D probability distribution
    let probs = final_state
        .position_probabilities_2d(rows, cols)
        .expect("Failed to extract 2D probabilities");

    // Verify normalization
    let total_prob: f64 = probs.iter().sum();
    println!("Total probability: {:.10} (should be 1.0)\n", total_prob);

    // Print 2D probability distribution as ASCII heatmap
    println!(
        "Probability distribution after {} steps (heatmap):",
        num_steps
    );
    println!("{}", "=".repeat(cols * 2 + 3));

    let max_prob = probs.iter().cloned().fold(0.0_f64, f64::max);
    let chars = [' ', '.', ':', '+', '*', '#', '@'];

    for i in 0..rows {
        print!("{:2}|", i);
        for j in 0..cols {
            let prob = probs[[i, j]];
            let level = ((prob / max_prob) * (chars.len() - 1) as f64) as usize;
            let level = level.min(chars.len() - 1);
            print!("{} ", chars[level]);
        }
        println!("|");
    }

    println!("{}", "=".repeat(cols * 2 + 3));

    // Find peak positions
    println!("\nPositions with highest probability:");
    let mut positions: Vec<_> = probs.indexed_iter().collect();
    positions.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    for ((i, j), prob) in positions.iter().take(5) {
        if **prob > 0.001 {
            println!("  ({}, {}): {:.4}", i, j, prob);
        }
    }

    // Statistics
    let mean_row: f64 = probs.indexed_iter().map(|((i, _), &p)| i as f64 * p).sum();
    let mean_col: f64 = probs.indexed_iter().map(|((_, j), &p)| j as f64 * p).sum();

    println!("\nMean position: ({:.2}, {:.2})", mean_row, mean_col);

    // Demonstrate characteristic 2D spreading pattern
    println!("\n2D quantum walks exhibit diagonal spreading patterns");
    println!("due to the coupling of horizontal and vertical coin degrees of freedom.");
}
