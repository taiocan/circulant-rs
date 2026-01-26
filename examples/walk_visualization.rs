//! Example: Quantum Walk Visualization
//!
//! This example demonstrates visualization of quantum walk probability distributions
//! using the plotters crate.
//!
//! Run with: `cargo run --example walk_visualization --features "physics visualize-bitmap"`

#[cfg(all(feature = "physics", feature = "visualize-bitmap"))]
fn main() {
    use circulant_rs::physics::{Coin, CoinedWalk1D, QuantumState, QuantumWalk};
    use circulant_rs::visualize::{plot_probabilities, plot_walk_evolution, PlotConfig};

    println!("=== Quantum Walk Visualization ===\n");

    // Parameters
    let num_positions = 101;
    let start_position = 50;
    let num_steps = 30;

    // Create the quantum walk
    let walk = CoinedWalk1D::<f64>::new(num_positions, Coin::Hadamard)
        .expect("Failed to create quantum walk");

    // Create initial state
    let initial_state = QuantumState::superposition_at(start_position, num_positions, 2)
        .expect("Failed to create initial state");

    println!("Simulating quantum walk...");
    println!("  Positions: {}", num_positions);
    println!("  Start: {}", start_position);
    println!("  Steps: {}", num_steps);

    // Collect probability distributions at various time steps
    let time_steps = [5, 10, 15, 20, 25, 30];
    let mut probabilities = Vec::new();

    let mut state = initial_state.clone();
    let mut current_step = 0;

    for &target_step in &time_steps {
        while current_step < target_step {
            walk.step(&mut state);
            current_step += 1;
        }
        probabilities.push(state.position_probabilities());
    }

    // Plot final distribution
    let final_probs = &probabilities[probabilities.len() - 1];
    let config = PlotConfig::with_title(format!("Quantum Walk at t={}", num_steps))
        .dimensions(800, 600)
        .labels("Position", "Probability");

    println!("Generating probability plot...");
    match plot_probabilities(final_probs, "quantum_walk_probability.png", &config) {
        Ok(()) => println!("  Saved: quantum_walk_probability.png"),
        Err(e) => eprintln!("  Error: {}", e),
    }

    // Plot evolution over time
    let evolution_config = PlotConfig::with_title("Quantum Walk Evolution")
        .dimensions(1000, 600)
        .labels("Position", "Probability");

    println!("Generating evolution plot...");
    match plot_walk_evolution(
        &probabilities,
        "quantum_walk_evolution.png",
        &evolution_config,
    ) {
        Ok(()) => println!("  Saved: quantum_walk_evolution.png"),
        Err(e) => eprintln!("  Error: {}", e),
    }

    // Show some statistics
    println!("\nProbability statistics at each time step:");
    for (i, probs) in probabilities.iter().enumerate() {
        let max_prob = probs.iter().cloned().fold(0.0_f64, f64::max);
        let (peak_pos, _) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let mean: f64 = probs.iter().enumerate().map(|(j, &p)| j as f64 * p).sum();
        let variance: f64 = probs
            .iter()
            .enumerate()
            .map(|(j, &p)| {
                let diff = j as f64 - mean;
                diff * diff * p
            })
            .sum();
        let std_dev = variance.sqrt();

        println!(
            "  t={:2}: max={:.4} at pos={}, std={:.2}",
            time_steps[i], max_prob, peak_pos, std_dev
        );
    }

    println!("\nVisualization complete!");
    println!("The quantum walk shows characteristic ballistic spreading");
    println!("with interference patterns, unlike diffusive classical walks.");
}

#[cfg(not(all(feature = "physics", feature = "visualize-bitmap")))]
fn main() {
    println!("This example requires the 'physics' and 'visualize-bitmap' features.");
    println!(
        "Run with: cargo run --example walk_visualization --features \"physics visualize-bitmap\""
    );
}
