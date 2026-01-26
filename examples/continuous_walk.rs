//! Example: Continuous-Time Quantum Walk
//!
//! This example demonstrates a continuous-time quantum walk using a circulant Hamiltonian.
//! The walker starts at a single node and evolves under exp(-iHt).
//!
//! Run with: `cargo run --example continuous_walk --features physics`

use circulant_rs::physics::{CirculantHamiltonian, Hamiltonian, QuantumState};

fn main() {
    println!("=== Continuous-Time Quantum Walk ===\n");

    // Parameters
    let n = 64; // Number of vertices in the cycle
    let start = 32; // Starting position (center)

    // Create the cycle graph Hamiltonian (Laplacian)
    let hamiltonian =
        CirculantHamiltonian::<f64>::cycle_graph(n).expect("Failed to create Hamiltonian");

    println!("System parameters:");
    println!("  Vertices: {} (cycle graph)", n);
    println!("  Start position: {}", start);
    println!("  Hamiltonian: Cycle graph Laplacian");
    println!();

    // Show eigenvalue spectrum
    let eigenvalues = hamiltonian.eigenvalues();
    let min_eig = eigenvalues
        .iter()
        .map(|c| c.re)
        .fold(f64::INFINITY, f64::min);
    let max_eig = eigenvalues
        .iter()
        .map(|c| c.re)
        .fold(f64::NEG_INFINITY, f64::max);

    println!("Eigenvalue spectrum:");
    println!("  Min eigenvalue: {:.4}", min_eig);
    println!("  Max eigenvalue: {:.4}", max_eig);
    println!("  (λ_k = 2(1 - cos(2πk/n)) for cycle graph)");
    println!();

    // Evolve for several time steps
    let times = [0.5, 1.0, 2.0, 5.0, 10.0];

    println!("Evolution snapshots:");
    println!("{}", "=".repeat(60));

    for &t in &times {
        // Create fresh initial state (localized at start)
        let mut state = QuantumState::localized(start, n, 1).expect("Failed to create state");

        // Propagate
        hamiltonian.propagate(&mut state, t);

        // Get probabilities
        let probs = state.position_probabilities();

        // Find peak and spread
        let (peak_pos, peak_prob) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

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

        println!(
            "t = {:5.2}: peak at {:2} ({:.4}), std = {:.2}",
            t, peak_pos, peak_prob, std_dev
        );
    }

    println!("{}", "=".repeat(60));

    // Detailed visualization for t = 5.0
    let t_detail = 5.0;
    let mut state = QuantumState::localized(start, n, 1).expect("Failed to create state");
    hamiltonian.propagate(&mut state, t_detail);
    let probs = state.position_probabilities();

    println!(
        "\nProbability distribution at t = {} (ASCII histogram):",
        t_detail
    );

    let max_prob = probs.iter().cloned().fold(0.0_f64, f64::max);
    let scale = 40.0 / max_prob;

    for (pos, &prob) in probs.iter().enumerate() {
        if prob > 0.005 {
            let bar_len = (prob * scale) as usize;
            let bar = "#".repeat(bar_len);
            println!("{:3}: {:6.4} |{}", pos, prob, bar);
        }
    }

    println!("\nContinuous-time quantum walks show wave-like spreading");
    println!("with interference patterns, unlike diffusive classical walks.");
}
