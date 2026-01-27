# circulant-rs User Guide

**Version:** 0.2.0 | **Updated:** 2026-01-27 | **Reading time:** 20 min

> Comprehensive guide to circulant-rs for matrix operations, quantum walks, and image processing.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Core Module](#core-module)
   - [Circulant Matrices](#circulant-matrices)
   - [Block Circulant Matrices](#block-circulant-matrices)
5. [Physics Module](#physics-module)
   - [Quantum States](#quantum-states)
   - [Coin Operators](#coin-operators)
   - [1D Quantum Walks](#1d-quantum-walks)
   - [2D Quantum Walks](#2d-quantum-walks)
   - [Hamiltonians](#hamiltonians)
6. [Vision Module](#vision-module)
   - [Kernels](#kernels)
   - [BCCB Filters](#bccb-filters)
7. [Visualization Module](#visualization-module)
   - [Probability Plots](#probability-plots)
   - [Evolution Plots](#evolution-plots)
   - [Heatmaps](#heatmaps)
8. [Python Bindings](#python-bindings)
9. [Feature Flags](#feature-flags)
10. [Performance Tips](#performance-tips)
11. [Error Handling](#error-handling)
12. [Examples](#examples)

---

## Installation

Add circulant-rs to your `Cargo.toml`:

```toml
[dependencies]
circulant-rs = "0.2"
```

### With Specific Features

```toml
[dependencies]
# Core + physics (default)
circulant-rs = "0.2"

# All features
circulant-rs = { version = "0.2", features = ["physics", "vision", "visualize-bitmap", "parallel", "serde"] }

# Minimal (no default features)
circulant-rs = { version = "0.2", default-features = false }
```

---

## Quick Start

```rust
use circulant_rs::prelude::*;

fn main() -> Result<()> {
    // Create a circulant matrix from a generator
    let mut circ = Circulant::from_real(vec![4.0, -1.0, 0.0, -1.0])?;

    // Precompute FFT for faster repeated operations
    circ.precompute();

    // Multiply by a vector (O(N log N) instead of O(N²))
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let result = circ.mul_vec_real(&x)?;

    println!("Result: {:?}", result);
    Ok(())
}
```

---

## Core Concepts

### What is a Circulant Matrix?

A circulant matrix is a square matrix where each row is a cyclic shift of the row above it. Given a generator vector `[c₀, c₁, c₂, ..., cₙ₋₁]`, the matrix is:

```
┌                           ┐
│ c₀   c₁   c₂   ...  cₙ₋₁ │
│ cₙ₋₁ c₀   c₁   ...  cₙ₋₂ │
│ cₙ₋₂ cₙ₋₁ c₀   ...  cₙ₋₃ │
│ ...  ...  ...  ...  ...  │
│ c₁   c₂   c₃   ...  c₀   │
└                           ┘
```

### Why FFT?

Circulant matrices are diagonalized by the Discrete Fourier Transform (DFT). This means:
- **Eigenvalues**: Computed via FFT of the generator in O(N log N)
- **Matrix-vector multiplication**: O(N log N) instead of O(N²)
- **Matrix inverse**: O(N log N) via eigenvalue inversion

---

## Core Module

### Circulant Matrices

The `Circulant<T>` struct represents a 1D circulant matrix.

#### Creating Circulant Matrices

```rust
use circulant_rs::Circulant;
use num_complex::Complex;

// From real generator
let circ = Circulant::from_real(vec![1.0, 2.0, 3.0, 4.0])?;

// From complex generator
let gen = vec![
    Complex::new(1.0, 0.0),
    Complex::new(0.0, 1.0),
    Complex::new(-1.0, 0.0),
    Complex::new(0.0, -1.0),
];
let circ_complex = Circulant::new(gen)?;
```

#### Matrix Operations

```rust
use circulant_rs::prelude::*;

let mut circ = Circulant::from_real(vec![4.0, -1.0, 0.0, -1.0])?;

// Precompute FFT spectra for repeated operations
circ.precompute();

// Get eigenvalues
let eigenvalues = circ.eigenvalues();
println!("Eigenvalues: {:?}", eigenvalues);

// Multiply by real vector
let x = vec![1.0, 2.0, 3.0, 4.0];
let y = circ.mul_vec_real(&x)?;

// Multiply by complex vector
let x_complex: Vec<Complex<f64>> = x.iter().map(|&r| Complex::new(r, 0.0)).collect();
let y_complex = circ.mul_vec(&x_complex)?;

// Get matrix size
let n = circ.size();

// Get generator
let gen = circ.generator();
```

#### Solving Linear Systems

```rust
// Solve Cx = b for x
let b = vec![1.0, 0.0, 0.0, 0.0];
let x = circ.solve_real(&b)?;
```

### Block Circulant Matrices

The `BlockCirculant<T>` struct represents a 2D block circulant with circulant blocks (BCCB) matrix.

```rust
use circulant_rs::BlockCirculant;
use ndarray::Array2;
use num_complex::Complex;

// Create a 4x4 block circulant matrix with 2x2 blocks
let generator = Array2::from_shape_fn((4, 4), |(i, j)| {
    Complex::new(((i + j) % 4) as f64, 0.0)
});

let bccb = BlockCirculant::new(generator)?;

// Multiply by a 2D array
let input = Array2::from_shape_fn((4, 4), |(i, j)| {
    Complex::new((i * 4 + j) as f64, 0.0)
});

let output = bccb.mul_array(&input)?;
```

---

## Physics Module

The physics module provides tools for simulating discrete-time and continuous-time quantum walks.

### Quantum States

`QuantumState<T>` represents a quantum state in the position ⊗ coin Hilbert space.

#### Creating States

```rust
use circulant_rs::physics::QuantumState;

// Localized state at position 5 (1D walk, 2D coin)
let state = QuantumState::<f64>::localized(5, 101, 2)?;

// Superposition state (equal superposition of coin states)
let state = QuantumState::<f64>::superposition_at(50, 101, 2)?;

// 2D localized state at (row=5, col=5) on a 10x10 grid with 4D coin
let state_2d = QuantumState::<f64>::localized_2d(5, 5, 10, 10, 4)?;

// 2D superposition state
let state_2d = QuantumState::<f64>::superposition_2d(5, 5, 10, 10, 4)?;
```

#### State Properties

```rust
// Get position probabilities (1D)
let probs: Vec<f64> = state.position_probabilities();

// Get 2D position probabilities
let probs_2d: Array2<f64> = state_2d.position_probabilities_2d(10, 10)?;

// Check normalization
let norm_sq = state.norm_squared();
assert!((norm_sq - 1.0).abs() < 1e-10);

// Get dimensions
let num_positions = state.num_positions();
let coin_dim = state.coin_dim();
```

### Coin Operators

The `Coin` enum defines various coin operators for quantum walks.

#### Available Coins

```rust
use circulant_rs::physics::Coin;

// 2D coins (for 1D walks)
let hadamard = Coin::Hadamard;           // Standard Hadamard coin
let grover_2 = Coin::Grover(2);          // 2D Grover diffusion
let dft_2 = Coin::Dft(2);                // 2D DFT coin
let identity_2 = Coin::Identity(2);      // 2D Identity

// 4D coins (for 2D walks)
let grover_4d = Coin::grover_4d();       // 4D Grover diffusion
let dft_4d = Coin::dft_4d();             // 4D DFT coin
let hadamard_4d = Coin::hadamard_4d();   // H⊗H tensor product

// General dimension coins
let grover_n = Coin::Grover(8);          // 8D Grover
let dft_n = Coin::Dft(8);                // 8D DFT
```

#### Coin Properties

```rust
// Get coin dimension
let dim = coin.dimension();

// Get coin matrix
let matrix: Array2<Complex<f64>> = coin.to_matrix::<f64>();

// Check unitarity
let is_unitary = coin.is_unitary::<f64>(1e-10);
```

### 1D Quantum Walks

`CoinedWalk1D<T>` implements a discrete-time quantum walk on a cycle.

```rust
use circulant_rs::prelude::*;

// Create a walk on a 101-position cycle with Hadamard coin
let walk = CoinedWalk1D::<f64>::new(101, Coin::Hadamard)?;

// Create initial state
let initial = QuantumState::superposition_at(50, 101, 2)?;

// Simulate for 30 steps
let final_state = walk.simulate(initial, 30);

// Get probability distribution
let probs = final_state.position_probabilities();

// Or evolve step by step
let mut state = QuantumState::superposition_at(50, 101, 2)?;
for _ in 0..30 {
    walk.step(&mut state);
}
```

### 2D Quantum Walks

`CoinedWalk2D<T>` implements a discrete-time quantum walk on a 2D torus.

```rust
use circulant_rs::prelude::*;

// Create a walk on a 20x20 torus with 4D Grover coin
let walk = CoinedWalk2D::<f64>::new(20, 20, Coin::grover_4d())?;

// Create initial state (superposition at center)
let initial = QuantumState::superposition_2d(10, 10, 20, 20, 4)?;

// Simulate for 15 steps
let final_state = walk.simulate(initial, 15);

// Get 2D probability distribution
let probs_2d = final_state.position_probabilities_2d(20, 20)?;

// Access walk properties
let rows = walk.rows();
let cols = walk.cols();
let num_positions = walk.num_positions();  // rows * cols
```

### Hamiltonians

For continuous-time quantum walks, use the `Hamiltonian` trait and `CirculantHamiltonian`.

```rust
use circulant_rs::physics::{CirculantHamiltonian, Hamiltonian, QuantumState};

// Create cycle graph Laplacian (standard CTQW Hamiltonian)
let hamiltonian = CirculantHamiltonian::<f64>::cycle_graph(64);

// Or from custom generator (must be symmetric for Hermitian H)
let custom_h = CirculantHamiltonian::from_real_generator(
    vec![2.0, -1.0, 0.0, 0.0, 0.0, -1.0]
)?;

// Create localized initial state (coin_dim=1 for CTQW)
let mut state = QuantumState::localized(32, 64, 1)?;

// Propagate by time t: |ψ(t)⟩ = exp(-iHt)|ψ(0)⟩
hamiltonian.propagate(&mut state, 5.0);

// Get probabilities after evolution
let probs = state.position_probabilities();

// Get eigenvalues
let eigenvalues = hamiltonian.eigenvalues();

// Get dimension
let dim = hamiltonian.dimension();
```

---

## Vision Module

The vision module provides FFT-accelerated image convolution using BCCB matrices.

### Kernels

The `Kernel<T>` struct represents 2D convolution kernels.

#### Built-in Kernels

```rust
use circulant_rs::vision::Kernel;

// Gaussian blur (sigma=2.0, kernel size=9x9)
let gaussian = Kernel::<f64>::gaussian(2.0, 9)?;

// Edge detection kernels
let sobel_x = Kernel::<f64>::sobel_x();      // Horizontal edges
let sobel_y = Kernel::<f64>::sobel_y();      // Vertical edges
let laplacian = Kernel::<f64>::laplacian();  // All edges

// Box blur (5x5 averaging filter)
let box_blur = Kernel::<f64>::box_blur(5)?;
```

#### Custom Kernels

```rust
use ndarray::Array2;

// Custom kernel from coefficients
let coeffs = Array2::from_shape_vec((3, 3), vec![
    0.0, -1.0, 0.0,
    -1.0, 4.0, -1.0,
    0.0, -1.0, 0.0,
])?;
let custom = Kernel::<f64>::custom(coeffs)?;
```

#### Kernel Properties

```rust
// Get kernel size
let (rows, cols) = kernel.size();

// Get kernel data (complex array)
let data: &Array2<Complex<f64>> = kernel.data();

// Get kernel name
let name: &str = kernel.name();
```

### BCCB Filters

`BCCBFilter<T>` applies kernels to images using FFT-based convolution.

```rust
use circulant_rs::vision::{BCCBFilter, Kernel};
use ndarray::Array2;

// Create a Gaussian kernel
let kernel = Kernel::<f64>::gaussian(2.0, 9)?;

// Create filter for 256x256 images
let filter = BCCBFilter::new(kernel, 256, 256)?;

// Apply to an image
let image: Array2<f64> = load_image_somehow();
let blurred = filter.apply(&image)?;

// For complex images
let complex_image: Array2<Complex<f64>> = /* ... */;
let result = filter.apply_complex(&complex_image)?;
```

#### Edge Detection Example

```rust
use circulant_rs::vision::{BCCBFilter, Kernel};

// Create Sobel filters
let sobel_x = BCCBFilter::new(Kernel::sobel_x(), width, height)?;
let sobel_y = BCCBFilter::new(Kernel::sobel_y(), width, height)?;

// Compute gradients
let gx = sobel_x.apply(&image)?;
let gy = sobel_y.apply(&image)?;

// Compute gradient magnitude
let magnitude: Array2<f64> = gx.iter()
    .zip(gy.iter())
    .map(|(&x, &y)| (x*x + y*y).sqrt())
    .collect::<Vec<_>>()
    .into();
```

---

## Visualization Module

The visualization module provides plotting functions using the `plotters` crate.

### Setup

Enable visualization in `Cargo.toml`:

```toml
[dependencies]
circulant-rs = { version = "0.2", features = ["visualize-bitmap"] }
# Or for SVG output:
# circulant-rs = { version = "0.2", features = ["visualize-svg"] }
```

### Plot Configuration

```rust
use circulant_rs::visualize::PlotConfig;

// Default configuration
let config = PlotConfig::default();

// Custom configuration
let config = PlotConfig::with_title("My Plot")
    .dimensions(1024, 768)
    .labels("X Axis", "Y Axis");
```

### Probability Plots

```rust
use circulant_rs::visualize::{plot_probabilities, PlotConfig};

let probs: Vec<f64> = state.position_probabilities();
let config = PlotConfig::with_title("Probability Distribution");

// Save as PNG (requires visualize-bitmap feature)
plot_probabilities(&probs, "output.png", &config)?;

// Save as SVG (requires visualize-svg feature)
plot_probabilities(&probs, "output.svg", &config)?;
```

### Evolution Plots

Plot multiple probability distributions over time:

```rust
use circulant_rs::visualize::{plot_walk_evolution, PlotConfig};

// Collect probability distributions at different time steps
let mut distributions: Vec<Vec<f64>> = Vec::new();
let mut state = initial_state;

for _ in 0..5 {
    walk.step(&mut state);
    distributions.push(state.position_probabilities());
}

let config = PlotConfig::with_title("Quantum Walk Evolution");
plot_walk_evolution(&distributions, "evolution.png", &config)?;
```

### Heatmaps

Plot 2D probability distributions:

```rust
use circulant_rs::visualize::{plot_heatmap, PlotConfig};
use ndarray::Array2;

let probs_2d: Array2<f64> = state.position_probabilities_2d(rows, cols)?;
let config = PlotConfig::with_title("2D Probability Heatmap")
    .labels("Column", "Row");

plot_heatmap(&probs_2d, "heatmap.png", &config)?;
```

---

## Python Bindings

circulant-rs provides Python bindings via PyO3.

### Building the Python Package

```bash
# Install maturin
pip install maturin

# Build and install in development mode
maturin develop --features python

# Build wheel for distribution
maturin build --features python --release
```

### Using in Python

```python
from circulant_rs import (
    Circulant,
    QuantumState,
    Coin,
    CoinedWalk1D,
    CoinedWalk2D,
    CirculantHamiltonian,
)

# Circulant matrix operations
circ = Circulant.from_real([4.0, -1.0, 0.0, -1.0])
circ.precompute()
result = circ.mul_vec([1.0, 2.0, 3.0, 4.0])
eigenvalues = circ.eigenvalues()

# 1D Quantum walk
walk = CoinedWalk1D(101, Coin.hadamard())
state = QuantumState.localized(50, 101, 2)
final_state = walk.simulate(state, 30)
probs = final_state.position_probabilities()

# 2D Quantum walk
walk_2d = CoinedWalk2D(20, 20, Coin.grover_4d())
state_2d = QuantumState.localized_2d(10, 10, 20, 20, 4)
final_state_2d = walk_2d.simulate(state_2d, 15)

# Continuous-time quantum walk
hamiltonian = CirculantHamiltonian.cycle_graph(64)
state = QuantumState.localized(32, 64, 1)
hamiltonian.propagate(state, 5.0)
probs = state.position_probabilities()
```

### Available Python Classes

| Class | Description |
|-------|-------------|
| `Circulant` | Circulant matrix operations |
| `QuantumState` | Quantum state representation |
| `Coin` | Coin operators (Hadamard, Grover, DFT) |
| `CoinedWalk1D` | 1D discrete-time quantum walk |
| `CoinedWalk2D` | 2D discrete-time quantum walk |
| `CirculantHamiltonian` | Circulant Hamiltonian for CTQW |

---

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `std` | Standard library support | Yes |
| `physics` | Quantum walk simulations | Yes |
| `parallel` | Parallel computation (rayon) | Yes |
| `serde` | Serialization support | Yes |
| `vision` | Image processing module | No |
| `visualize` | Base visualization | No |
| `visualize-bitmap` | PNG output | No |
| `visualize-svg` | SVG output | No |
| `python` | Python bindings | No |

### Minimal Build

```toml
[dependencies]
circulant-rs = { version = "0.2", default-features = false }
```

### Full Build

```toml
[dependencies]
circulant-rs = { version = "0.2", features = [
    "physics",
    "vision",
    "visualize-bitmap",
    "visualize-svg",
    "parallel",
    "serde",
] }
```

---

## Performance Tips

### 1. Use Precomputation

For repeated operations with the same matrix:

```rust
let mut circ = Circulant::from_real(generator)?;
circ.precompute();  // Compute FFT spectra once

// Now multiplications are faster
for x in vectors {
    let y = circ.mul_vec_real(&x)?;
}
```

### 2. Enable Parallel Feature

For large matrices, enable parallel computation:

```toml
[dependencies]
circulant-rs = { version = "0.2", features = ["parallel"] }
```

### 3. Use Release Mode

Always benchmark and run production code in release mode:

```bash
cargo run --release --example quantum_walk_1d
```

### 4. Batch Operations

When possible, batch multiple operations:

```rust
// Instead of many single steps
for _ in 0..100 {
    walk.step(&mut state);
}

// Use simulate for better performance
let final_state = walk.simulate(initial, 100);
```

### 5. Choose Appropriate Precision

Use `f64` for most applications. Only use `f32` if memory is critical:

```rust
// Standard precision (recommended)
let state = QuantumState::<f64>::localized(50, 101, 2)?;

// Lower precision (saves memory)
let state = QuantumState::<f32>::localized(50, 101, 2)?;
```

---

## Error Handling

circulant-rs uses a unified error type `CirculantError`:

```rust
use circulant_rs::error::{CirculantError, Result};

fn my_function() -> Result<()> {
    let circ = Circulant::from_real(vec![])?;  // Returns EmptyGenerator error
    Ok(())
}

// Handle specific errors
match Circulant::from_real(vec![]) {
    Ok(circ) => { /* use circ */ },
    Err(CirculantError::EmptyGenerator) => {
        println!("Generator cannot be empty");
    },
    Err(CirculantError::DimensionMismatch { expected, got }) => {
        println!("Expected {} elements, got {}", expected, got);
    },
    Err(e) => {
        println!("Other error: {}", e);
    }
}
```

### Error Types

| Error | Description |
|-------|-------------|
| `EmptyGenerator` | Generator vector is empty |
| `DimensionMismatch` | Vector/matrix dimensions don't match |
| `InvalidCoinDimension` | Coin dimension is invalid |
| `InvalidPosition` | Position out of bounds |
| `InvalidKernel` | Kernel dimensions invalid |
| `ImageDimensionMismatch` | Image doesn't match filter size |
| `NotHermitian` | Hamiltonian is not Hermitian |
| `InvalidTime` | Time parameter is invalid |
| `VisualizationError` | Plotting failed |

---

## Examples

### Run Built-in Examples

```bash
# 1D Quantum walk
cargo run --example quantum_walk_1d --features physics

# 2D Quantum walk
cargo run --example quantum_walk_2d --features physics

# Continuous-time quantum walk
cargo run --example continuous_walk --features physics

# Image blur
cargo run --example image_blur --features vision

# Edge detection
cargo run --example edge_detection --features vision

# Visualization (requires fonts)
cargo run --example walk_visualization --features "physics visualize-bitmap"
```

### Complete Example: Quantum Walk Analysis

```rust
use circulant_rs::prelude::*;

fn main() -> Result<()> {
    // Setup
    let n = 101;
    let center = n / 2;
    let steps = 50;

    // Create walk with Hadamard coin
    let walk = CoinedWalk1D::<f64>::new(n, Coin::Hadamard)?;

    // Create symmetric initial state
    let initial = QuantumState::superposition_at(center, n, 2)?;

    // Simulate
    let final_state = walk.simulate(initial, steps);

    // Analyze
    let probs = final_state.position_probabilities();

    // Statistics
    let mean: f64 = probs.iter()
        .enumerate()
        .map(|(i, &p)| i as f64 * p)
        .sum();

    let variance: f64 = probs.iter()
        .enumerate()
        .map(|(i, &p)| {
            let diff = i as f64 - mean;
            diff * diff * p
        })
        .sum();

    let std_dev = variance.sqrt();

    println!("After {} steps:", steps);
    println!("  Mean position: {:.2}", mean);
    println!("  Std deviation: {:.2}", std_dev);
    println!("  Ballistic ratio: {:.2}", std_dev / steps as f64);

    Ok(())
}
```

### Complete Example: Image Processing

```rust
use circulant_rs::vision::{BCCBFilter, Kernel};
use ndarray::Array2;

fn main() -> circulant_rs::error::Result<()> {
    // Create test image (64x64)
    let size = 64;
    let mut image = Array2::zeros((size, size));

    // Add a bright rectangle
    for i in 20..44 {
        for j in 20..44 {
            image[[i, j]] = 1.0;
        }
    }

    // Apply Gaussian blur
    let gaussian = Kernel::<f64>::gaussian(2.0, 9)?;
    let blur_filter = BCCBFilter::new(gaussian, size, size)?;
    let blurred = blur_filter.apply(&image)?;

    // Detect edges with Sobel
    let sobel_x = BCCBFilter::new(Kernel::sobel_x(), size, size)?;
    let sobel_y = BCCBFilter::new(Kernel::sobel_y(), size, size)?;

    let gx = sobel_x.apply(&image)?;
    let gy = sobel_y.apply(&image)?;

    // Gradient magnitude
    let edges: Vec<f64> = gx.iter()
        .zip(gy.iter())
        .map(|(&x, &y)| (x*x + y*y).sqrt())
        .collect();

    println!("Image processed successfully!");
    println!("Blurred max value: {:.4}", blurred.iter().cloned().fold(0.0_f64, f64::max));
    println!("Edge max value: {:.4}", edges.iter().cloned().fold(0.0_f64, f64::max));

    Ok(())
}
```

---

## Further Resources

- **API Documentation**: `cargo doc --all-features --open`
- **Source Code**: [GitHub Repository](https://github.com/your-repo/circulant-rs)
- **Examples**: See the `examples/` directory
- **Benchmarks**: Run with `cargo bench`

## License

circulant-rs is dual-licensed under MIT and Apache-2.0.
