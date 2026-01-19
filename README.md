# circulant-rs

A high-performance Rust library for block-circulant matrix operations and quantum walk simulations.

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)]()
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)]()

## Overview

`circulant-rs` exploits the mathematical structure of circulant matrices to achieve **O(N log N)** complexity instead of **O(N²)** for matrix-vector multiplication. This is accomplished through the FFT diagonalization property: every circulant matrix can be diagonalized by the Discrete Fourier Transform.

The library provides:

- **1D Circulant matrices** - For signal processing, convolution, and 1D quantum walks
- **2D Block Circulant (BCCB) matrices** - For image processing and 2D periodic systems
- **Quantum Walk simulation** - Discrete-time coined quantum walks with FFT-accelerated evolution
- **Full serialization support** - Save and restore quantum states and walk configurations

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
circulant-rs = "0.1"
```

### Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | Yes | Standard library support |
| `physics` | Yes | Quantum walk simulation module |
| `parallel` | Yes | Rayon-based parallelization |
| `serde` | Yes | Serialization with serde/bincode |

## Quick Start

### 1D Circulant Matrix Multiplication

```rust
use circulant_rs::prelude::*;
use num_complex::Complex;

fn main() -> Result<(), CirculantError> {
    // Create a circulant matrix from its first row
    let generator = vec![1.0, 2.0, 3.0, 4.0];
    let matrix = Circulant::from_real(generator)?;

    // Multiply by a vector - O(N log N) via FFT!
    let x = vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
    ];
    let result = matrix.mul_vec(&x)?;

    println!("Result: {:?}", result);
    Ok(())
}
```

### Quantum Walk Simulation

```rust
use circulant_rs::physics::{CoinedWalk1D, Coin, QuantumState, QuantumWalk};

fn main() {
    // Create a quantum walk on a 256-node ring with Hadamard coin
    let walk = CoinedWalk1D::<f64>::new(256, Coin::Hadamard);

    // Initialize walker at position 128
    let initial_state = QuantumState::localized(128, 256, 2).unwrap();

    // Simulate 100 steps
    let final_state = walk.simulate(initial_state, 100);

    // Get probability distribution
    let probabilities = final_state.position_probabilities();

    // Quantum walks show ballistic spreading (linear in time)
    // compared to classical random walks (sqrt of time)
    println!("Norm preserved: {}", final_state.norm_squared());
}
```

### 2D Block Circulant Operations

```rust
use circulant_rs::core::BlockCirculant;
use ndarray::Array2;
use num_complex::Complex;

fn main() {
    // Create a 2D convolution kernel
    let kernel = Array2::from_shape_vec((3, 3), vec![
        Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(1.0, 0.0),
        Complex::new(2.0, 0.0), Complex::new(4.0, 0.0), Complex::new(2.0, 0.0),
        Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(1.0, 0.0),
    ]).unwrap();

    // Create BCCB filter (zero-padded to output size)
    let filter = BlockCirculant::from_kernel(kernel, 64, 64).unwrap();

    // Apply to 2D data - O(N log N) via 2D FFT!
    let input = Array2::zeros((64, 64));
    let output = filter.mul_array(&input).unwrap();
}
```

## Performance

The FFT-based approach provides dramatic speedups for large matrices:

| Size N | Dense O(N²) | Circulant O(N log N) | Speedup |
|--------|-------------|---------------------|---------|
| 1,024 | 1M ops | 10K ops | 100x |
| 65,536 | 4B ops | 1M ops | 4,000x |
| 1M | 1T ops | 20M ops | 50,000x |

Memory usage is also reduced from O(N²) to O(N) since only the generator needs to be stored.

## Mathematical Background

### Circulant Matrix Structure

A circulant matrix is defined by its first row `[c₀, c₁, ..., cₙ₋₁]`:

```
| c₀    c₁    c₂    ...  cₙ₋₁ |
| cₙ₋₁  c₀    c₁    ...  cₙ₋₂ |
| cₙ₋₂  cₙ₋₁  c₀    ...  cₙ₋₃ |
| ...                         |
| c₁    c₂    c₃    ...  c₀   |
```

### FFT Diagonalization

Every circulant matrix C can be diagonalized by the DFT matrix F:

```
C = F⁻¹ · D · F
```

Where D is diagonal with eigenvalues equal to the DFT of the generator. This enables O(N log N) multiplication:

```
y = C·x = F⁻¹ · D · F · x = IFFT(FFT(c) ⊙ FFT(x))
```

### Quantum Walk Evolution

The discrete-time coined quantum walk evolves as:

1. **Coin flip**: Apply coin operator C to internal degree of freedom
2. **Shift**: Move walker based on coin state

The shift operator is a circulant matrix, enabling FFT-accelerated simulation.

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed technical documentation including:

- Module structure and dependencies
- Type system and trait hierarchy
- FFT backend abstraction
- Quantum walk implementation details
- Mathematical derivations

## Modules

| Module | Description |
|--------|-------------|
| `core` | `Circulant<T>` and `BlockCirculant<T>` types |
| `fft` | FFT backend trait and RustFFT implementation |
| `traits` | `Scalar`, `CirculantOps`, `BlockOps` traits |
| `physics` | Quantum state, coins, and walk simulation |
| `error` | Error types and Result alias |
| `prelude` | Convenient re-exports |

## Roadmap

### v0.1.0 (Current)
- [x] 1D Circulant with FFT multiplication
- [x] 2D Block Circulant (BCCB)
- [x] Quantum walk simulation (1D coined walks)
- [x] Serde serialization
- [x] Rayon parallelization

### v0.2.0 (Planned)
- [ ] Vision module (image filtering with BCCB)
- [ ] 2D quantum walks on torus
- [ ] Hamiltonian module for physics
- [ ] Visualization with plotters

### v0.3.0 (Future)
- [ ] GPU acceleration (wgpu/cuda)
- [ ] N-dimensional circulant tensors
- [ ] Circulant neural network layers

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
