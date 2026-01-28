# circulant-rs

**Version:** 1.0.0 | **Updated:** 2026-01-28 | **Reading time:** 5 min

A high-performance Rust library for block-circulant tensor operations and quantum walk simulations.

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)]()
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)]()

---

<table>
<tr>
<td width="33%" align="center">

### ⚡ 676× Faster

FFT-based O(N log N) vs naive O(N²)

*Scales to 50,000× at N=1M*

</td>
<td width="33%" align="center">

### 💾 99.9999% Less Memory

N=1M: Dense 7.5 TB → circulant-rs 8 MB

*From impossible to trivial*

</td>

<td width="33%" align="center">

### 🔬 1000× Larger Simulations

Quantum walks with N=1,000,000 positions

*Previously limited to N≈1,000*

</td>
</tr>
</table>

---

## Overview

`circulant-rs` exploits the mathematical structure of circulant matrices to achieve **O(N log N)** complexity instead of **O(N²)** for matrix-vector multiplication. This is accomplished through the FFT diagonalization property: every circulant matrix can be diagonalized by the Discrete Fourier Transform.

The library provides:

- **N-D Circulant Tensors** (New in 1.0) - Generic N-dimensional circulant operations with O(N log N) complexity
- **1D Circulant matrices** - For signal processing, convolution, and 1D quantum walks
- **2D Block Circulant (BCCB) matrices** - For image processing and 2D periodic systems
- **3D/4D Tensor operations** - For volumetric data, video processing, and scientific computing
- **Quantum Walk simulation** - Discrete-time coined quantum walks with FFT-accelerated evolution
- **Full serialization support** - Save and restore quantum states and walk configurations

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
circulant-rs = "1.0"
```

### Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | Yes | Standard library support |
| `physics` | Yes | Quantum walk simulation module |
| `parallel` | Yes | Rayon-based parallelization |
| `serde` | Yes | Serialization with serde/bincode |
| `vision` | No | Image processing with BCCB filters |
| `visualize` | No | Visualization with plotters |
| `visualize-svg` | No | SVG output backend |
| `visualize-bitmap` | No | PNG/bitmap output backend |
| `python` | No | Python bindings via PyO3 |

## Quick Start

### 1D Circulant Matrix Multiplication

```rust
use circulant_rs::prelude::*;
use num_complex::Complex;

fn main() -> Result<()> {
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
use circulant_rs::prelude::*;

fn main() -> Result<()> {
    // Create a quantum walk on a 256-node ring with Hadamard coin
    let walk = CoinedWalk1D::<f64>::new(256, Coin::Hadamard)?;

    // Initialize walker at position 128
    let initial_state = QuantumState::localized(128, 256, 2)?;

    // Simulate 100 steps
    let final_state = walk.simulate(initial_state, 100);

    // Get probability distribution
    let _probabilities = final_state.position_probabilities();

    // Quantum walks show ballistic spreading (linear in time)
    // compared to classical random walks (sqrt of time)
    println!("Norm preserved: {}", final_state.norm_squared());
    Ok(())
}
```

### N-D Circulant Tensor Operations (New in 1.0)

```rust
use circulant_rs::{CirculantTensor, Circulant3D, TensorOps};
use ndarray::{ArrayD, IxDyn};
use num_complex::Complex;

fn main() -> circulant_rs::Result<()> {
    // Create a 3D circulant tensor (e.g., for volumetric convolution)
    let shape = [16, 16, 16];
    let gen = ArrayD::from_shape_vec(
        IxDyn(&shape),
        (0..4096).map(|i| Complex::new((i as f64 * 0.01).sin(), 0.0)).collect(),
    ).unwrap();

    let mut tensor: Circulant3D<f64> = CirculantTensor::new(gen)?;

    // Precompute for repeated operations
    tensor.precompute();

    // Multiply - O(N log N) via separable N-D FFT
    let input = ArrayD::from_elem(IxDyn(&shape), Complex::new(1.0, 0.0));
    let result = tensor.mul_tensor(&input)?;

    println!("3D tensor multiplication complete: {} elements", result.len());
    Ok(())
}
```

### 2D Block Circulant Operations (Legacy)

```rust
use circulant_rs::core::BlockCirculant;
use circulant_rs::traits::BlockOps; // Nujno za mul_array!
use ndarray::Array2; // ndarray v0.16.1 !!!
use num_complex::Complex;

fn main() {
    // 1. Ustvarjanje 2D konvolucijskega jedra
    let kernel = Array2::from_shape_vec((3, 3), vec![
        Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(1.0, 0.0),
        Complex::new(2.0, 0.0), Complex::new(4.0, 0.0), Complex::new(2.0, 0.0),
        Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(1.0, 0.0),
    ]).unwrap();

    // 2. Ustvarjanje BCCB filtra (zero-padding na 64x64)
    // To uporabi 2D FFT za izjemno hitro procesiranje
    let filter = BlockCirculant::from_kernel(kernel, 64, 64).unwrap();

    // 3. Priprava vhodnih podatkov (npr. slika 64x64)
    let input: Array2<Complex<f64>> = Array2::zeros((64, 64));
    
    // 4. Množenje z uporabo BlockOps traita
    let output = filter.mul_array(&input).unwrap();

    println!("Konvolucija uspešno izvedena. Velikost izhoda: {:?}", output.dim());
}
```

## Performance

### Measured Benchmarks: FFT vs Naive O(N²)

| Size N | circulant-rs (FFT) | Naive O(N²) | **Speedup** |
|-------:|-------------------:|------------:|------------:|
| 64 | 573 ns | 18.8 µs | **33×** |
| 256 | 3.2 µs | 335 µs | **105×** |
| 1,024 | 12.3 µs | 4.76 ms | **387×** |
| 2,048 | 28.4 µs | 19.2 ms | **676×** |
| 65,536 | ~800 µs | ~68 min* | **~5,000×** |
| 1,048,576 | ~15 ms | ~17 days* | **~50,000×** |

*Extrapolated from measured scaling

### Quantum Walk Simulation (100 steps, Hadamard coin)

| Positions | Total Time | Per Step | Throughput |
|----------:|------------|----------|------------|
| 256 | 578 µs | 5.8 µs | 44M pos/sec |
| 1,024 | 2.44 ms | 24.4 µs | 42M pos/sec |
| 4,096 | 11.9 ms | 119 µs | 34M pos/sec |
| 16,384 | 71.3 ms | 713 µs | 23M pos/sec |
| 65,536 | 505 ms | 5.05 ms | 13M pos/sec |

### Memory Scaling

| Size N | Generator Only | Dense Matrix | **Reduction** |
|-------:|---------------:|-------------:|--------------:|
| 1,000 | 16 KB | 16 MB | 1,000× |
| 10,000 | 160 KB | 1.6 GB | 10,000× |
| 100,000 | 1.6 MB | 160 GB | 100,000× |
| 1,000,000 | 16 MB | 16 TB | **1,000,000×** |

## Use Cases

| Domain | Application | Why circulant-rs Helps |
|--------|-------------|------------------------|
| **Quantum Computing** | Quantum walk simulations on rings | Simulate N=1M positions vs N=1K with dense matrices |
| **Signal Processing** | Real-time periodic FIR filtering | O(N log N) convolution at audio sample rates |
| **Image Processing** | 2D periodic convolution (blur, edge detection) | 4K image filtering in milliseconds |
| **Cryptography** | Lattice-based schemes (NTRU) | Efficient polynomial multiplication |
| **Machine Learning** | Circulant neural network layers | Reduced parameter count with structured weights |

See [docs/OVERVIEW.md](./docs/OVERVIEW.md) for detailed use case examples with code.

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

See [ARCHITECTURE.md](./docs/ARCHITECTURE.md) for detailed technical documentation including:

- Module structure and dependencies
- Type system and trait hierarchy
- FFT backend abstraction
- Quantum walk implementation details
- Mathematical derivations

## Modules

| Module | Description |
|--------|-------------|
| `core` | `CirculantTensor<T, D>`, `Circulant1D/2D/3D/4D<T>` types (1.0), plus legacy `Circulant<T>` and `BlockCirculant<T>` |
| `fft` | FFT backend trait and RustFFT implementation |
| `traits` | `TensorOps<T, D>` (1.0), `Scalar`, `CirculantOps`, `BlockOps` traits |
| `physics` | Quantum state, coins, 1D/2D walks, Hamiltonian |
| `vision` | BCCB image filtering and kernels |
| `visualize` | Quantum state and heatmap plotting |
| `python` | PyO3 bindings for Python |
| `error` | Error types and Result alias |
| `prelude` | Convenient re-exports |

## Running Benchmarks

You can reproduce all benchmark results on your own machine.

### Prerequisites

```bash
# Rust 1.70+ required
rustc --version

# For Python comparison (optional)
pip install numpy scipy
```

### Rust Benchmarks (Criterion)

```bash
# Run all benchmarks (includes FFT vs naive, quantum walk, 2D BCCB)
cargo bench --bench scaling_benchmark --features "physics parallel serde"

# Run core FFT multiplication benchmark only
cargo bench --bench fft_multiply

# View results in browser (after running benchmarks)
open target/criterion/report/index.html   # macOS
xdg-open target/criterion/report/index.html  # Linux
```

**Benchmark output location:** `target/criterion/`

Each benchmark group generates:
- `report/index.html` - Interactive HTML report with graphs
- `<group>/report/index.html` - Detailed per-group analysis
- Raw data in JSON format for custom analysis

### Python Comparison Benchmarks

```bash
cd benchmarks/python

# Run all comparisons (NumPy FFT vs SciPy dense vs naive)
python compare_circulant.py

# Results saved to results_1d.json
```

**Output includes:**
- 1D multiplication timing across sizes (64 to 262,144)
- Accuracy verification (FFT vs naive)
- Memory usage comparison
- Quantum walk simulation timing

### Quick Validation

```bash
# Run test suite to verify correctness
cargo test --all-features

# Run example to see library in action
cargo run --example quantum_walk_1d --features physics
```

### Interpreting Results

- **Speedup** = Naive time / FFT time (higher is better)
- **Throughput** = Elements processed per second
- Criterion reports include statistical analysis (mean, std dev, outliers)
- Compare your results with the tables above; relative speedups should be consistent across hardware

For detailed methodology, see [docs/BENCHMARKS.md](./docs/BENCHMARKS.md).

## Roadmap

### v1.0.0 (Current)
- [x] **N-dimensional circulant tensors** (`CirculantTensor<T, D>`)
- [x] **Parallel N-D FFT** via rayon
- [x] **Type aliases**: `Circulant1D`, `Circulant2D`, `Circulant3D`, `Circulant4D`
- [x] 1D Circulant with FFT multiplication
- [x] 2D Block Circulant (BCCB)
- [x] Quantum walk simulation (1D/2D coined walks)
- [x] Serde serialization
- [x] Rayon parallelization
- [x] Vision module (image filtering with BCCB)
- [x] Continuous-time walks (Hamiltonian module)
- [x] Visualization with plotters
- [x] Python bindings via PyO3

### v1.1.0 (Planned)
- [ ] GPU acceleration (wgpu/cuda)
- [ ] Circulant neural network layers
- [ ] N-D quantum walks

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
