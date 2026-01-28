# API Tree

**Owner:** doc_expert | **Version:** 0.3.0 | **Updated:** 2026-01-28

> 1-page overview of all public API.

---

## Core Module (`crate::core`)

### `Circulant<T>`
1D circulant matrix with O(N log N) operations.

| Method | Description |
|--------|-------------|
| `::new(Vec<Complex<T>>)` | Create from complex generator |
| `::from_real(Vec<T>)` | Create from real generator |
| `.mul_vec(&[Complex<T>])` | Multiply by complex vector |
| `.mul_vec_real(&[T])` | Multiply by real vector |
| `.eigenvalues()` | Get eigenvalues via FFT |
| `.precompute()` | Cache spectrum for repeated operations |
| `.generator()` | Access the generator vector |
| `.size()` | Matrix dimension |

### `BlockCirculant<T>`
2D Block Circulant with Circulant Blocks (BCCB) matrix.

| Method | Description |
|--------|-------------|
| `::new(Array2<Complex<T>>)` | Create from 2D generator |
| `::from_kernel(Array2, rows, cols)` | Create from convolution kernel |
| `.mul_array(&Array2)` | 2D matrix-array product |
| `.rows()` / `.cols()` | Dimensions |

### Helper Functions

| Function | Description |
|----------|-------------|
| `naive_circulant_mul(&[C], &[C])` | Reference O(N²) multiplication |
| `naive_bccb_mul(&Array2, &Array2)` | Reference O(N⁴) multiplication |

---

## Physics Module (`crate::physics`, feature = "physics")

### `QuantumState<T>`
Quantum state representation for coined walks.

| Method | Description |
|--------|-------------|
| `::localized(pos, size, coin_dim)` | State localized at position with |0⟩ coin |
| `::localized_with_coin(pos, coin, size, coin_dim)` | Localized with specific coin state |
| `::superposition_at(pos, size, coin_dim)` | Equal superposition of coin states |
| `.get(pos, coin)` | Get amplitude |
| `.set(pos, coin, value)` | Set amplitude |
| `.normalize()` | Normalize to unit norm |
| `.norm_squared()` | Compute ||ψ||² |
| `.is_normalized(eps)` | Check if normalized |
| `.position_probabilities()` | Marginal position distribution |
| `.num_positions()` | Number of positions |
| `.coin_dim()` | Coin space dimension |

### `Coin`
Quantum coin operators.

| Variant | Description |
|---------|-------------|
| `Coin::Hadamard` | 2D Hadamard coin (1/√2)[[1,1],[1,-1]] |
| `Coin::Grover(dim)` | d-dimensional Grover diffusion coin |
| `Coin::Dft(dim)` | d-dimensional DFT coin |
| `Coin::Identity(dim)` | d-dimensional identity |

| Method | Description |
|--------|-------------|
| `.dimension()` | Coin space dimension |
| `.matrix::<T>()` | Get coin as Array2 |
| `.is_unitary::<T>(eps)` | Check unitarity |

### `CoinedWalk1D<T>`
1D coined quantum walk on a cycle.

| Method | Description |
|--------|-------------|
| `::new(size, coin)` | Create walk with size and coin |
| `.step(&mut state)` | Single evolution step |
| `.simulate(state, steps)` | Multi-step evolution |
| `.size()` | Number of positions |

### `CoinedWalk2D<T>`
2D coined quantum walk on a torus.

| Method | Description |
|--------|-------------|
| `::new(rows, cols, coin)` | Create 2D walk |
| `.step(&mut state)` | Single evolution step |
| `.simulate(state, steps)` | Multi-step evolution |

### `Hamiltonian<T>`
Hamiltonian-based time evolution.

| Method | Description |
|--------|-------------|
| `::from_circulant(Circulant)` | Create from circulant Hamiltonian |
| `.evolve(&state, time)` | Time evolution via exp(-iHt) |

### Traits

| Trait | Description |
|-------|-------------|
| `QuantumWalk` | Common interface for quantum walks |

---

## Vision Module (`crate::vision`, feature = "vision")

### `Kernel`
Predefined convolution kernels.

| Method | Description |
|--------|-------------|
| `::identity(size)` | Identity kernel |
| `::sharpen()` | 3x3 sharpening filter |
| `::sobel_x()` | Horizontal Sobel edge detector |
| `::sobel_y()` | Vertical Sobel edge detector |

### `BccbFilter<T>`
2D image filtering via BCCB.

| Method | Description |
|--------|-------------|
| `::new(kernel, rows, cols)` | Create filter from kernel |
| `.apply(&Array2)` | Apply filter to image |

---

## Visualize Module (`crate::visualize`, feature = "visualize")

### `Heatmap`
Probability distribution visualization.

| Method | Description |
|--------|-------------|
| `::new(data)` | Create heatmap from 2D data |
| `.render()` | Render to terminal |

### Quantum Visualization

| Function | Description |
|----------|-------------|
| `plot_probability_1d(&state)` | Plot 1D probability distribution |
| `plot_probability_2d(&state)` | Plot 2D probability heatmap |

---

## FFT Module (`crate::fft`)

### `FftBackend` Trait
Pluggable FFT implementation.

| Method | Description |
|--------|-------------|
| `.fft(&[Complex])` | Forward FFT |
| `.ifft(&[Complex])` | Inverse FFT |
| `.fft2d(&Array2)` | 2D forward FFT |
| `.ifft2d(&Array2)` | 2D inverse FFT |

### `RustFftBackend`
Default backend using rustfft crate.

---

## Traits Module (`crate::traits`)

### `CirculantOps<T>`
Core operations for circulant matrices.

| Method | Description |
|--------|-------------|
| `.mul_vec(&[Complex<T>])` | Matrix-vector product |
| `.eigenvalues()` | Compute eigenvalues |

### `BlockOps<T>`
Operations for block circulant matrices.

| Method | Description |
|--------|-------------|
| `.mul_array(&Array2)` | Matrix-array product |

### `Numeric`
Trait bounds for numeric types (f32, f64).

---

## Error Types (`crate::error`)

### `CirculantError`

| Variant | Description |
|---------|-------------|
| `DimensionMismatch { expected, got }` | Size mismatch in operation |
| `EmptyGenerator` | Empty generator vector |
| `InvalidSize(String)` | Invalid matrix/state size |
| `FftError(String)` | FFT operation failed |
| `NormalizationError` | Cannot normalize zero state |

---

## Prelude (`crate::prelude`)

Re-exports common items for convenience:

```rust
use circulant_rs::prelude::*;

// Includes:
// - Circulant, BlockCirculant
// - CirculantOps, BlockOps
// - CirculantError, Result
// - Complex (from num_complex)
// - Array1, Array2 (from ndarray)
// - Feature-gated: QuantumState, Coin, CoinedWalk1D, etc.
```

---

## Python Bindings (`crate::python`, feature = "python")

PyO3-based Python interface (when compiled as Python module).

| Class | Description |
|-------|-------------|
| `PyCirculant` | Python wrapper for Circulant |
| `PyBlockCirculant` | Python wrapper for BlockCirculant |
| `PyQuantumState` | Python wrapper for QuantumState |
| `PyQuantumWalk1D` | Python wrapper for CoinedWalk1D |
