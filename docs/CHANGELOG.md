# Changelog

**Version:** 1.0.0 | **Updated:** 2026-01-28 | **Reading time:** 3 min

> All notable changes to circulant-rs.

This format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-01-28

### Added

- **N-Dimensional Circulant Tensors** (DE-2026-002)
  - `CirculantTensor<T, D>` - Generic N-D circulant tensor with compile-time dimension checking
  - `TensorOps<T, D>` trait for unified tensor operations
  - Separable N-D FFT via sequential 1D transforms along each axis
  - O(N log N) multiplication where N = total elements
  - Type aliases: `Circulant1D<T>`, `Circulant2D<T>`, `Circulant3D<T>`, `Circulant4D<T>`

- **Parallel N-D FFT** (`parallel` feature)
  - Parallel axis FFT processing using rayon
  - `mul_tensor_parallel()` for large tensors
  - `mul_tensor_auto()` with automatic parallelism threshold

- **New Error Types**
  - `InvalidTensorShape` - Shape mismatch for tensor operations
  - `InvalidTensorDimension` - Dimension count mismatch
  - `ReshapeFailed` - Reshape operation failures

### Changed

- **API Maturity** - Version 1.0.0 marks stable API commitment

### Deprecated

- `Circulant<T>` - Use `CirculantTensor<T, 1>` or `Circulant1D<T>` instead
- `BlockCirculant<T>` - Use `CirculantTensor<T, 2>` or `Circulant2D<T>` instead

### Migration Guide

```rust
// Old API (deprecated)
use circulant_rs::Circulant;
let c = Circulant::from_real(vec![1.0, 2.0, 3.0]).unwrap();

// New API
use circulant_rs::{CirculantTensor, TensorOps};
use ndarray::ArrayD;
let gen = ArrayD::from_shape_vec(vec![3], vec![...]).unwrap();
let tensor = CirculantTensor::<f64, 1>::new(gen).unwrap();
```

---

## [0.3.0] - 2026-01-28

### Added

- **Vision Module**
  - `Kernel::sharpen()` - 3x3 sharpening filter for edge enhancement (DE-2026-001)

---

## [0.2.0] - 2026-01-27

### Added

- **Core Module**
  - 1D Circulant matrix with FFT-based O(N log N) multiplication
  - 2D Block Circulant (BCCB) matrix operations
  - Eigenvalue precomputation for repeated operations

- **Physics Module** (`physics` feature)
  - Quantum state management with normalization
  - Coin operators: Hadamard, Grover, DFT, custom
  - 1D coined quantum walk simulation
  - 2D coined quantum walk simulation
  - Hamiltonian-based continuous-time evolution

- **Vision Module** (`vision` feature)
  - Convolution kernels (Gaussian, Sobel, Laplacian)
  - BCCB-based image filtering

- **Visualization Module** (`visualize` feature)
  - Probability distribution plots
  - Evolution heatmaps

- **Integrations**
  - Serde serialization (`serde` feature)
  - Rayon parallelization (`parallel` feature)
  - Python bindings via PyO3 (`python` feature)

### Performance

- 676× speedup vs naive O(N²) at N=2048
- O(N) memory vs O(N²) for dense matrices
- Scales to N=1,000,000+ positions

---

## Related Documents

- [README.md](../README.md) - Project overview
- [BENCHMARKS.md](./BENCHMARKS.md) - Performance methodology
