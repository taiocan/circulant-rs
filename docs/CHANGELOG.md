# Changelog

**Version:** 0.3.0 | **Updated:** 2026-01-28 | **Reading time:** 2 min

> All notable changes to circulant-rs.

This format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
