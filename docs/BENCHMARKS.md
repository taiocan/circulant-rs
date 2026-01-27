# Benchmarks

**Version:** 0.2.0 | **Updated:** 2026-01-27 | **Reading time:** 15 min

> Performance validation methodology and results for circulant-rs.

---

## Executive Summary

This document proposes a comprehensive benchmark suite to validate the performance claims of `circulant-rs` against competing solutions. The benchmarks will measure:

1. **Throughput**: Operations per second for core operations
2. **Latency**: Time to complete single operations
3. **Memory**: Peak and sustained memory usage
4. **Scaling**: Performance across problem sizes (N = 64 to N = 1M)
5. **Accuracy**: Numerical precision of results

Results will provide objective data for the marketing claims and identify areas for optimization.

---

## Competing Solutions

### Tier 1: Direct Competitors (Same Problem Domain)

| Solution | Language | Approach | Notes |
|----------|----------|----------|-------|
| **SciPy `scipy.linalg.circulant`** | Python | Dense matrix construction | Standard scientific Python |
| **SciPy `scipy.signal.fftconvolve`** | Python | FFT-based convolution | Different API, same math |
| **NumPy manual FFT** | Python | `np.fft.ifft(np.fft.fft(c) * np.fft.fft(x))` | Fair FFT comparison |
| **Julia `DSP.jl` / `FFTW.jl`** | Julia | FFT-based | Known for performance |

### Tier 2: Rust Ecosystem Alternatives

| Solution | Crate | Approach | Notes |
|----------|-------|----------|-------|
| **ndarray + manual FFT** | `ndarray` + `rustfft` | Manual FFT implementation | What users would build themselves |
| **nalgebra dense** | `nalgebra` | Dense O(N²) multiply | Baseline for speedup claims |
| **fftw-rs** | `fftw` | FFTW bindings | C library, potentially faster FFT |

### Tier 3: Quantum Walk Specific

| Solution | Language | Notes |
|----------|----------|-------|
| **QuTiP** | Python | General quantum simulation, dense matrices |
| **Qiskit Aer** | Python | Quantum circuit simulation |
| **Custom NumPy** | Python | Hand-optimized quantum walk |

---

## Benchmark Scenarios

### Scenario 1: 1D Circulant Matrix-Vector Multiplication

**Objective**: Measure core operation performance across sizes.

**Operation**: `y = C · x` where C is circulant

**Parameters**:
| Parameter | Values |
|-----------|--------|
| Size N | 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576 |
| Data type | f64 (Complex<f64>) |
| Generator | Random complex values |
| Input vector | Random complex values |
| Iterations | 1000 (warm), 10000 (measured) |

**Metrics**:
- Time per operation (μs)
- Throughput (operations/second)
- Memory usage (bytes)

**Competitors**:
- circulant-rs `Circulant::mul_vec`
- circulant-rs with `precompute()`
- NumPy manual FFT
- SciPy dense circulant multiply
- ndarray + rustfft (manual implementation)
- nalgebra dense multiply (for N ≤ 16384)

**Expected Outcome**:
```
N=1024:    circulant-rs ~10x faster than NumPy FFT, ~100x vs dense
N=65536:   circulant-rs ~5x faster than NumPy FFT, ~1000x vs dense
N=1048576: Dense infeasible, circulant-rs vs NumPy FFT comparison only
```

---

### Scenario 2: 2D Block Circulant (BCCB) Convolution

**Objective**: Measure 2D convolution performance for image-sized data.

**Operation**: `Y = G ⊛ X` (2D circular convolution)

**Parameters**:
| Parameter | Values |
|-----------|--------|
| Image size | 64×64, 256×256, 512×512, 1024×1024, 2048×2048, 4096×4096 |
| Kernel size | 3×3, 7×7, 15×15, 31×31 |
| Data type | f64 |
| Iterations | 100 (warm), 1000 (measured) |

**Metrics**:
- Time per convolution (ms)
- Throughput (megapixels/second)
- Memory usage

**Competitors**:
- circulant-rs `BlockCirculant::mul_array`
- SciPy `scipy.signal.fftconvolve` (mode='wrap')
- OpenCV `filter2D` with border wrap
- NumPy manual 2D FFT

**Expected Outcome**:
```
512×512:  circulant-rs competitive with scipy.signal.fftconvolve
4096×4096: circulant-rs ~2-5x faster due to Rust overhead elimination
```

---

### Scenario 3: Quantum Walk Simulation

**Objective**: Measure end-to-end quantum walk performance.

**Operation**: Simulate T steps of coined quantum walk on N-node ring

**Parameters**:
| Parameter | Values |
|-----------|--------|
| Positions N | 256, 1024, 4096, 16384, 65536, 262144 |
| Steps T | 10, 100, 500, 1000 |
| Coin | Hadamard |
| Initial state | Localized at N/2 |

**Metrics**:
- Total simulation time (ms)
- Time per step (μs)
- Final state norm error (should be < 10⁻¹⁰)
- Memory usage

**Competitors**:
- circulant-rs `CoinedWalk1D::simulate`
- QuTiP sparse matrix evolution
- Custom NumPy implementation (optimized)
- Julia implementation

**Expected Outcome**:
```
N=1024, T=100:  circulant-rs ~10x faster than QuTiP
N=65536, T=100: QuTiP runs out of memory, circulant-rs completes in seconds
```

---

### Scenario 4: Memory Scaling

**Objective**: Validate O(N) memory claims.

**Operation**: Create circulant structure and measure memory

**Parameters**:
| Parameter | Values |
|-----------|--------|
| Size N | 1K, 10K, 100K, 1M, 10M |

**Metrics**:
- Peak memory allocation (bytes)
- Resident set size (RSS)
- Theoretical vs actual

**Competitors**:
- circulant-rs `Circulant::new`
- SciPy `scipy.linalg.circulant` (creates dense)
- nalgebra dense matrix

**Expected Outcome**:
```
N=1M: circulant-rs ~16 MB, dense ~8 TB (infeasible)
```

---

### Scenario 5: Numerical Accuracy

**Objective**: Verify correctness and measure numerical error.

**Operation**: Compare FFT-based result with naive O(N²) computation

**Parameters**:
| Parameter | Values |
|-----------|--------|
| Size N | 64, 256, 1024, 4096 |
| Test cases | 100 random instances each |

**Metrics**:
- Maximum absolute error
- Mean absolute error
- Relative error (L2 norm)
- Condition number sensitivity

**Competitors**:
- circulant-rs vs naive implementation
- NumPy FFT vs naive
- Julia FFT vs naive

**Expected Outcome**:
```
All implementations: max relative error < 10⁻¹⁰ for well-conditioned inputs
```

---

### Scenario 6: Precomputation Benefit

**Objective**: Quantify speedup from caching eigenvalues.

**Operation**: Multiple multiplications with same circulant matrix

**Parameters**:
| Parameter | Values |
|-----------|--------|
| Size N | 1024, 16384, 262144 |
| Multiplications | 1, 10, 100, 1000 |

**Metrics**:
- Time with precompute() vs without
- Break-even point (when precompute pays off)

**Expected Outcome**:
```
precompute() pays off after 2-3 multiplications
```

---

## Methodology

### Hardware Configuration

**Primary Test System** (document actual specs):
```
CPU: [e.g., AMD Ryzen 9 5900X, 12 cores, 3.7 GHz base]
RAM: [e.g., 64 GB DDR4-3600]
OS:  [e.g., Ubuntu 22.04 LTS]
```

**Secondary Test System** (for cross-validation):
```
CPU: [e.g., Apple M2, 8 cores]
RAM: [e.g., 16 GB unified]
OS:  [e.g., macOS 14]
```

### Software Versions

| Software | Version | Notes |
|----------|---------|-------|
| Rust | 1.75+ | Release build, LTO enabled |
| Python | 3.11+ | With NumPy 1.26+, SciPy 1.12+ |
| Julia | 1.10+ | With FFTW.jl |
| circulant-rs | 0.2.0 | All features enabled |

### Build Configuration

**Rust (circulant-rs)**:
```toml
[profile.bench]
opt-level = 3
lto = "fat"
codegen-units = 1
```

**Python**:
```bash
# Ensure NumPy uses optimized BLAS
python -c "import numpy; numpy.show_config()"
```

### Measurement Protocol

1. **Warm-up**: Run operation 100-1000 times before measuring
2. **Timing**: Use high-resolution timers (Criterion for Rust, `timeit` for Python)
3. **Statistical rigor**: Report median, mean, std dev, min, max
4. **Isolation**: Disable CPU frequency scaling, close other applications
5. **Repetition**: Run full benchmark suite 3 times, report best median

### Fairness Principles

1. **Apples to apples**: Compare FFT-based methods separately from dense methods
2. **Optimize all competitors**: Use best-known practices for each
3. **Include setup time**: For precomputation benchmarks, include cache building
4. **Realistic data**: Use random data, not adversarial cases
5. **Document limitations**: Note where comparisons are imperfect

---

## Implementation Plan

### Phase 1: Infrastructure (Week 1)

- [ ] Create `benches/comparative/` directory structure
- [ ] Set up Criterion benchmark harness for Rust
- [ ] Create Python benchmark scripts with proper timing
- [ ] Create Julia benchmark scripts
- [ ] Implement common data generation utilities
- [ ] Set up results collection (JSON/CSV output)

### Phase 2: Core Benchmarks (Week 2)

- [ ] Implement Scenario 1 (1D multiply) for all competitors
- [ ] Implement Scenario 2 (2D BCCB) for all competitors
- [ ] Implement Scenario 4 (memory scaling)
- [ ] Implement Scenario 5 (numerical accuracy)

### Phase 3: Application Benchmarks (Week 3)

- [ ] Implement Scenario 3 (quantum walk) for all competitors
- [ ] Implement Scenario 6 (precomputation)
- [ ] Create QuTiP comparison script
- [ ] Create custom NumPy quantum walk baseline

### Phase 4: Analysis & Reporting (Week 4)

- [ ] Run full benchmark suite on primary system
- [ ] Run validation on secondary system
- [ ] Generate plots and tables
- [ ] Write analysis document
- [ ] Update marketing materials with actual numbers

---

## Expected Results Summary

Based on algorithmic complexity and preliminary testing:

### Performance Predictions

| Scenario | vs SciPy Dense | vs NumPy FFT | vs QuTiP |
|----------|---------------|--------------|----------|
| 1D multiply (N=64K) | ~1000× faster | ~3-10× faster | N/A |
| 2D convolve (1K×1K) | ~100× faster | ~2-5× faster | N/A |
| Quantum walk (N=64K) | N/A | ~5-10× faster | ~50× faster |

### Memory Predictions

| Scenario | circulant-rs | Dense Alternative | Ratio |
|----------|--------------|-------------------|-------|
| N = 10,000 | ~160 KB | ~800 MB | 5000× |
| N = 100,000 | ~1.6 MB | ~80 GB | 50000× |
| N = 1,000,000 | ~16 MB | ~8 TB | 500000× |

---

## Reporting Format

### Benchmark Results Document

```markdown
# circulant-rs Benchmark Results

## Test Configuration
- Date: YYYY-MM-DD
- Hardware: [specs]
- Software: [versions]

## Summary Table
| Benchmark | circulant-rs | Best Competitor | Speedup |
|-----------|--------------|-----------------|---------|
| ...       | ...          | ...             | ...     |

## Detailed Results

### Scenario 1: 1D Multiplication
[Charts, tables, analysis]

### Scenario 2: 2D BCCB
[Charts, tables, analysis]

[etc.]

## Conclusions
[Key findings, caveats, recommendations]
```

### Visualization Requirements

1. **Log-log scaling plots**: Time vs N for each scenario
2. **Memory scaling plots**: Bytes vs N
3. **Speedup bar charts**: circulant-rs vs each competitor
4. **Accuracy scatter plots**: Error vs N

---

## Benchmark Code Skeletons

### Rust (Criterion)

```rust
// benches/comparative/multiply_1d.rs
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use circulant_rs::core::Circulant;
use num_complex::Complex;

fn bench_circulant_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("1d_multiply");

    for size in [64, 256, 1024, 4096, 16384, 65536] {
        let generator: Vec<Complex<f64>> = (0..size)
            .map(|i| Complex::new(i as f64, 0.0))
            .collect();
        let x: Vec<Complex<f64>> = (0..size)
            .map(|i| Complex::new(1.0 / (i + 1) as f64, 0.0))
            .collect();

        let circulant = Circulant::new(generator).unwrap();

        group.bench_with_input(
            BenchmarkId::new("circulant-rs", size),
            &size,
            |b, _| b.iter(|| circulant.mul_vec(&x)),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_circulant_multiply);
criterion_main!(benches);
```

### Python (NumPy/SciPy)

```python
# benchmarks/python/multiply_1d.py
import numpy as np
from scipy.linalg import circulant
import timeit
import json

def bench_scipy_dense(generator, x, iterations=1000):
    C = circulant(generator)
    # Warm-up
    for _ in range(100):
        _ = C @ x
    # Measure
    start = timeit.default_timer()
    for _ in range(iterations):
        _ = C @ x
    elapsed = timeit.default_timer() - start
    return elapsed / iterations

def bench_numpy_fft(generator, x, iterations=1000):
    g_fft = np.fft.fft(generator)
    # Warm-up
    for _ in range(100):
        _ = np.fft.ifft(g_fft * np.fft.fft(x))
    # Measure
    start = timeit.default_timer()
    for _ in range(iterations):
        _ = np.fft.ifft(g_fft * np.fft.fft(x))
    elapsed = timeit.default_timer() - start
    return elapsed / iterations

if __name__ == "__main__":
    results = []
    for size in [64, 256, 1024, 4096, 16384, 65536]:
        generator = np.random.randn(size) + 1j * np.random.randn(size)
        x = np.random.randn(size) + 1j * np.random.randn(size)

        if size <= 16384:  # Dense becomes infeasible
            t_dense = bench_scipy_dense(generator, x)
        else:
            t_dense = None

        t_fft = bench_numpy_fft(generator, x)

        results.append({
            "size": size,
            "scipy_dense_us": t_dense * 1e6 if t_dense else None,
            "numpy_fft_us": t_fft * 1e6,
        })

    print(json.dumps(results, indent=2))
```

### Quantum Walk Comparison

```python
# benchmarks/python/quantum_walk.py
import numpy as np
import timeit

def numpy_quantum_walk(n_positions, n_steps, coin_matrix):
    """Hand-optimized NumPy quantum walk."""
    # State: [pos0_coin0, pos0_coin1, pos1_coin0, pos1_coin1, ...]
    state = np.zeros(n_positions * 2, dtype=np.complex128)
    state[n_positions] = 1.0  # Localized at center, coin 0

    # Precompute shift FFTs
    left_shift = np.zeros(n_positions, dtype=np.complex128)
    left_shift[-1] = 1.0
    left_fft = np.fft.fft(left_shift)

    right_shift = np.zeros(n_positions, dtype=np.complex128)
    right_shift[1] = 1.0
    right_fft = np.fft.fft(right_shift)

    for _ in range(n_steps):
        # Reshape for coin application
        state_2d = state.reshape(n_positions, 2)

        # Apply coin
        new_state = state_2d @ coin_matrix.T

        # Extract coin components
        coin0 = new_state[:, 0]
        coin1 = new_state[:, 1]

        # Shift via FFT
        shifted0 = np.fft.ifft(left_fft * np.fft.fft(coin0))
        shifted1 = np.fft.ifft(right_fft * np.fft.fft(coin1))

        # Recombine
        state[0::2] = shifted0
        state[1::2] = shifted1

    return state

# Hadamard coin
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

for n in [256, 1024, 4096, 16384]:
    t = timeit.timeit(
        lambda: numpy_quantum_walk(n, 100, H),
        number=10
    ) / 10
    print(f"N={n}: {t*1000:.2f} ms")
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| NumPy uses MKL/OpenBLAS (unfair comparison) | Document BLAS backend, also test with vanilla NumPy |
| Different FFT implementations (RustFFT vs FFTW) | Include FFTW Rust bindings as separate data point |
| Python overhead varies by setup | Use `timeit`, run in isolated environment |
| Results not reproducible | Publish all benchmark code, document exact versions |
| Cherry-picking favorable results | Report all results, including where we lose |

---

## Success Criteria

The benchmarks will be considered successful if:

1. **Performance claims validated**: circulant-rs shows ≥10× speedup over dense methods for N ≥ 1024
2. **Competitive with FFT baselines**: Within 5× of optimized NumPy FFT (Rust overhead acceptable)
3. **Memory claims validated**: O(N) scaling demonstrated empirically
4. **Numerical accuracy confirmed**: Relative error < 10⁻⁹ for all test cases
5. **Reproducible**: Independent parties can replicate results within 20%

---

## Deliverables

1. **Benchmark code repository**: All scripts and harnesses
2. **Raw results**: JSON/CSV files with all measurements
3. **Analysis document**: Interpretation and conclusions
4. **Updated marketing materials**: With validated claims
5. **Blog post draft**: Narrative version of findings

---

## Timeline

| Week | Milestone |
|------|-----------|
| 1 | Infrastructure complete, basic benchmarks running |
| 2 | Core benchmarks (Scenarios 1, 2, 4, 5) complete |
| 3 | Application benchmarks (Scenarios 3, 6) complete |
| 4 | Analysis, visualization, documentation |

---

## Appendix: Reference Implementations

### Naive O(N²) Circulant Multiply (Ground Truth)

```rust
fn naive_circulant_mul(generator: &[Complex<f64>], x: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let n = generator.len();
    let mut result = vec![Complex::new(0.0, 0.0); n];

    for i in 0..n {
        for j in 0..n {
            let idx = (j + n - i) % n;
            result[i] += generator[idx] * x[j];
        }
    }

    result
}
```

### QuTiP Quantum Walk (Baseline)

```python
from qutip import *
import numpy as np

def qutip_quantum_walk(n_positions, n_steps):
    # Coin space
    coin_0 = basis(2, 0)
    coin_1 = basis(2, 1)

    # Position space
    positions = [basis(n_positions, i) for i in range(n_positions)]

    # Hadamard coin
    H = (1/np.sqrt(2)) * Qobj([[1, 1], [1, -1]])

    # Shift operators
    S_plus = sum(
        tensor(positions[(i+1) % n_positions], positions[i].dag(), coin_1, coin_1.dag())
        for i in range(n_positions)
    )
    S_minus = sum(
        tensor(positions[(i-1) % n_positions], positions[i].dag(), coin_0, coin_0.dag())
        for i in range(n_positions)
    )
    S = S_plus + S_minus

    # Coin operator (identity on position)
    C = tensor(qeye(n_positions), H)

    # Walk operator
    U = S * C

    # Initial state: center, coin 0
    psi0 = tensor(positions[n_positions // 2], coin_0)

    # Evolve
    psi = psi0
    for _ in range(n_steps):
        psi = U * psi

    return psi
```

---

*This benchmark proposal provides the foundation for objective, reproducible performance validation of circulant-rs.*
