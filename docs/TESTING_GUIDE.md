# Testing & Benchmarking Guide for circulant-rs

A hands-on guide for manually testing, validating, and benchmarking the circulant-rs library.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Validation (5 minutes)](#quick-validation-5-minutes)
3. [Running the Test Suite](#running-the-test-suite)
4. [Running Benchmarks](#running-benchmarks)
5. [Manual Testing Examples](#manual-testing-examples)
6. [Python Comparison Tests](#python-comparison-tests)
7. [Performance Profiling](#performance-profiling)
8. [Interpreting Results](#interpreting-results)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

| Software | Version | Check Command | Installation |
|----------|---------|---------------|--------------|
| Rust | 1.70+ | `rustc --version` | [rustup.rs](https://rustup.rs) |
| Cargo | 1.70+ | `cargo --version` | Included with Rust |
| Python | 3.9+ | `python3 --version` | For comparison tests |
| NumPy | 1.20+ | `python3 -c "import numpy; print(numpy.__version__)"` | `pip install numpy` |
| SciPy | 1.7+ | `python3 -c "import scipy; print(scipy.__version__)"` | `pip install scipy` |

### Clone and Setup

```bash
# Clone the repository (if not already done)
git clone https://github.com/your-org/circulant-rs.git
cd circulant-rs

# Verify the project structure
ls -la src/

# Expected output:
# src/
# ├── core/
# ├── fft/
# ├── physics/
# ├── traits/
# ├── error.rs
# ├── lib.rs
# └── prelude.rs
```

---

## Quick Validation (5 minutes)

### Step 1: Build the Library

```bash
# Debug build (faster compilation, slower execution)
cargo build --features "physics parallel serde"

# Release build (slower compilation, optimized execution)
cargo build --release --features "physics parallel serde"
```

**Expected output**: `Finished` message with no errors.

### Step 2: Run All Tests

```bash
cargo test --features "physics parallel serde"
```

**Expected output**:
```
test result: ok. 76 passed; 0 failed; 0 ignored
...
test result: ok. 96 passed (total across all test files)
```

### Step 3: Run the Example

```bash
cargo run --example quantum_walk_1d --features "physics parallel serde"
```

**Expected output**: A probability distribution showing quantum walk spreading with:
- Total probability ≈ 1.0
- Asymmetric distribution (more probability on left side with Hadamard coin)
- Ballistic spreading pattern

### Step 4: Quick Benchmark

```bash
cargo bench --bench fft_multiply --features "physics parallel serde" -- --sample-size 10
```

**Expected output**: Timing results showing microsecond-scale operations.

---

## Running the Test Suite

### Run All Tests

```bash
# Standard test run
cargo test --features "physics parallel serde"

# With output (see println! statements)
cargo test --features "physics parallel serde" -- --nocapture

# Run tests in parallel (default) or single-threaded
cargo test --features "physics parallel serde" -- --test-threads=1
```

### Run Specific Test Categories

```bash
# Unit tests only (in src/)
cargo test --lib --features "physics parallel serde"

# Integration tests only (in tests/)
cargo test --test core_tests --features "physics parallel serde"
cargo test --test physics_tests --features "physics parallel serde"
cargo test --test property_tests --features "physics parallel serde"

# Doc tests only
cargo test --doc --features "physics parallel serde"
```

### Run Specific Tests

```bash
# Run tests matching a pattern
cargo test fft_multiply --features "physics parallel serde"
cargo test quantum_walk --features "physics parallel serde"
cargo test circulant --features "physics parallel serde"

# Run a single specific test
cargo test test_single_step_preserves_norm --features "physics parallel serde" -- --exact
```

### Test with Different Features

```bash
# Minimal (no optional features)
cargo test

# Without parallelization
cargo test --features "physics serde"

# Without serialization
cargo test --features "physics parallel"

# Physics only
cargo test --features "physics"
```

---

## Running Benchmarks

### Available Benchmarks

| Benchmark | File | Description |
|-----------|------|-------------|
| `fft_multiply` | `benches/fft_multiply.rs` | Basic FFT multiplication |
| `scaling_benchmark` | `benches/comparative/scaling_benchmark.rs` | Comprehensive scaling tests |

### Run All Benchmarks

```bash
# Run all benchmarks (takes several minutes)
cargo bench --features "physics parallel serde"
```

### Run Specific Benchmarks

```bash
# FFT multiply benchmark
cargo bench --bench fft_multiply --features "physics parallel serde"

# Scaling benchmark
cargo bench --bench scaling_benchmark --features "physics parallel serde"

# Filter by benchmark name
cargo bench --features "physics parallel serde" -- "1d_circulant"
cargo bench --features "physics parallel serde" -- "quantum_walk"
cargo bench --features "physics parallel serde" -- "2d_bccb"
```

### Benchmark Options

```bash
# Reduce sample size for quick results
cargo bench --features "physics parallel serde" -- --sample-size 10

# Increase measurement time for more accuracy
cargo bench --features "physics parallel serde" -- --measurement-time 10

# Save baseline for comparison
cargo bench --features "physics parallel serde" -- --save-baseline my_baseline

# Compare against baseline
cargo bench --features "physics parallel serde" -- --baseline my_baseline
```

### View Benchmark Reports

After running benchmarks, HTML reports are generated:

```bash
# Open the report in browser
open target/criterion/report/index.html      # macOS
xdg-open target/criterion/report/index.html  # Linux
start target/criterion/report/index.html     # Windows
```

---

## Manual Testing Examples

### Test 1: Verify Circulant Matrix Multiplication

Create a file `manual_test.rs` or run in a Rust playground:

```rust
use circulant_rs::prelude::*;
use num_complex::Complex;

fn main() {
    // Create a simple circulant matrix
    // Generator [1, 2, 3, 4] creates:
    // | 1 2 3 4 |
    // | 4 1 2 3 |
    // | 3 4 1 2 |
    // | 2 3 4 1 |

    let generator = vec![1.0, 2.0, 3.0, 4.0];
    let circulant = Circulant::from_real(generator).unwrap();

    // Multiply by [1, 0, 0, 0]
    let x = vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
    ];

    let result = circulant.mul_vec(&x).unwrap();

    // Result should be first column: [1, 4, 3, 2]
    println!("Result: {:?}", result);
    println!("Expected: [1, 4, 3, 2]");

    // Verify
    assert!((result[0].re - 1.0).abs() < 1e-10);
    assert!((result[1].re - 4.0).abs() < 1e-10);
    assert!((result[2].re - 3.0).abs() < 1e-10);
    assert!((result[3].re - 2.0).abs() < 1e-10);

    println!("✓ Circulant multiplication verified!");
}
```

Run it:
```bash
# Add to examples/manual_test.rs, then:
cargo run --example manual_test --features "physics parallel serde"
```

### Test 2: Verify Quantum Walk Norm Preservation

```rust
use circulant_rs::prelude::*;

fn main() -> Result<()> {
    let walk = CoinedWalk1D::<f64>::new(256, Coin::Hadamard)?;
    let mut state = QuantumState::localized(128, 256, 2)?;

    println!("Step | Norm² | Deviation from 1.0");
    println!("-----|-------|-------------------");

    for step in 0..=100 {
        if step % 10 == 0 {
            let norm_sq = state.norm_squared();
            let deviation = (norm_sq - 1.0).abs();
            println!("{:4} | {:.10} | {:.2e}", step, norm_sq, deviation);

            // Verify norm is preserved
            assert!(deviation < 1e-10, "Norm not preserved at step {}", step);
        }
        walk.step(&mut state);
    }

    println!("\n✓ Norm preserved to machine precision over 100 steps!");
    Ok(())
}
```

### Test 3: Verify FFT vs Naive Results Match

```rust
use circulant_rs::core::{Circulant, naive_circulant_mul};
use circulant_rs::traits::CirculantOps;
use num_complex::Complex;

fn main() {
    let sizes = [8, 16, 32, 64, 128];

    for &n in &sizes {
        // Random-ish generator
        let generator: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();

        // Random-ish input
        let x: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new(1.0 / (i + 1) as f64, 0.1 * i as f64))
            .collect();

        // FFT method
        let circulant = Circulant::new(generator.clone()).unwrap();
        let fft_result = circulant.mul_vec(&x).unwrap();

        // Naive method
        let naive_result = naive_circulant_mul(&generator, &x);

        // Compare
        let max_error: f64 = fft_result.iter()
            .zip(naive_result.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);

        println!("N={:4}: max_error = {:.2e}", n, max_error);
        assert!(max_error < 1e-10, "FFT and naive don't match for N={}", n);
    }

    println!("\n✓ FFT results match naive implementation!");
}
```

### Test 4: Measure Speedup Manually

```rust
use circulant_rs::core::{Circulant, naive_circulant_mul};
use circulant_rs::traits::CirculantOps;
use num_complex::Complex;
use std::time::Instant;

fn main() {
    let sizes = [64, 256, 1024, 2048];

    println!("Size  | FFT Time   | Naive Time  | Speedup");
    println!("------|------------|-------------|--------");

    for &n in &sizes {
        let generator: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let x: Vec<Complex<f64>> = (0..n)
            .map(|i| Complex::new(1.0 / (i + 1) as f64, 0.0))
            .collect();

        let circulant = Circulant::new(generator.clone()).unwrap();

        // Warm up
        for _ in 0..10 {
            let _ = circulant.mul_vec(&x);
        }

        // Time FFT
        let iterations = 1000;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = circulant.mul_vec(&x);
        }
        let fft_time = start.elapsed().as_secs_f64() / iterations as f64;

        // Time naive (fewer iterations for large N)
        let naive_iterations = if n > 512 { 10 } else { 100 };
        let start = Instant::now();
        for _ in 0..naive_iterations {
            let _ = naive_circulant_mul(&generator, &x);
        }
        let naive_time = start.elapsed().as_secs_f64() / naive_iterations as f64;

        let speedup = naive_time / fft_time;

        println!("{:5} | {:>10.2} µs | {:>10.2} µs | {:>6.1}×",
            n, fft_time * 1e6, naive_time * 1e6, speedup);
    }
}
```

---

## Python Comparison Tests

### Setup Python Environment

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install numpy scipy
```

### Run Python Benchmarks

```bash
cd benchmarks/python
python compare_circulant.py
```

**Expected output**:
```
======================================================================
1D Circulant Matrix-Vector Multiplication Benchmark
======================================================================
      Size       NumPy FFT     SciPy Dense           Naive  FFT Speedup
----------------------------------------------------------------------
        64         X.XX µs        XX.XX µs         XXX.XX µs       XX.Xx
       256         X.XX µs       XXX.XX µs             N/A        XX.Xx
      1024        XX.XX µs      XXXX.XX µs             N/A       XXX.Xx
...
```

### Manual Python Comparison

```python
#!/usr/bin/env python3
"""Compare Python FFT with expected Rust results."""

import numpy as np
from scipy.linalg import circulant
import time

def test_correctness():
    """Verify Python FFT matches dense multiplication."""
    print("=== Correctness Test ===")

    for n in [8, 16, 32, 64]:
        # Generator
        c = np.random.randn(n) + 1j * np.random.randn(n)
        x = np.random.randn(n) + 1j * np.random.randn(n)

        # Dense method
        C = circulant(c)
        dense_result = C @ x

        # FFT method (cross-correlation semantics)
        spectrum = np.conj(np.fft.fft(np.conj(c)))
        fft_result = np.fft.ifft(spectrum * np.fft.fft(x))

        error = np.max(np.abs(dense_result - fft_result))
        print(f"N={n:3d}: max_error = {error:.2e}")
        assert error < 1e-10, f"Mismatch at N={n}"

    print("✓ Python FFT matches dense multiplication\n")

def test_performance():
    """Benchmark Python implementations."""
    print("=== Performance Test ===")
    print(f"{'Size':>6} | {'FFT':>12} | {'Dense':>12} | {'Speedup':>8}")
    print("-" * 50)

    for n in [64, 256, 1024, 2048]:
        c = np.random.randn(n) + 1j * np.random.randn(n)
        x = np.random.randn(n) + 1j * np.random.randn(n)

        # Time FFT
        spectrum = np.conj(np.fft.fft(np.conj(c)))
        iterations = 10000
        start = time.perf_counter()
        for _ in range(iterations):
            _ = np.fft.ifft(spectrum * np.fft.fft(x))
        fft_time = (time.perf_counter() - start) / iterations

        # Time dense (skip for large N)
        if n <= 2048:
            C = circulant(c)
            dense_iters = 100 if n <= 512 else 10
            start = time.perf_counter()
            for _ in range(dense_iters):
                _ = C @ x
            dense_time = (time.perf_counter() - start) / dense_iters
            speedup = dense_time / fft_time
            print(f"{n:>6} | {fft_time*1e6:>10.2f} µs | {dense_time*1e6:>10.2f} µs | {speedup:>7.1f}×")
        else:
            print(f"{n:>6} | {fft_time*1e6:>10.2f} µs | {'N/A':>12} | {'N/A':>8}")

if __name__ == "__main__":
    test_correctness()
    test_performance()
```

Save as `manual_python_test.py` and run:
```bash
python manual_python_test.py
```

---

## Performance Profiling

### Using Cargo Flamegraph

```bash
# Install flamegraph
cargo install flamegraph

# Run with profiling (Linux, requires perf)
cargo flamegraph --bench scaling_benchmark --features "physics parallel serde" -- --bench

# View the SVG
open flamegraph.svg  # or xdg-open on Linux
```

### Using perf (Linux)

```bash
# Build release binary
cargo build --release --example quantum_walk_1d --features "physics parallel serde"

# Profile with perf
perf record ./target/release/examples/quantum_walk_1d
perf report
```

### Memory Profiling with Valgrind

```bash
# Install valgrind (Linux)
sudo apt install valgrind

# Run with memory profiling
valgrind --tool=massif ./target/release/examples/quantum_walk_1d
ms_print massif.out.*
```

---

## Interpreting Results

### Test Results

| Result | Meaning | Action |
|--------|---------|--------|
| `ok` | Test passed | None needed |
| `FAILED` | Test failed | Check assertion message |
| `ignored` | Test skipped | Usually intentional (e.g., `#[ignore]`) |

### Benchmark Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| `time` | Median execution time | Lower is better |
| `thrpt` | Throughput (ops/sec) | Higher is better |
| `change` | vs baseline | Negative = improvement |

### Expected Performance Ranges

| Operation | Size | Expected Time | Acceptable Range |
|-----------|------|---------------|------------------|
| 1D multiply | N=1024 | ~10-15 µs | 5-50 µs |
| 1D multiply | N=65536 | ~500-1000 µs | 200-2000 µs |
| QW step | N=1024 | ~20-30 µs | 10-100 µs |
| QW 100 steps | N=65536 | ~500 ms | 200-1000 ms |

### Speedup Expectations

| Size | vs Naive | Notes |
|------|----------|-------|
| N=64 | 20-50× | FFT overhead visible |
| N=256 | 80-150× | Sweet spot begins |
| N=1024 | 300-500× | Clear advantage |
| N=4096 | 1000-2000× | Significant speedup |

---

## Troubleshooting

### Common Issues

#### Build Fails

```bash
# Clean and rebuild
cargo clean
cargo build --features "physics parallel serde"

# Check Rust version
rustc --version
# Minimum: 1.70.0
```

#### Tests Fail

```bash
# Run specific failing test with output
cargo test test_name --features "physics parallel serde" -- --nocapture

# Check for floating-point precision issues
# Tests use epsilon = 1e-10, your system may need adjustment
```

#### Benchmarks Show High Variance

```bash
# Increase sample size
cargo bench -- --sample-size 100

# Disable CPU frequency scaling (Linux)
sudo cpupower frequency-set --governor performance

# Close other applications
# Run on idle system
```

#### Python Comparison Differs

```bash
# Check NumPy BLAS backend
python -c "import numpy; numpy.show_config()"

# Different BLAS = different performance characteristics
# Results should still be within 2-3× of each other
```

### Getting Help

1. **Check existing tests**: Look at `tests/` and `src/**/tests.rs` for examples
2. **Read the docs**: `cargo doc --open --features "physics parallel serde"`
3. **Open an issue**: Include test output, Rust version, and OS

---

## Checklist for Complete Validation

```
□ Prerequisites installed
  □ Rust 1.70+
  □ Python 3.9+ with numpy, scipy

□ Basic validation
  □ cargo build succeeds
  □ cargo test passes (96 tests)
  □ Example runs correctly

□ Benchmark validation
  □ cargo bench completes
  □ FFT faster than naive for N ≥ 256
  □ Speedup increases with N

□ Manual verification
  □ Circulant multiply matches expected values
  □ Quantum walk preserves norm
  □ FFT matches naive implementation

□ Python comparison
  □ Python benchmark runs
  □ Results are in expected range
  □ Rust competitive with NumPy FFT

□ (Optional) Profiling
  □ No obvious performance bottlenecks
  □ Memory usage is O(N) not O(N²)
```

---

## Quick Reference Commands

```bash
# Build
cargo build --release --features "physics parallel serde"

# Test
cargo test --features "physics parallel serde"

# Bench
cargo bench --features "physics parallel serde"

# Example
cargo run --example quantum_walk_1d --features "physics parallel serde"

# Docs
cargo doc --open --features "physics parallel serde"

# Clean
cargo clean

# Python comparison
cd benchmarks/python && python compare_circulant.py
```

---

*Happy testing!*
