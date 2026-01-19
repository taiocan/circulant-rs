#!/usr/bin/env python3
"""
Benchmark comparison script for circulant matrix operations.

Compares:
- NumPy FFT-based circulant multiply
- SciPy dense circulant multiply
- NumPy naive O(N²) multiply

Run with: python compare_circulant.py

Requirements: numpy, scipy
"""

import numpy as np
from scipy.linalg import circulant as scipy_circulant
import timeit
import json
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkResult:
    size: int
    method: str
    time_us: float
    throughput_mops: float  # Million operations per second


def generate_random_complex(n: int) -> np.ndarray:
    """Generate random complex vector."""
    return np.random.randn(n) + 1j * np.random.randn(n)


def numpy_fft_multiply(generator: np.ndarray, x: np.ndarray) -> np.ndarray:
    """FFT-based circulant multiply (cross-correlation semantics)."""
    # For cross-correlation: y = IFFT(conj(FFT(conj(c))) * FFT(x))
    spectrum = np.conj(np.fft.fft(np.conj(generator)))
    return np.fft.ifft(spectrum * np.fft.fft(x))


def scipy_dense_multiply(generator: np.ndarray, x: np.ndarray) -> np.ndarray:
    """SciPy dense circulant matrix multiply."""
    C = scipy_circulant(generator)
    return C @ x


def naive_multiply(generator: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Naive O(N²) circulant multiply."""
    n = len(generator)
    result = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            idx = (j - i) % n
            result[i] += generator[idx] * x[j]
    return result


def benchmark_method(func, generator, x, warmup=100, iterations=1000) -> float:
    """Benchmark a method and return average time in microseconds."""
    # Warm-up
    for _ in range(warmup):
        func(generator, x)

    # Measure
    start = timeit.default_timer()
    for _ in range(iterations):
        func(generator, x)
    elapsed = timeit.default_timer() - start

    return (elapsed / iterations) * 1e6  # Convert to microseconds


def run_1d_benchmarks(sizes: list[int], output_file: Optional[str] = None):
    """Run 1D circulant benchmarks across different sizes."""
    results = []

    print("=" * 70)
    print("1D Circulant Matrix-Vector Multiplication Benchmark")
    print("=" * 70)
    print(f"{'Size':>10} {'NumPy FFT':>15} {'SciPy Dense':>15} {'Naive':>15} {'FFT Speedup':>12}")
    print("-" * 70)

    for size in sizes:
        generator = generate_random_complex(size)
        x = generate_random_complex(size)

        # NumPy FFT (always run)
        t_fft = benchmark_method(numpy_fft_multiply, generator, x)
        results.append(BenchmarkResult(size, "numpy_fft", t_fft, size / t_fft))

        # SciPy dense (skip for large sizes)
        if size <= 8192:
            t_dense = benchmark_method(scipy_dense_multiply, generator, x,
                                       warmup=10, iterations=100)
            results.append(BenchmarkResult(size, "scipy_dense", t_dense, size / t_dense))
            dense_str = f"{t_dense:>12.2f} µs"
            speedup = t_dense / t_fft
        else:
            dense_str = "N/A (too slow)"
            speedup = None

        # Naive (only for small sizes)
        if size <= 1024:
            t_naive = benchmark_method(naive_multiply, generator, x,
                                       warmup=5, iterations=50)
            results.append(BenchmarkResult(size, "naive", t_naive, size / t_naive))
            naive_str = f"{t_naive:>12.2f} µs"
        else:
            naive_str = "N/A"

        speedup_str = f"{speedup:>10.1f}x" if speedup else "N/A"
        print(f"{size:>10} {t_fft:>12.2f} µs {dense_str:>15} {naive_str:>15} {speedup_str:>12}")

    print("=" * 70)

    if output_file:
        with open(output_file, 'w') as f:
            json.dump([r.__dict__ for r in results], f, indent=2)
        print(f"Results saved to {output_file}")

    return results


def run_accuracy_test(sizes: list[int]):
    """Verify FFT method matches naive method."""
    print("\n" + "=" * 70)
    print("Accuracy Verification (FFT vs Naive)")
    print("=" * 70)
    print(f"{'Size':>10} {'Max Abs Error':>20} {'Max Rel Error':>20}")
    print("-" * 70)

    for size in sizes:
        generator = generate_random_complex(size)
        x = generate_random_complex(size)

        result_fft = numpy_fft_multiply(generator, x)
        result_naive = naive_multiply(generator, x)

        abs_error = np.max(np.abs(result_fft - result_naive))
        rel_error = np.max(np.abs(result_fft - result_naive) / (np.abs(result_naive) + 1e-15))

        print(f"{size:>10} {abs_error:>20.2e} {rel_error:>20.2e}")

    print("=" * 70)


def run_memory_comparison(sizes: list[int]):
    """Compare memory usage: generator-only vs dense matrix."""
    print("\n" + "=" * 70)
    print("Memory Usage Comparison")
    print("=" * 70)
    print(f"{'Size':>10} {'Generator (MB)':>15} {'Dense (MB)':>15} {'Ratio':>12}")
    print("-" * 70)

    for size in sizes:
        # Generator: N complex numbers = N * 16 bytes
        gen_bytes = size * 16
        gen_mb = gen_bytes / (1024 * 1024)

        # Dense: N² complex numbers
        dense_bytes = size * size * 16
        dense_mb = dense_bytes / (1024 * 1024)

        ratio = dense_bytes / gen_bytes

        if dense_mb > 1024:
            dense_str = f"{dense_mb/1024:>12.1f} GB"
        else:
            dense_str = f"{dense_mb:>12.2f} MB"

        print(f"{size:>10} {gen_mb:>12.4f} MB {dense_str:>15} {ratio:>10.0f}x")

    print("=" * 70)


def run_quantum_walk_comparison(sizes: list[int], steps: int = 100):
    """Benchmark quantum walk simulation (NumPy implementation)."""
    print("\n" + "=" * 70)
    print(f"Quantum Walk Benchmark ({steps} steps)")
    print("=" * 70)
    print(f"{'Positions':>12} {'Time (ms)':>15} {'Time/Step (µs)':>18}")
    print("-" * 70)

    # Hadamard coin
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

    for n_positions in sizes:
        def quantum_walk():
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

            for _ in range(steps):
                # Reshape for coin application
                state_2d = state.reshape(n_positions, 2)

                # Apply coin
                new_state = state_2d @ H.T

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

        # Warm-up
        for _ in range(5):
            quantum_walk()

        # Measure
        iterations = max(1, 100 // (n_positions // 1000 + 1))
        start = timeit.default_timer()
        for _ in range(iterations):
            quantum_walk()
        elapsed = timeit.default_timer() - start

        time_ms = (elapsed / iterations) * 1000
        time_per_step_us = (time_ms * 1000) / steps

        print(f"{n_positions:>12} {time_ms:>12.2f} ms {time_per_step_us:>15.2f} µs")

    print("=" * 70)


def main():
    print("circulant-rs Python Comparison Benchmarks")
    print(f"NumPy version: {np.__version__}")
    print()

    # 1D multiplication benchmarks
    sizes_1d = [64, 256, 1024, 4096, 16384, 65536, 262144]
    run_1d_benchmarks(sizes_1d, output_file="results_1d.json")

    # Accuracy verification
    sizes_accuracy = [64, 256, 512, 1024]
    run_accuracy_test(sizes_accuracy)

    # Memory comparison
    sizes_memory = [1000, 10000, 100000, 1000000]
    run_memory_comparison(sizes_memory)

    # Quantum walk
    sizes_qw = [256, 1024, 4096, 16384, 65536]
    run_quantum_walk_comparison(sizes_qw, steps=100)


if __name__ == "__main__":
    main()
