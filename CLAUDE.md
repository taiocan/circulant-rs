# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

circulant-rs is a Rust library for high-performance block-circulant matrix operations. It exploits FFT to reduce O(N²) matrix operations to O(N log N), enabling efficient simulation of quantum walks, signal processing, and lattice-based physics.

**Note:** This project is in early development. No source code exists yet—only a README describing the intended design.

## Build Commands

Once the project has a Cargo.toml:
- `cargo build` - Build the library
- `cargo test` - Run all tests
- `cargo test <test_name>` - Run a single test
- `cargo clippy` - Run linter
- `cargo doc --open` - Generate and view documentation

## Intended Architecture

The README describes a layered trait architecture:

1. **Core Layer** - 1D FFT-based circulant matrix multiplication
2. **Block Layer** - `BlockCirculant<T>` struct for "matrix-of-matrices" operations
3. **Traits Layer** - Domain-specific interfaces (`QuantumWalk`, `ConvolutionalFilter`)
4. **Backends Layer** - Pluggable FFT backends (default: rustfft)

### Target Modules

- `circulant_rs::physics` - Quantum walk simulation (`CoinedWalk1D`, `Coin::Hadamard`)
- `circulant_rs::vision` - Image processing with BCCB filters (`BCCBFilter`, `Kernel`)
- `circulant_rs::visualize` - Plotting probability distributions

### Key Dependencies to Consider

- `rustfft` - FFT computation
- `num-complex` - Complex number support for quantum amplitudes
- `rayon` - Parallel execution
- A plotting library for visualization
