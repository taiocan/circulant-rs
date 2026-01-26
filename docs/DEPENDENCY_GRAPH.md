# Module Dependency Graph

**Purpose:** Consult this file BEFORE loading context to minimize token usage.
Load ONLY modules related to current task.

---

## Quick Dependency Lookup

### Core Module (`src/core/`)

| File | Dependencies | Tests |
|------|--------------|-------|
| `circulant.rs` | `error.rs`, `fft/rustfft_backend.rs`, `traits/numeric.rs`, `traits/ops.rs` | `tests/core_tests.rs` |
| `block_circulant.rs` | `error.rs`, `fft/rustfft_backend.rs`, `traits/numeric.rs`, `traits/ops.rs` | `tests/core_tests.rs` |
| `indexing.rs` | (standalone) | inline |

### FFT Module (`src/fft/`)

| File | Dependencies | Tests |
|------|--------------|-------|
| `backend.rs` | `traits/numeric.rs` | - |
| `rustfft_backend.rs` | `fft/backend.rs`, `traits/numeric.rs` | inline |

### Traits Module (`src/traits/`)

| File | Dependencies | Tests |
|------|--------------|-------|
| `numeric.rs` | (external: num-complex, num-traits) | inline |
| `ops.rs` | `error.rs`, `traits/numeric.rs` | inline |

### Physics Module (`src/physics/`) - feature: `physics`

| File | Dependencies | Tests |
|------|--------------|-------|
| `state.rs` | `error.rs`, `traits/numeric.rs` | `tests/physics_tests.rs` |
| `coin.rs` | `traits/numeric.rs` | inline |
| `walk.rs` | `physics/state.rs`, `traits/numeric.rs` | - |
| `walk_1d.rs` | `fft/*`, `physics/coin.rs`, `physics/state.rs`, `physics/walk.rs`, `traits/numeric.rs` | inline |
| `walk_2d.rs` | `fft/*`, `physics/coin.rs`, `physics/state.rs`, `physics/walk.rs`, `traits/numeric.rs` | inline |
| `hamiltonian.rs` | `error.rs`, `fft/*`, `physics/state.rs`, `traits/numeric.rs`, `core/circulant.rs` | inline |

### Vision Module (`src/vision/`) - feature: `vision`

| File | Dependencies | Tests |
|------|--------------|-------|
| `kernel.rs` | `error.rs`, `traits/numeric.rs` | inline |
| `filter.rs` | `core/block_circulant.rs`, `error.rs`, `traits/*`, `vision/kernel.rs` | inline |

### Visualize Module (`src/visualize/`) - feature: `visualize`

| File | Dependencies | Tests |
|------|--------------|-------|
| `quantum.rs` | `error.rs` | inline |
| `heatmap.rs` | `error.rs`, `visualize/quantum.rs` | inline |

### Python Bindings (`src/python/`) - feature: `python`

| File | Dependencies | Tests |
|------|--------------|-------|
| `circulant.rs` | `traits/ops.rs`, `core/circulant.rs` | - |
| `physics.rs` | `physics/*` | - |
| `error.rs` | `error.rs` | - |

---

## Task-Based Loading Guide

### Adding/Modifying FFT Operations
```
Load: src/fft/backend.rs, src/fft/rustfft_backend.rs, src/traits/numeric.rs
Test: cargo test fft
```

### Modifying Circulant Matrix Operations
```
Load: src/core/circulant.rs, src/fft/rustfft_backend.rs, src/traits/ops.rs, src/error.rs
Test: cargo test circulant
```

### Modifying Block Circulant (2D) Operations
```
Load: src/core/block_circulant.rs, src/fft/rustfft_backend.rs, src/traits/ops.rs
Test: cargo test block
```

### Adding New Coin Operator
```
Load: src/physics/coin.rs, src/traits/numeric.rs
Test: cargo test coin
```

### Modifying Quantum Walk (1D)
```
Load: src/physics/walk_1d.rs, src/physics/coin.rs, src/physics/state.rs, src/fft/rustfft_backend.rs
Test: cargo test walk_1d
```

### Modifying Quantum Walk (2D)
```
Load: src/physics/walk_2d.rs, src/physics/coin.rs, src/physics/state.rs, src/core/block_circulant.rs
Test: cargo test walk_2d
```

### Modifying Hamiltonian Evolution
```
Load: src/physics/hamiltonian.rs, src/physics/state.rs, src/core/circulant.rs, src/fft/*
Test: cargo test hamiltonian
```

### Adding Vision Filters
```
Load: src/vision/filter.rs, src/vision/kernel.rs, src/core/block_circulant.rs
Test: cargo test --features vision filter
```

### Modifying Error Types
```
Load: src/error.rs
Affected: ALL modules - run full test suite
Test: cargo test --all-features
```

### Modifying Trait Bounds
```
Load: src/traits/numeric.rs, src/traits/ops.rs
Affected: ALL modules - run full test suite
Test: cargo test --all-features
```

---

## Dependency Graph (Visual)

```
                         lib.rs
                            │
           ┌────────────────┼────────────────┐
           │                │                │
           ▼                ▼                ▼
        traits           error            prelude
       ┌──┴──┐              │                │
       │     │              │         (re-exports)
    numeric  ops ◄──────────┘
       │     │
       └──┬──┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
   fft        core ─────────────────┐
    │       ┌──┴──┐                 │
    │       │     │                 │
    ▼       ▼     ▼                 ▼
 backend  circ  block           physics
    │       │     │            ┌───┼───┐
    ▼       │     │            │   │   │
 rustfft◄───┴─────┘          coin state walk
    │                          │   │    │
    │                          └───┼────┘
    │                              │
    └──────────────────────────────┼─────────┐
                                   │         │
                              ┌────┴────┐    │
                              │         │    │
                           walk_1d  walk_2d  │
                              │         │    │
                              └────┬────┘    │
                                   │         │
                              hamiltonian    │
                                             │
                         ┌───────────────────┘
                         │
                    ┌────┴────┐
                    │         │
                  vision  visualize
                 ┌──┴──┐   ┌──┴──┐
                 │     │   │     │
              kernel filter quantum heatmap
```

---

## Feature Flag Matrix

| Module | `std` | `physics` | `parallel` | `serde` | `vision` | `visualize` |
|--------|-------|-----------|------------|---------|----------|-------------|
| core | ✓ | - | ✓ | ✓ | - | - |
| fft | ✓ | - | - | - | - | - |
| traits | ✓ | - | - | - | - | - |
| error | ✓ | - | - | - | - | - |
| physics | ✓ | ✓ | - | ✓ | - | - |
| vision | ✓ | - | - | - | ✓ | - |
| visualize | ✓ | - | - | - | - | ✓ |

---

## Consultation Rules

**BEFORE loading files, check this graph:**
- Loading >3 source files? Consult dependencies first
- Adding imports? Update this file after
- Cross-module changes? Identify all affected modules

**Agent responsibility:** Planner checks before task assignment.

---

## Update Protocol

When adding new modules or changing dependencies:
1. Update this file immediately
2. Add entry to Quick Dependency Lookup
3. Add to Task-Based Loading Guide if applicable
4. Update visual graph if structure changes

---

## Documentation Dependencies

### Public Documents

| Document | Depends On | Update When |
|----------|------------|-------------|
| README.md | - | Release, headline perf change |
| docs/OVERVIEW.md | BENCHMARKS.md | Use case additions |
| docs/USER_GUIDE.md | API (src/), examples/ | API changes |
| docs/ARCHITECTURE.md | src/ internals | Internal refactors |
| docs/WHITEPAPER.md | Core algorithms | Algorithmic changes |
| docs/BENCHMARKS.md | Benchmark results | Perf changes |
| docs/CHANGELOG.md | All | Every release |
| docs/CONTRIBUTING.md | ARCHITECTURE.md | Process changes |

### Update Protocol (Documenter Agent at Release)

1. Update all version headers
2. Add CHANGELOG entry
3. Verify cross-references
4. Run link checker (see DOC_POLICY.md)
