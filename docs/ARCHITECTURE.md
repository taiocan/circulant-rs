# Architecture: circulant-rs

**Version:** 0.1.0 | **Updated:** 2024-01-26 | **Reading time:** 30 min

> In-depth technical description for developers who want to understand, extend, or contribute.

## Table of Contents

1. [Mathematical Foundation](#mathematical-foundation)
2. [Module Architecture](#module-architecture)
3. [Type System and Traits](#type-system-and-traits)
4. [Core Implementation Details](#core-implementation-details)
5. [FFT Backend Abstraction](#fft-backend-abstraction)
6. [Physics Module Deep Dive](#physics-module-deep-dive)
7. [Memory Layout and Performance](#memory-layout-and-performance)
8. [Serialization Strategy](#serialization-strategy)
9. [Error Handling Philosophy](#error-handling-philosophy)
10. [Extension Points](#extension-points)

---

## Mathematical Foundation

### 1. Circulant Matrix Definition

A circulant matrix **C** ∈ ℂⁿˣⁿ is completely determined by its first row (the *generator*) `c = [c₀, c₁, ..., cₙ₋₁]`:

```
C[i,j] = c[(j - i) mod n]
```

This means each row is a cyclic right-shift of the previous row:

```
    | c₀    c₁    c₂    ...  cₙ₋₁ |
    | cₙ₋₁  c₀    c₁    ...  cₙ₋₂ |
C = | cₙ₋₂  cₙ₋₁  c₀    ...  cₙ₋₃ |
    | ...                         |
    | c₁    c₂    c₃    ...  c₀   |
```

### 2. DFT Diagonalization Theorem

**Theorem**: Every circulant matrix C can be diagonalized by the DFT matrix F:

```
C = F⁻¹ · D · F
```

Where:
- **F** is the n×n DFT matrix: `F[j,k] = ω^(jk)` where `ω = e^(-2πi/n)`
- **D** is diagonal with eigenvalues `λₖ = Σⱼ c[j] · ω^(jk)` (the DFT of generator)
- **F⁻¹** is the inverse DFT matrix (IDFT)

**Proof sketch**: The eigenvectors of any circulant matrix are the columns of F (the Fourier modes), and the eigenvalues are the DFT of the first row.

### 3. FFT-Based Multiplication Algorithm

For computing **y = C·x**:

```
Algorithm FFT_MULTIPLY(generator c, vector x):
    1. ĉ ← FFT(c)           // O(n log n) - compute eigenvalues
    2. x̂ ← FFT(x)           // O(n log n) - transform input
    3. ŷ ← ĉ ⊙ x̂            // O(n) - element-wise multiply
    4. y ← IFFT(ŷ)          // O(n log n) - inverse transform
    return y
```

**Total complexity**: O(n log n) vs O(n²) for naive matrix multiplication.

### 4. Convolution vs Cross-Correlation

The library implements **cross-correlation** semantics for 1D circulant multiplication:

```
y[i] = Σⱼ c[(j-i) mod n] · x[j]
```

This corresponds to the circulant matrix C[i,j] = c[(j-i) mod n].

For the FFT formula, cross-correlation requires:
```
spectrum = conj(FFT(conj(c)))    // For complex generators
y = IFFT(spectrum ⊙ FFT(x))
```

For 2D BCCB, we implement **convolution** semantics:
```
y[i,j] = Σₖₗ g[(i-k) mod m, (j-l) mod n] · x[k,l]
```

Which uses the simpler FFT formula:
```
spectrum = FFT2(g)
y = IFFT2(spectrum ⊙ FFT2(x))
```

### 5. Block Circulant with Circulant Blocks (BCCB)

A BCCB matrix is a block circulant matrix where each block is itself circulant. For 2D periodic convolution:

```
BCCB structure:
| B₀    B₁    ...  Bₘ₋₁ |
| Bₘ₋₁  B₀    ...  Bₘ₋₂ |
| ...                    |
| B₁    B₂    ...  B₀   |

Where each Bᵢ is an n×n circulant matrix.
```

BCCB matrices can be diagonalized by the 2D DFT, enabling O(MN log(MN)) operations.

---

## Module Architecture

```
circulant-rs/
├── src/
│   ├── lib.rs              # Crate root, feature gates, public API
│   ├── error.rs            # Error types (CirculantError)
│   ├── prelude.rs          # Convenient re-exports
│   │
│   ├── traits/
│   │   ├── mod.rs          # Module exports
│   │   ├── numeric.rs      # Scalar, ComplexScalar trait bounds
│   │   └── ops.rs          # CirculantOps, BlockOps traits
│   │
│   ├── fft/
│   │   ├── mod.rs          # Module exports
│   │   ├── backend.rs      # FftBackend trait definition
│   │   └── rustfft_backend.rs  # RustFFT implementation
│   │
│   ├── core/
│   │   ├── mod.rs          # Module exports
│   │   ├── circulant.rs    # Circulant<T> - 1D implementation
│   │   ├── block_circulant.rs  # BlockCirculant<T> - 2D BCCB
│   │   └── indexing.rs     # Circular indexing utilities
│   │
│   └── physics/            # Feature-gated (physics)
│       ├── mod.rs          # Module exports
│       ├── state.rs        # QuantumState<T>
│       ├── coin.rs         # Coin operators (Hadamard, Grover, etc.)
│       ├── walk.rs         # QuantumWalk trait
│       └── walk_1d.rs      # CoinedWalk1D implementation
│
├── tests/
│   ├── core_tests.rs       # Integration tests for core module
│   ├── physics_tests.rs    # Integration tests for physics
│   └── property_tests.rs   # Property-based tests (proptest)
│
├── benches/
│   └── fft_multiply.rs     # Criterion benchmarks
│
└── examples/
    └── quantum_walk_1d.rs  # Runnable quantum walk example
```

### Dependency Graph

```
                    ┌─────────────┐
                    │   prelude   │
                    └──────┬──────┘
                           │ re-exports
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│    traits     │  │     core      │  │    physics    │
│  ┌─────────┐  │  │  ┌─────────┐  │  │  ┌─────────┐  │
│  │ Scalar  │  │  │  │Circulant│  │  │  │  State  │  │
│  │CircOps  │◄─┼──┼──┤  Block  │  │  │  │  Coin   │  │
│  │BlockOps │  │  │  └────┬────┘  │  │  │  Walk   │  │
│  └─────────┘  │  │       │       │  │  └────┬────┘  │
└───────────────┘  │       │       │  │       │       │
        ▲          │       ▼       │  │       │       │
        │          │  ┌─────────┐  │  │       │       │
        └──────────┼──┤   fft   │◄─┼──┼───────┘       │
                   │  │ Backend │  │  │               │
                   │  └─────────┘  │  │               │
                   └───────────────┘  └───────────────┘
```

---

## Type System and Traits

### 1. Scalar Trait Hierarchy

```rust
/// Base trait for real scalar types (f32, f64)
pub trait Scalar:
    Float           // From num-traits: floating point operations
    + FloatConst    // Mathematical constants (PI, E, etc.)
    + NumAssign     // +=, -=, *=, /= operations
    + Send          // Safe to send between threads
    + Sync          // Safe to share between threads
    + Copy          // Implicit copy semantics
    + 'static       // No borrowed references
{
    /// Human-readable type name for error messages
    fn type_name() -> &'static str;
}

// Implementations provided for f32 and f64
impl Scalar for f32 { fn type_name() -> &'static str { "f32" } }
impl Scalar for f64 { fn type_name() -> &'static str { "f64" } }
```

### 2. Operation Traits

```rust
/// Operations on 1D circulant matrices
pub trait CirculantOps<T: Scalar> {
    /// Matrix-vector multiplication: y = C·x
    fn mul_vec(&self, x: &[Complex<T>]) -> Result<Vec<Complex<T>>>;

    /// Get eigenvalues (DFT of generator)
    fn eigenvalues(&self) -> Vec<Complex<T>>;

    /// Matrix dimension
    fn size(&self) -> usize;
}

/// Operations on 2D block circulant matrices
pub trait BlockOps<T: Scalar> {
    /// Matrix-array multiplication (2D convolution)
    fn mul_array(&self, x: &Array2<Complex<T>>) -> Result<Array2<Complex<T>>>;

    /// Flattened multiplication (treats 2D as 1D)
    fn mul_vec(&self, x: &Array1<Complex<T>>) -> Result<Array1<Complex<T>>>;

    /// Get 2D eigenvalues (2D DFT of generator)
    fn eigenvalues_2d(&self) -> Array2<Complex<T>>;

    /// Dimensions: (num_blocks, block_rows, block_cols)
    fn dimensions(&self) -> (usize, usize, usize);

    /// Whether this is a true BCCB (both levels circulant)
    fn is_bccb(&self) -> bool;
}
```

### 3. FFT Backend Trait

```rust
/// Abstraction over FFT implementations
pub trait FftBackend<T: Scalar>: Send + Sync {
    /// Forward FFT (DFT): time domain → frequency domain
    fn fft_forward(&self, buffer: &mut [Complex<T>]);

    /// Inverse FFT (IDFT): frequency domain → time domain
    /// Includes 1/n normalization
    fn fft_inverse(&self, buffer: &mut [Complex<T>]);

    /// Transform size
    fn size(&self) -> usize;
}
```

### 4. Quantum Walk Trait

```rust
/// Generic quantum walk interface
pub trait QuantumWalk<T: Scalar + rustfft::FftNum> {
    /// Get the coin operator matrix
    fn coin_operator(&self) -> Array2<Complex<T>>;

    /// Perform one step of the walk (coin + shift)
    fn step(&self, state: &mut QuantumState<T>);

    /// Simulate multiple steps, returning final state
    fn simulate(&self, initial: QuantumState<T>, steps: usize) -> QuantumState<T> {
        let mut state = initial;
        for _ in 0..steps {
            self.step(&mut state);
        }
        state
    }

    /// Number of positions in the walk
    fn num_positions(&self) -> usize;

    /// Dimension of coin space
    fn coin_dim(&self) -> usize;
}
```

---

## Core Implementation Details

### 1. Circulant<T> Structure

```rust
pub struct Circulant<T: Scalar + rustfft::FftNum> {
    /// Generator (first row) - defines the entire matrix
    generator: Vec<Complex<T>>,

    /// Cached eigenvalues for repeated multiplications
    /// Computed as conj(FFT(conj(generator))) for cross-correlation
    cached_spectrum: Option<Vec<Complex<T>>>,

    /// FFT backend instance (shared via Arc)
    fft: Option<Arc<RustFftBackend<T>>>,
}
```

**Key design decisions:**

1. **Generator storage**: Only the first row is stored, reducing O(n²) → O(n) memory
2. **Spectrum caching**: Eigenvalues can be precomputed for repeated multiplications
3. **Arc-wrapped FFT**: The FFT planner is expensive; sharing via Arc avoids redundant planning
4. **Optional fields**: Support serde deserialization (FFT backend isn't serializable)

### 2. Multiplication Algorithm (1D)

```rust
fn mul_vec(&self, x: &[Complex<T>]) -> Result<Vec<Complex<T>>> {
    let n = self.generator.len();
    let fft = self.fft.as_ref().unwrap();

    // Get or compute eigenvalues
    let eigenvalues = match &self.cached_spectrum {
        Some(spectrum) => spectrum.clone(),
        None => {
            // For cross-correlation: spectrum = conj(FFT(conj(c)))
            let mut spectrum: Vec<Complex<T>> = self.generator
                .iter()
                .map(|v| Complex::new(v.re, -v.im))  // Step 1: conjugate
                .collect();
            fft.fft_forward(&mut spectrum);          // Step 2: FFT
            for val in spectrum.iter_mut() {
                *val = Complex::new(val.re, -val.im); // Step 3: conjugate
            }
            spectrum
        }
    };

    // FFT of input
    let mut x_fft = x.to_vec();
    fft.fft_forward(&mut x_fft);

    // Element-wise multiply in frequency domain
    for (y, lambda) in x_fft.iter_mut().zip(eigenvalues.iter()) {
        *y *= *lambda;
    }

    // Inverse FFT to get result
    fft.fft_inverse(&mut x_fft);

    Ok(x_fft)
}
```

### 3. BlockCirculant<T> Structure

```rust
pub struct BlockCirculant<T: Scalar + rustfft::FftNum> {
    /// 3D generator array: [num_block_cols, block_rows, block_cols]
    /// For BCCB, this is typically [1, rows, cols]
    generator: Array3<Complex<T>>,

    /// Outer circulant dimension
    num_blocks: usize,

    /// Inner block dimensions
    block_size: (usize, usize),

    /// True if both levels are circulant (BCCB)
    is_bccb: bool,

    /// Cached 2D spectrum
    cached_spectrum: Option<Array2<Complex<T>>>,

    /// Separate FFT backends for rows and columns
    fft_rows: Option<Arc<RustFftBackend<T>>>,
    fft_cols: Option<Arc<RustFftBackend<T>>>,
}
```

### 4. 2D FFT Implementation

The 2D FFT is implemented as separable 1D FFTs:

```rust
fn compute_spectrum(&self) -> Array2<Complex<T>> {
    let (rows, cols) = self.block_size;

    // For 2D convolution: spectrum = FFT2(generator)
    let mut spectrum = self.generator.slice(s![0, .., ..]).to_owned();

    // FFT along rows (each row independently)
    for i in 0..rows {
        let mut row: Vec<Complex<T>> = spectrum.row(i).to_vec();
        self.fft_cols.fft_forward(&mut row);
        for (j, val) in row.into_iter().enumerate() {
            spectrum[(i, j)] = val;
        }
    }

    // FFT along columns (each column independently)
    for j in 0..cols {
        let mut col: Vec<Complex<T>> = spectrum.column(j).to_vec();
        self.fft_rows.fft_forward(&mut col);
        for (i, val) in col.into_iter().enumerate() {
            spectrum[(i, j)] = val;
        }
    }

    spectrum
}
```

---

## FFT Backend Abstraction

### RustFFT Implementation

```rust
pub struct RustFftBackend<T: Scalar + rustfft::FftNum> {
    forward: Arc<dyn Fft<T>>,   // Forward FFT plan
    inverse: Arc<dyn Fft<T>>,   // Inverse FFT plan
    size: usize,
    scale: T,                    // 1/n normalization factor
}

impl<T: Scalar + rustfft::FftNum> RustFftBackend<T> {
    pub fn new(size: usize) -> Self {
        let mut planner = FftPlanner::new();
        let forward = planner.plan_fft_forward(size);
        let inverse = planner.plan_fft_inverse(size);
        let scale = T::from(size).unwrap().recip();

        Self { forward, inverse, size, scale }
    }
}

impl<T: Scalar + rustfft::FftNum> FftBackend<T> for RustFftBackend<T> {
    fn fft_forward(&self, buffer: &mut [Complex<T>]) {
        self.forward.process(buffer);
    }

    fn fft_inverse(&self, buffer: &mut [Complex<T>]) {
        self.inverse.process(buffer);
        // Apply 1/n normalization (RustFFT doesn't normalize)
        for val in buffer.iter_mut() {
            *val = val.scale(self.scale);
        }
    }

    fn size(&self) -> usize {
        self.size
    }
}
```

**Why RustFFT?**

1. Pure Rust (no C dependencies)
2. SIMD-accelerated on x86_64
3. Arbitrary-length FFT support (not just power-of-2)
4. Thread-safe planning

### Extending with Other Backends

To add a new FFT backend (e.g., FFTW, cuFFT):

```rust
pub struct FftwBackend<T> {
    plan_forward: fftw::Plan<T>,
    plan_inverse: fftw::Plan<T>,
    // ...
}

impl<T: Scalar> FftBackend<T> for FftwBackend<T> {
    fn fft_forward(&self, buffer: &mut [Complex<T>]) {
        // Use FFTW's execute
    }
    // ...
}
```

---

## Physics Module Deep Dive

### 1. Quantum State Representation

```rust
pub struct QuantumState<T: Scalar> {
    /// Amplitude vector: |ψ⟩ = Σ ψ(x,c) |x,c⟩
    /// Layout: [pos0_coin0, pos0_coin1, pos1_coin0, pos1_coin1, ...]
    amplitudes: Array1<Complex<T>>,

    /// Number of spatial positions
    num_positions: usize,

    /// Dimension of coin space (2 for standard walks)
    coin_dim: usize,
}
```

**Index mapping**: Position `x` with coin state `c` maps to index `x * coin_dim + c`

**Invariants**:
- `amplitudes.len() == num_positions * coin_dim`
- Normalized states satisfy `Σ|ψ(x,c)|² = 1`

### 2. Coin Operators

```rust
pub enum Coin {
    /// Hadamard: Creates equal superposition
    /// H = (1/√2) [[1, 1], [1, -1]]
    Hadamard,

    /// Grover diffusion operator of dimension d
    /// G = 2|ψ⟩⟨ψ| - I where |ψ⟩ = (1/√d) Σ|i⟩
    Grover(usize),

    /// Discrete Fourier Transform coin of dimension d
    /// DFT[j,k] = (1/√d) ω^(jk) where ω = e^(2πi/d)
    Dft(usize),

    /// Identity (no mixing)
    Identity(usize),

    /// Custom unitary matrix
    Custom(Array2<Complex<f64>>),
}
```

**Unitarity verification**:

```rust
pub fn is_unitary<T: Scalar>(&self, tolerance: T) -> bool {
    let matrix = self.to_matrix::<T>();
    let d = self.dimension();

    // Check C† · C = I
    for i in 0..d {
        for j in 0..d {
            let mut sum = Complex::new(T::zero(), T::zero());
            for k in 0..d {
                let c_dag_ik = matrix[(k, i)].conj();
                sum += c_dag_ik * matrix[(k, j)];
            }
            let expected = if i == j { T::one() } else { T::zero() };
            if (sum.re - expected).abs() > tolerance || sum.im.abs() > tolerance {
                return false;
            }
        }
    }
    true
}
```

### 3. Coined Quantum Walk Implementation

```rust
pub struct CoinedWalk1D<T: Scalar + rustfft::FftNum> {
    num_positions: usize,
    coin: Coin,
    coin_matrix: Option<Array2<Complex<T>>>,
    fft: Option<Arc<RustFftBackend<T>>>,
    left_shift_spectrum: Option<Vec<Complex<T>>>,  // FFT of [0,0,...,0,1]
    right_shift_spectrum: Option<Vec<Complex<T>>>, // FFT of [0,1,0,...,0]
}
```

**Evolution step**:

```rust
fn step(&self, state: &mut QuantumState<T>) {
    let n = self.num_positions;
    let coin_matrix = self.coin_matrix.as_ref().unwrap();

    // Step 1: Apply coin operator at each position
    // |x,c⟩ → Σ_c' C_{c',c} |x,c'⟩
    let amplitudes = state.amplitudes();
    let mut after_coin = amplitudes.to_vec();

    for pos in 0..n {
        let idx0 = pos * 2;
        let idx1 = pos * 2 + 1;
        let a0 = amplitudes[idx0];
        let a1 = amplitudes[idx1];

        // Apply 2x2 coin matrix
        after_coin[idx0] = coin_matrix[(0,0)] * a0 + coin_matrix[(0,1)] * a1;
        after_coin[idx1] = coin_matrix[(1,0)] * a0 + coin_matrix[(1,1)] * a1;
    }

    // Step 2: Apply shift using FFT
    // |x,0⟩ → |x-1,0⟩ (left shift)
    // |x,1⟩ → |x+1,1⟩ (right shift)

    // Extract coin components
    let coin0: Vec<_> = (0..n).map(|i| after_coin[i * 2]).collect();
    let coin1: Vec<_> = (0..n).map(|i| after_coin[i * 2 + 1]).collect();

    // Apply shifts via FFT multiplication
    let shifted0 = self.apply_shift_fft(&coin0, &self.left_shift_spectrum);
    let shifted1 = self.apply_shift_fft(&coin1, &self.right_shift_spectrum);

    // Recombine
    let state_amps = state.amplitudes_mut();
    for pos in 0..n {
        state_amps[pos * 2] = shifted0[pos];
        state_amps[pos * 2 + 1] = shifted1[pos];
    }
}
```

**Shift spectrum precomputation**:

```rust
fn compute_shift_spectra(n: usize, fft: &RustFftBackend<T>)
    -> (Vec<Complex<T>>, Vec<Complex<T>>)
{
    // Left shift generator: [0, 0, ..., 0, 1]
    // Multiplying by this shifts all elements left by 1
    let mut left_gen = vec![Complex::zero(); n];
    left_gen[n - 1] = Complex::one();
    fft.fft_forward(&mut left_gen);

    // Right shift generator: [0, 1, 0, ..., 0]
    // Multiplying by this shifts all elements right by 1
    let mut right_gen = vec![Complex::zero(); n];
    right_gen[1] = Complex::one();
    fft.fft_forward(&mut right_gen);

    (left_gen, right_gen)
}
```

---

## Memory Layout and Performance

### 1. Cache-Friendly Access Patterns

**QuantumState layout**: `[pos0_c0, pos0_c1, pos1_c0, pos1_c1, ...]`

This interleaved layout ensures that when processing a single position (applying the coin), both coin amplitudes are adjacent in memory, maximizing cache utilization.

**Alternative layout** (not used): `[pos0_c0, pos1_c0, ..., pos0_c1, pos1_c1, ...]`

This would require non-contiguous memory access for coin operations.

### 2. Allocation Strategy

- **Preallocation**: FFT buffers are allocated once and reused
- **Spectrum caching**: `precompute()` method stores eigenvalues
- **Arc sharing**: FFT planners are expensive; sharing avoids redundant work

### 3. Parallelization Opportunities

With `parallel` feature:

```rust
// Parallel coin application (not yet implemented, but possible)
after_coin.par_chunks_mut(2)
    .enumerate()
    .for_each(|(pos, chunk)| {
        let a0 = amplitudes[pos * 2];
        let a1 = amplitudes[pos * 2 + 1];
        chunk[0] = coin_matrix[(0,0)] * a0 + coin_matrix[(0,1)] * a1;
        chunk[1] = coin_matrix[(1,0)] * a0 + coin_matrix[(1,1)] * a1;
    });
```

---

## Serialization Strategy

### 1. Serde Integration

```rust
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CoinedWalk1D<T> {
    num_positions: usize,
    coin: Coin,

    // Skip non-serializable fields
    #[cfg_attr(feature = "serde", serde(skip))]
    coin_matrix: Option<Array2<Complex<T>>>,

    #[cfg_attr(feature = "serde", serde(skip))]
    fft: Option<Arc<RustFftBackend<T>>>,

    #[cfg_attr(feature = "serde", serde(skip))]
    left_shift_spectrum: Option<Vec<Complex<T>>>,

    #[cfg_attr(feature = "serde", serde(skip))]
    right_shift_spectrum: Option<Vec<Complex<T>>>,
}
```

### 2. Reinitialization After Deserialization

```rust
impl<T> CoinedWalk1D<T> {
    /// Reinitialize computed fields after deserialization
    pub fn ensure_initialized(&mut self) {
        if self.fft.is_none() {
            let fft = Arc::new(RustFftBackend::new(self.num_positions));
            let (left, right) = Self::compute_shift_spectra(self.num_positions, &fft);
            self.fft = Some(fft);
            self.left_shift_spectrum = Some(left);
            self.right_shift_spectrum = Some(right);
            self.coin_matrix = Some(self.coin.to_matrix());
        }
    }
}
```

**Usage pattern**:

```rust
// Serialize
let walk = CoinedWalk1D::new(256, Coin::Hadamard);
let bytes = bincode::serialize(&walk)?;

// Deserialize
let mut restored: CoinedWalk1D<f64> = bincode::deserialize(&bytes)?;
restored.ensure_initialized();  // Rebuild FFT plans
```

---

## Error Handling Philosophy

### Error Types

```rust
#[derive(Debug, Error)]
pub enum CirculantError {
    #[error("Generator cannot be empty")]
    EmptyGenerator,

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Invalid block structure: {0}")]
    InvalidBlockStructure(String),

    #[error("Invalid quantum state: {0}")]
    InvalidState(String),
}

pub type Result<T> = std::result::Result<T, CirculantError>;
```

### Design Principles

1. **Fail fast**: Validate inputs at API boundaries
2. **Informative messages**: Include expected vs actual values
3. **No panics in library code**: Return `Result` for recoverable errors
4. **Panic for invariant violations**: Use `assert!` for internal logic errors

---

## Extension Points

### 1. Adding a New Coin Operator

```rust
// In coin.rs, extend the enum:
pub enum Coin {
    // ... existing variants ...

    /// Your custom coin
    MyCustomCoin { param: f64 },
}

// Implement to_matrix:
impl Coin {
    pub fn to_matrix<T: Scalar>(&self) -> Array2<Complex<T>> {
        match self {
            Coin::MyCustomCoin { param } => {
                // Build your unitary matrix
                let theta = T::from(*param).unwrap();
                Array2::from_shape_vec((2, 2), vec![
                    Complex::new(theta.cos(), T::zero()),
                    Complex::new(theta.sin(), T::zero()),
                    Complex::new(-theta.sin(), T::zero()),
                    Complex::new(theta.cos(), T::zero()),
                ]).unwrap()
            }
            // ... existing cases ...
        }
    }
}
```

### 2. Implementing a 2D Quantum Walk

```rust
pub struct CoinedWalk2D<T: Scalar + rustfft::FftNum> {
    rows: usize,
    cols: usize,
    coin: Coin,  // 4D coin for 2D walk
    // Use BlockCirculant for 2D shifts
    shift_up: BlockCirculant<T>,
    shift_down: BlockCirculant<T>,
    shift_left: BlockCirculant<T>,
    shift_right: BlockCirculant<T>,
}

impl<T> QuantumWalk<T> for CoinedWalk2D<T> {
    // ...
}
```

### 3. Adding GPU Acceleration

```rust
// New module: src/fft/cuda_backend.rs
pub struct CudaFftBackend<T> {
    device: cuda::Device,
    plan: cufft::Plan,
    // ...
}

impl<T: Scalar> FftBackend<T> for CudaFftBackend<T> {
    fn fft_forward(&self, buffer: &mut [Complex<T>]) {
        // Transfer to GPU, execute, transfer back
    }
    // ...
}
```

---

## Testing Strategy

### 1. Unit Tests

- FFT roundtrip: `IFFT(FFT(x)) == x`
- Eigenvalue verification: `eigenvalues == DFT(generator)`
- Unitarity: `C† · C == I` for all coins

### 2. Property-Based Tests (proptest)

```rust
proptest! {
    #[test]
    fn test_linearity(
        gen in complex_vec(8),
        x in complex_vec(8),
        y in complex_vec(8),
        alpha in -10.0..10.0f64,
    ) {
        let c = Circulant::new(gen)?;
        // C(αx + y) = αC(x) + C(y)
        let lhs = c.mul_vec(&(alpha * &x + &y))?;
        let rhs = alpha * c.mul_vec(&x)? + c.mul_vec(&y)?;
        assert_vectors_close(&lhs, &rhs, 1e-10);
    }
}
```

### 3. Integration Tests

- Full quantum walk simulation
- Probability conservation over many steps
- Ballistic spreading verification

### 4. Benchmarks

```rust
fn benchmark_fft_multiply(c: &mut Criterion) {
    for size in [64, 256, 1024, 4096, 16384] {
        c.bench_function(&format!("circulant_mul_{}", size), |b| {
            let circulant = Circulant::from_real(vec![1.0; size]).unwrap();
            let x = vec![Complex::new(1.0, 0.0); size];
            b.iter(|| circulant.mul_vec(&x))
        });
    }
}
```

---

## References

1. Davis, P.J. (1979). *Circulant Matrices*. Wiley.
2. Gray, R.M. (2006). "Toeplitz and Circulant Matrices: A Review". *Foundations and Trends in Communications and Information Theory*.
3. Kempe, J. (2003). "Quantum random walks: An introductory overview". *Contemporary Physics*, 44(4), 307-327.
4. Portugal, R. (2013). *Quantum Walks and Search Algorithms*. Springer.

---

## Glossary

| Term | Definition |
|------|------------|
| **Circulant** | Matrix where each row is a cyclic shift of the previous |
| **BCCB** | Block Circulant with Circulant Blocks |
| **DFT** | Discrete Fourier Transform |
| **FFT** | Fast Fourier Transform (efficient DFT algorithm) |
| **Generator** | First row of a circulant matrix (defines entire matrix) |
| **Spectrum** | Eigenvalues of a circulant (DFT of generator) |
| **Coined walk** | Quantum walk with internal coin degree of freedom |
| **Ballistic spreading** | Linear spreading σ ∝ t (vs diffusive σ ∝ √t) |
