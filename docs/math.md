# Mathematical Foundations

**Owner:** math_expert | **Version:** 1.0.0 | **Updated:** 2026-01-28

> AI-facing documentation of algorithms, proofs, and numerical considerations.

---

## Core Algorithms

### Circulant Matrix Multiplication

A circulant matrix C is fully determined by its first row (generator) g = [g₀, g₁, ..., gₙ₋₁]:

```
C = ⌈ g₀    g_{n-1}  g_{n-2}  ...  g₁   ⌉
    | g₁    g₀       g_{n-1}  ...  g₂   |
    | g₂    g₁       g₀       ...  g₃   |
    | ...                               |
    ⌊ g_{n-1} g_{n-2} g_{n-3} ...  g₀   ⌋
```

**Multiplication Algorithm:**

Given Cx = y, we compute:
1. λ = FFT(g) — eigenvalues of C
2. x̂ = FFT(x) — transform input
3. ŷ = λ ⊙ x̂ — element-wise product
4. y = IFFT(ŷ) — inverse transform

**Complexity:** O(N log N) vs O(N²) for naive multiplication

**Proof:** Circulant matrices are diagonalized by the DFT matrix F:
- C = F⁻¹ Λ F where Λ = diag(FFT(g))
- Cx = F⁻¹ Λ F x = F⁻¹ (Λ (Fx)) = IFFT(λ ⊙ FFT(x))

### Block Circulant (BCCB) Matrix Multiplication

For 2D operations, we use Block Circulant with Circulant Blocks (BCCB):

**Structure:** Each block is itself circulant, and blocks are arranged circulantly.

**Algorithm (2D FFT-based):**
1. λ = FFT2D(G) — 2D eigenvalues from generator
2. X̂ = FFT2D(X) — transform input
3. Ŷ = λ ⊙ X̂ — element-wise product
4. Y = IFFT2D(Ŷ) — inverse transform

**Complexity:** O(N² log N) for N×N matrix

### N-Dimensional Circulant Tensor Multiplication

For N-D operations, we generalize BCCB to N-dimensional circulant tensors.

**Definition:** An N-D circulant tensor C of shape [n₁, n₂, ..., n_D] is defined by a generator tensor G of the same shape, where:

```
C[i₁, i₂, ..., i_D, j₁, j₂, ..., j_D] = G[(i₁-j₁) mod n₁, (i₂-j₂) mod n₂, ..., (i_D-j_D) mod n_D]
```

**N-D FFT Diagonalization Theorem:**

Any N-D circulant tensor is diagonalized by the N-D DFT:
```
C = F_D⁻¹ Λ F_D
```

where:
- F_D = F₁ ⊗ F₂ ⊗ ... ⊗ F_D (Kronecker product of 1D DFT matrices)
- Λ = diag(FFT_ND(G)) (eigenvalues from N-D FFT of generator)

**Separable N-D FFT:**

N-D FFT is computed by applying 1D FFT along each axis sequentially:
```
FFT_ND(X) = FFT_axis(D-1, FFT_axis(D-2, ... FFT_axis(0, X)))
```

**Algorithm (N-D FFT-based):**
```
Input: Generator G[n₁,...,n_D], tensor X of same shape
Output: Y = C·X

Algorithm:
  1. Reshape X to tensor (if needed)
  2. λ ← FFT_ND(G)           // Separable: FFT along each axis
  3. X̂ ← FFT_ND(X)
  4. Ŷ ← λ ⊙ X̂              // Element-wise product
  5. Y ← IFFT_ND(Ŷ)

Complexity: O(N log N) where N = ∏ nᵢ
```

**Memory:** O(N) for generator storage vs O(N²) for dense matrix representation.

**Properties:**
1. **Eigenvalue count:** N-D tensor has ∏ᵢ nᵢ eigenvalues
2. **Linearity:** C(αx + βy) = αC(x) + βC(y)
3. **Commutativity:** C₁C₂ = C₂C₁ (same shape tensors)
4. **Associativity:** (C₁C₂)x = C₁(C₂x)
5. **Identity:** When G has 1 at origin and 0 elsewhere, C·x = x

### FFT-Based Eigenvalue Computation

For circulant matrix C with generator g:

**Eigenvalues:** λₖ = Σⱼ gⱼ ω^{jk} where ω = e^{-2πi/n}

This is exactly the DFT of g:
```rust
eigenvalues = fft(&generator)
```

**Eigenvectors:** Columns of the DFT matrix F, where F_{jk} = ω^{jk}/√n

---

## Numerical Stability

### Tolerance Justifications

| Operation | Epsilon | Rationale |
|-----------|---------|-----------|
| FFT roundtrip | 1e-10 | Double precision (64-bit) limit ~1e-15, allow for accumulation |
| Probability sum | 1e-9 | Accumulated error over many amplitudes |
| Unitarity check | 1e-10 | Matrix product accumulates error |
| Eigenvalue comparison | 1e-8 | Large matrix FFT accumulates more error |
| N-D FFT roundtrip | 1e-10 | Same as 1D/2D |
| N-D eigenvalue comparison | 1e-8 × D | Accumulation scales with dimension count |
| Large tensor (>1M elements) | 1e-9 | Higher accumulation for very large tensors |

### Precision Considerations

**FFT Precision:**
- RustFFT uses Cooley-Tukey algorithm
- Numerical error grows as O(log N) for size N
- For N = 10⁶, expect ~20 bits of accumulated error

**Complex Arithmetic:**
- Use `Complex<f64>` throughout (not `Complex<f32>`)
- Avoid explicit magnitude computations when possible (use |z|² instead)

### Condition Numbers

**Circulant Matrix Condition:**
```
cond(C) = max|λᵢ| / min|λᵢ|
```

Operations on ill-conditioned matrices may lose precision. Document when condition checking is needed.

---

## Property Invariants

### Unitarity (Quantum Coins)

A matrix U is unitary if U†U = I.

**Test Strategy:**
```rust
// For coin matrix U
let product = u.conjugate_transpose() * &u;
let identity = Array2::eye(dim);
assert!(matrices_approx_eq(&product, &identity, 1e-10));
```

**Standard Coins:**
- Hadamard: H = (1/√2)[[1, 1], [1, -1]]
- Grover(d): G = 2|ψ⟩⟨ψ| - I where |ψ⟩ = (1/√d)Σ|i⟩
- DFT(d): F_{jk} = ω^{jk}/√d

### Probability Conservation

For quantum state |ψ⟩, probability conservation requires:
```
Σᵢ |⟨i|ψ⟩|² = 1
```

**Test Strategy:**
```rust
let total: f64 = state.amplitudes().iter()
    .map(|a| a.norm_sqr())
    .sum();
assert_relative_eq!(total, 1.0, epsilon = 1e-9);
```

### Circulant Commutativity

All circulant matrices of the same size commute:
```
C₁ C₂ = C₂ C₁
```

**Proof:** Both are diagonalized by the same DFT matrix.

### Circulant Eigenvalue Properties

1. **Count:** N×N circulant has exactly N eigenvalues
2. **Reality:** If generator is real and symmetric, eigenvalues are real
3. **Determinant:** det(C) = ∏ᵢ λᵢ
4. **Trace:** tr(C) = Σᵢ λᵢ = N × g₀

### N-D Circulant Tensor Properties

1. **Eigenvalue count:** ∏ᵢ nᵢ eigenvalues for shape [n₁, n₂, ..., n_D]
2. **Linearity:** C(αx + βy) = αC(x) + βC(y) for scalars α, β
3. **FFT correctness:** IFFT_ND(FFT_ND(X)) = X within 1e-10
4. **Convolution theorem:** C·x = IFFT_ND(FFT_ND(G) ⊙ FFT_ND(x))
5. **Separability:** FFT_ND = ∏ FFT_1D (sequential along each axis)
6. **Identity generator:** G with 1 at origin, 0 elsewhere → C·x = x

---

## Quantum Walk Mathematics

### Coined Quantum Walk (1D)

**State Space:** ℋ = ℋ_position ⊗ ℋ_coin

**Evolution Operator:** U = S · (I ⊗ C)
- C: Coin operator (unitary on coin space)
- S: Conditional shift operator

**Shift Operator:**
```
S|x, 0⟩ = |x-1, 0⟩  (left shift for coin 0)
S|x, 1⟩ = |x+1, 1⟩  (right shift for coin 1)
```

With periodic boundaries on n positions:
```
S = Σₓ |x-1 mod n⟩⟨x| ⊗ |0⟩⟨0| + |x+1 mod n⟩⟨x| ⊗ |1⟩⟨1|
```

**Circulant Structure:** The shift operators are circulant permutation matrices.

### Ballistic Spreading

Quantum walks exhibit ballistic spreading (σ ∝ t) vs classical diffusive spreading (σ ∝ √t).

**Test:** After t steps from localized initial state, probability should reach distance ~t from origin.

### Hadamard Walk Asymmetry

With initial state |0⟩ (coin pointing left), Hadamard walk is asymmetric:
- More probability accumulates to the left
- Use balanced initial state (|0⟩ + i|1⟩)/√2 for symmetric spreading

---

## FFT Properties

### Linearity
```
FFT(αx + βy) = α·FFT(x) + β·FFT(y)
```

### Parseval's Theorem
```
||x||² = ||FFT(x)||² / N
```

### Convolution Theorem
```
FFT(x * y) = FFT(x) · FFT(y)
```

Where * is circular convolution and · is element-wise product.

### Inverse Relationship
```
IFFT(FFT(x)) = x
FFT(IFFT(x)) = x
```

---

## Algorithm Specifications

### Circulant::mul_vec

```
Input: generator g[0..n], vector x[0..n]
Output: y = C·x where C is circulant matrix from g

Algorithm:
  1. λ ← FFT(g)           // O(n log n)
  2. x̂ ← FFT(x)           // O(n log n)
  3. ŷ ← λ ⊙ x̂            // O(n) element-wise
  4. y ← IFFT(ŷ)          // O(n log n)
  return y

Complexity: O(n log n) time, O(n) space
```

### Circulant::precompute

```
Purpose: Cache FFT(generator) for repeated multiplications

Algorithm:
  1. λ ← FFT(generator)
  2. Store λ in cached_spectrum

Subsequent mul_vec skips step 1, reducing cost by 1/3.
```

### CirculantTensor::mul_tensor (N-D)

```
Input: generator G[n₁,...,n_D], tensor X[n₁,...,n_D]
Output: Y = C·X where C is N-D circulant tensor from G

Algorithm:
  1. If cached_spectrum exists:
       λ ← cached_spectrum
     Else:
       λ ← FFT_ND(G)          // Separable N-D FFT
  2. X̂ ← FFT_ND(X)            // Separable N-D FFT
  3. Ŷ ← λ ⊙ X̂               // O(N) element-wise product
  4. Y ← IFFT_ND(Ŷ)           // Separable N-D IFFT
  return Y

FFT_ND implementation (separable):
  For each axis k in 0..D:
    Apply 1D FFT along axis k to all lanes

Complexity: O(N log N) time where N = ∏ nᵢ, O(N) space
```

### CirculantTensor::mul_tensor_parallel (N-D with rayon)

```
Purpose: Parallel N-D tensor multiplication for large tensors

Algorithm:
  1. λ ← FFT_ND_parallel(G)    // Parallel lanes per axis
  2. X̂ ← FFT_ND_parallel(X)
  3. Ŷ ← λ ⊙ X̂                // Element-wise (parallelizable)
  4. Y ← IFFT_ND_parallel(Ŷ)

Parallel FFT_ND:
  For each axis k in 0..D:
    Extract all lanes along axis k
    Process lanes in parallel with rayon
    Scatter results back to tensor

Threshold: Use parallel for total_size > 32,768 elements
Scaling: Near-linear speedup up to ~8 cores for large tensors
```

### QuantumWalk::step

```
Input: state ψ ∈ ℂ^(n×d) where n=positions, d=coin_dim
Output: ψ' = U·ψ where U = S·(I⊗C)

Algorithm:
  1. ψ_coined ← apply_coin(ψ, C)  // O(n·d²)
  2. ψ' ← apply_shift(ψ_coined)   // O(n·d) via circulant

Complexity: O(n·d²) time (dominated by coin, shift is O(n·d))
```

---

## Common Pitfalls

### Avoid
- Computing eigenvalues via characteristic polynomial (numerically unstable)
- Direct matrix inversion (use linear solve)
- Storing full N×N matrix when circulant structure available
- Using `==` for floating-point comparison

### Prefer
- FFT-based eigenvalue computation
- Iterative methods for large systems
- Storing only the generator vector
- Using `assert_relative_eq!` with appropriate epsilon

---

## References

1. Gray, R.M. "Toeplitz and Circulant Matrices: A Review"
2. Golub & Van Loan. "Matrix Computations"
3. Kempe, J. "Quantum Random Walks: An Introductory Overview"
4. Venegas-Andraca, S.E. "Quantum walks: a comprehensive review"
5. Van Loan, C. "Computational Frameworks for the Fast Fourier Transform" (Ch. 4: Multi-dimensional FFT)
6. Davis, P.J. "Circulant Matrices" (Ch. 7: Block circulant generalizations)
7. Jain, A.K. "Fundamentals of Digital Image Processing" (Ch. 5: 2D transforms)
