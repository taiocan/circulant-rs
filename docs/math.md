# Mathematical Foundations

**Owner:** math_expert | **Version:** 0.2.2 | **Updated:** 2026-01-28

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
