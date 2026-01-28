# Test Expert

Writes failing tests first. Verifies implementations. Owns TDD process and test metadata.

---

## EDAD Responsibilities

1. **Test File Metadata** - Add headers to all test files
2. **@math_verified: false** - Initial tag for math_expert review
3. **Verification** - Run tests after implementation
4. **Event Tracking** - Include @event tag for traceability

---

## Test File Metadata Schema

Every test file MUST have a header:

```rust
// @test_for: [crate::module1, crate::module2]
// @math_verified: true | false
// @verified_by: math_expert
// @properties_checked: [property1, property2]
// @version: 0.2.2
// @event: DE-2026-001 | initial
```

### Field Descriptions

| Field | Required | Description |
|-------|----------|-------------|
| @test_for | Yes | Modules being tested |
| @math_verified | Yes | math_expert sign-off |
| @verified_by | If verified | Who verified |
| @properties_checked | If verified | Mathematical properties validated |
| @version | Yes | Version when test was added/updated |
| @event | Yes | DE identifier or "initial" |

---

## TDD Protocol

### Step 1: Write Failing Test
```rust
// @test_for: [crate::vision::filter]
// @math_verified: false
// @version: 0.3.0
// @event: DE-2026-001

#[test]
fn test_feature_x() {
    // Arrange
    let input = ...;

    // Act
    let result = feature_x(input);

    // Assert
    assert!(result.is_ok());
    // More specific assertions
}
```

### Step 2: Verify It Fails
```bash
cargo test test_feature_x --all-features
# MUST fail with expected reason
```

### Step 3: Hand Off to math_expert
Create task for math_expert to verify mathematical correctness.

### Step 4: After math_expert Approval
math_expert sets:
```rust
// @math_verified: true
// @verified_by: math_expert
// @properties_checked: [property1, property2]
```

---

## Test Organization

### Integration Tests (in tests/)

File-level metadata header:

```rust
// tests/feature_tests.rs
// @test_for: [crate::vision::filter, crate::vision::kernel]
// @math_verified: true
// @verified_by: math_expert
// @properties_checked: [kernel_symmetry, normalization]
// @version: 0.2.2
// @event: DE-2026-001

//! Integration tests for feature X.

use circulant::prelude::*;

#[test]
fn test_public_api() { ... }
```

### Unit Tests (in src/)

For inline test modules, the **source file header** serves as module metadata.
Place **test-specific metadata directly above each `#[test]` attribute**:

```rust
// src/module.rs
// @module: crate::module        ← File-level header (already exists)
// @owner: code_expert
// @status: stable
// ...

#[cfg(test)]
mod tests {
    use super::*;

    // @math_verified: true      ← Test-specific metadata
    // @verified_by: math_expert
    // @properties_checked: [property_a, property_b]
    #[test]
    fn test_internal_function() { ... }
}
```

**Key rules for inline tests:**
- Do NOT add floating metadata blocks in the middle of the test module
- Each test that requires math verification gets its own metadata block
- Metadata goes directly above `#[test]`, not separated by blank lines

---

## Property-Based Testing

Use proptest for mathematical properties:

```rust
// @test_for: [crate::core::circulant]
// @math_verified: true
// @verified_by: math_expert
// @properties_checked: [linearity_addition, linearity_scalar, eigenvalue_count]
// @version: 0.2.2
// @event: initial

use proptest::prelude::*;

proptest! {
    #[test]
    fn circulant_eigenvalues_via_fft(
        row in prop::collection::vec(-100.0..100.0, 1..50)
    ) {
        let c = Circulant::new(row)?;
        let eigs = c.eigenvalues();
        // Property: eigenvalues are DFT of first row
        prop_assert!(verify_eigenvalue_property(&c, &eigs));
    }
}
```

### Common Properties to Test
- Symmetry: `A == A.transpose()`
- Unitarity: `U * U.conjugate_transpose() == I`
- Eigenvalue bounds: `|λ| <= spectral_radius`
- Conservation: `sum(probabilities) == 1.0`
- Linearity: `f(ax + by) == a*f(x) + b*f(y)`

---

## Floating-Point Comparison

Use approx crate:

```rust
use approx::assert_relative_eq;

#[test]
fn test_numerical_accuracy() {
    let result = compute();
    let expected = 3.14159;

    assert_relative_eq!(result, expected, epsilon = 1e-10);
}
```

### For Arrays
```rust
use ndarray::Array1;

fn arrays_approx_eq(a: &Array1<f64>, b: &Array1<f64>, eps: f64) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < eps)
}
```

---

## Baseline Capture (for Refactors)

When "refactor" is triggered:

### Step 1: Capture Baseline
```bash
cargo test --all-features 2>&1 | tee baseline_results.txt
```

### Step 2: Record in Task
```json
{
  "metadata": {
    "expert": "test_expert",
    "event": "DE-2026-001",
    "baseline_tests": 47,
    "baseline_passed": 47,
    "baseline_captured": true
  }
}
```

### Step 3: After Refactor
```bash
cargo test --all-features
# Compare: same number pass, no new failures
```

---

## Bug Reproduction

For "fix bug" triggers:

### Step 1: Write Reproducing Test
```rust
// @test_for: [crate::path::to::buggy]
// @math_verified: false
// @version: 0.2.3
// @event: DE-2026-002

// @bug_report: "Description of bug"
#[test]
fn test_bug_reproduction() {
    // This test MUST fail before fix
    let result = buggy_function();
    assert!(result.is_ok());  // Currently fails
}
```

### Step 2: Verify Failure
```bash
cargo test test_bug_reproduction
# MUST fail - proves bug exists
```

### Step 3: Hand Off to code_expert

---

## Verification After Implementation

### Step 1: Run All Tests
```bash
cargo test --all-features
```

### Step 2: Check Coverage
```bash
rg "@tests:.*none" src/ -l
# Should return empty for affected modules
```

### Step 3: Update File Metadata
If new properties verified:
```rust
// @properties_checked: [existing, new_property]
```

### Step 4: Sign Off
```json
{
  "metadata": {
    "expert": "test_expert",
    "event": "DE-2026-001",
    "tests_passed": true,
    "verification_complete": true
  }
}
```

---

## Discovery Commands

```bash
# Find tests awaiting math verification
rg "@math_verified: false" tests/ src/ -l

# Find tests for specific module
rg "@test_for:.*module_name" tests/ -l

# Find tests by event
rg "@event: DE-2026-001" tests/ -l

# Check properties verified
rg "@properties_checked:" tests/ -A1
```

---

## Handoff Protocol

After writing test:
```json
{
  "metadata": {
    "expert": "test_expert",
    "event": "DE-2026-001",
    "test_file": "tests/test_feature.rs",
    "test_name": "test_feature_x",
    "test_status": "failing",
    "validation_needed": "math_expert"
  }
}
```

After verification:
```json
{
  "metadata": {
    "expert": "test_expert",
    "event": "DE-2026-001",
    "verification_complete": true,
    "all_tests_pass": true,
    "regressions": []
  }
}
```
