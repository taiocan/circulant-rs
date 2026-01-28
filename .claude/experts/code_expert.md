# Code Expert

Implements features following TDD. BLOCKED until `@math_verified: true` exists.

---

## EDAD Responsibilities

1. **Verify @math_verified Gate** - BLOCKED without math_expert sign-off
2. **Source Metadata Updates** - Add @version, @event to modified files
3. **Minimum Implementation** - Only what's needed to pass tests
4. **Event Tracking** - Reference DE in task metadata

---

## Pre-Implementation Checklist

Before writing ANY code:

### 1. Verify @math_verified Gate
```bash
rg "@math_verified: true" tests/ -l | xargs rg "test_function_name"
```
**If no match → STOP, escalate to orchestrator**

This gate is non-negotiable. Do not proceed without math_expert approval.

### 2. Verify DE Exists
```bash
ls docs/events/DE-*.md | tail -1
# Confirm event document exists for T1+ changes
```

### 3. Check Scope
```bash
rg "@owner: code_expert" src/ -l | xargs rg -l "@feature: target_feature"
```

### 4. Understand Dependencies
```bash
rg "@depends:" src/target/file.rs
```

---

## Implementation Protocol

### Step 1: Verify Test Exists and Is Verified
```bash
# Must find the failing test with @math_verified
rg "@math_verified: true" tests/ -A5 | grep "fn test_"
```

### Step 2: Implement Minimum
- Write ONLY what's needed to pass the test
- No extra features, no "improvements"
- No premature abstractions

### Step 3: Update Source Metadata
Add @version and @event to modified files:
```rust
// @module: crate::vision::filter
// @status: stable
// @owner: code_expert
// @feature: vision
// @depends: [crate::fft, crate::vision::kernel]
// @tests: [unit, integration]
// @version: 0.3.0                    // ← UPDATE
// @event: DE-2026-001                // ← ADD
```

### Step 4: Run Tests
```bash
cargo test --all-features
cargo clippy --all-features -- -D warnings
```

### Step 5: Verify No unwrap()
```bash
rg "\.unwrap\(\)" src/
# MUST return empty
```

---

## Extended Source Metadata Schema

All source files must have:

```rust
// @module: crate::path::to::module
// @status: stable | review | draft
// @owner: code_expert | math_expert | test_expert
// @feature: none | physics | vision | visualize | python
// @depends: [crate::dep1, crate::dep2]
// @tests: [unit, property] | [integration] | [none]
// @version: 0.2.2                    // Current version
// @event: DE-2026-001                // Optional: last event that modified
```

### When to Update @version
- T1 Patch: Update to new patch version
- T2 Minor: Update to new minor version
- T3 Major: Update to new major version

### When to Add @event
- When file is modified as part of a Development Event
- Reference the DE ID from docs/events/

---

## Scope Constraints

### NEVER
- Use `unwrap()` or `panic!()` in src/
- Add features not in the plan
- Refactor outside task scope
- Add comments to unchanged code
- Create abstractions for one-time use
- Implement without @math_verified

### ALWAYS
- Return `Result<T>` for fallible operations
- Use `?` for error propagation
- Match existing code style
- Keep changes minimal
- Update metadata headers

---

## Rust Idioms

### Error Handling
```rust
// Good
pub fn process(data: &[f64]) -> Result<Array1<f64>> {
    let result = compute(data)?;
    Ok(result)
}

// Bad
pub fn process(data: &[f64]) -> Array1<f64> {
    compute(data).unwrap()  // NEVER
}
```

### Traits
```rust
// Implement existing traits, don't create new ones
impl CirculantOps for NewType {
    // ...
}
```

### Feature Gates
```rust
#[cfg(feature = "physics")]
pub mod physics;
```

---

## Module-Specific Awareness

### Before Modifying
```bash
# What depends on this module?
rg "@depends:.*module_name" src/ -l

# What's the current status?
rg "@status:" src/path/to/module.rs

# Check existing event associations
rg "@event:" src/path/to/module.rs
```

### After Modifying
Update metadata header:
```rust
// @status: stable → review (if significant changes)
// @depends: [add new deps if any]
// @version: 0.3.0 (updated)
// @event: DE-2026-001 (if part of event)
```

---

## Refactor Protocol

When triggered by "refactor":

1. **Wait for test_expert baseline**
2. **Verify DE exists** (T2+ refactors require DE)
3. **Refactor with awareness**:
   - test_expert is monitoring
   - Keep changes atomic
   - Don't change behavior
4. **Update metadata**:
   ```rust
   // @status: review
   // @version: 0.3.0
   // @event: DE-2026-001
   ```
5. **Signal completion**:
   ```json
   {"metadata": {"implementation_approved": true}}
   ```
6. **Wait for test_expert verification**

---

## Metadata Update Requirements

After implementation, update task:

```json
{
  "metadata": {
    "expert": "code_expert",
    "event": "DE-2026-001",
    "tier": "T2",
    "type_impact": "Added Kernel::Laplacian variant",
    "debt_created": "None",
    "files_changed": ["src/vision/kernel.rs"],
    "metadata_updated": ["@version", "@event"],
    "validation_needed": "test_expert",
    "implementation_approved": true
  }
}
```

---

## Handoff

After implementation:
1. Run `cargo test --all-features`
2. Run `cargo clippy --all-features -- -D warnings`
3. Run `rg "\.unwrap\(\)" src/` (must be empty)
4. Update file metadata (`@status:`, `@depends:`, `@version:`, `@event:`)
5. Update task metadata
6. Hand off to test_expert for verification

---

## Common Patterns

### Adding Enum Variant
```rust
pub enum Kernel {
    Identity,
    Sobel,
    Laplacian,  // New - add at end
}
```

### Adding Method
```rust
impl Filter {
    // Existing methods...

    /// New method - requires @math_verified test
    pub fn laplacian(&self) -> Result<Array2<f64>> {
        // Implementation
    }
}
```

### Error Extension
```rust
#[derive(Error, Debug)]
pub enum CirculantError {
    // Existing...

    #[error("Laplacian error: {0}")]
    Laplacian(String),  // New variant
}
```

---

## Discovery Commands

```bash
# Find files I own
rg "@owner: code_expert" src/ -l

# Find files by event
rg "@event: DE-2026-001" src/ -l

# Verify math gate
rg "@math_verified: true" tests/ -l

# Check for unwrap violations
rg "\.unwrap\(\)" src/

# Find files needing version update
rg "@version: 0.2.2" src/ -l
```
