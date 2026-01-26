# Workflow Guide

Detailed procedures for TDD, change management, and verification workflows.

---

## Task Management Rules

1. **Initialize:** Start every feature with `/plan`
2. **Atomicity:** Each task = one logical change (e.g., "Add error variant" = 1 task)
3. **Checkpoints:** Use `/tasks` to track progress, sync with `TASKS.md`
4. **Completion:** Task only marked "Complete" after Tester confirms `cargo test` passes
5. **Documentation:** Update relevant docs as part of task completion

---

## TDD Sequence

### The Cycle
1. **Write failing test** (verify it fails)
2. **Update `VALIDATION_SUITE.md`** if new test category
3. **Minimal implementation** to pass test
4. **Refactor** only after green tests

### Test Organization
- Unit tests: In same file as implementation (`#[cfg(test)]`)
- Integration tests: In `tests/` directory
- Property tests: Use `proptest` crate when applicable
- Benchmarks: In `benches/` directory

### Test Naming
```rust
#[test]
fn test_<function>_<scenario>_<expected>() {
    // Given: setup
    // When: action
    // Then: assert
}
```

---

## Change Management

### Principles
- **Minimum Scope:** Change ONLY what's necessary
- **Additive Policy:** Prefer adding over modifying existing logic
- **Verification:** Run `cargo test` after EVERY task completion
- **Preserve:** Don't refactor working code without explicit instruction
- **Document:** Update `DECISION_LOG.md` for significant changes

### Change Categories

| Type | Scope | Verification |
|------|-------|--------------|
| Bug fix | Single file | Unit tests + affected integration |
| Feature | Module | Full module tests + examples |
| Refactor | Across modules | Full test suite + benchmarks |
| API change | Public interface | All tests + doc examples |

---

## Verification Steps

### After Each Task
```bash
cargo test --all-features
cargo test --release  # numerical stability
cargo clippy --all-features -- -D warnings
```

### After Each Feature
```bash
cargo test --all-features
cargo test --release
cargo clippy --all-features -- -D warnings
cargo doc --all-features --no-deps  # check doc warnings
cargo run --example <relevant_example>
```

### Pre-Commit
```bash
cargo fmt -- --check
cargo clippy --all-features -- -D warnings
cargo test --all-features
```

### Pre-Release
See `RELEASE_GUIDE.md` for full checklist.

---

## Commit Protocol

### Message Format
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code change without feature/fix
- `test`: Adding/modifying tests
- `bench`: Benchmark changes
- `chore`: Maintenance tasks

### Examples
```
feat(physics): add 2D quantum walk implementation
fix(fft): correct normalization for inverse transform
docs(readme): update installation instructions
refactor(core): extract common circulant operations
```

---

## Development Philosophy

### Performance First
- **Benchmark changes:** `cargo bench` with reference to `PERFORMANCE_MODEL.md`
- **Precompute** repeated operations
- **Avoid allocations** in hot paths
- **`unsafe` only** with benchmarks proving necessity

### Correctness
- **Validate against naive O(N^2)** implementations in tests
- **Property-based testing** for invariants
- **Numerical stability tests** in release mode
- **Cross-verification** with reference implementations

### Safety
- **Safe abstractions** preferred
- **Zero-warning policy** for Clippy
- **All examples must compile and run**
- **Error handling** documented in `ERROR_CATALOG.md`

---

## Quality Gates

| Gate | Requirements |
|------|--------------|
| Commit | Tests pass + Clippy clean |
| Feature | Tests + Examples + Docs |
| Release | All gates + Performance verified |
