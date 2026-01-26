# Contributing to circulant-rs

**Version:** 0.1.0 | **Updated:** 2024-01-26 | **Reading time:** 5 min

> Guidelines for contributing code, documentation, and bug reports.

---

## Quick Start

```bash
# Clone and build
git clone https://github.com/yourname/circulant-rs.git
cd circulant-rs
cargo build --all-features

# Run tests
cargo test --all-features

# Check lints
cargo clippy --all-features -- -D warnings
```

---

## Development Setup

### Requirements

- Rust 1.70+ (stable)
- Optional: Python 3.10+ (for Python bindings)

### Build Commands

| Task | Command |
|------|---------|
| Build | `cargo build --all-features` |
| Test | `cargo test --all-features` |
| Lint | `cargo clippy --all-features -- -D warnings` |
| Format check | `cargo fmt -- --check` |
| Docs | `cargo doc --all-features --no-deps` |
| Bench | `cargo bench` |

---

## Code Standards

### Error Handling

- **No `unwrap()` or `panic!()` in `src/`** - use `Result<T>` with `?`
- Tests may use `unwrap()` for brevity
- Verify: `grep -r "unwrap()" src/` must return 0 matches

### Documentation

Public items require:
```rust
/// Brief summary.
///
/// # Arguments
/// * `param` - Description
///
/// # Returns
/// Description of return value
///
/// # Errors
/// When and why this can fail
///
/// # Example
/// ```
/// // Working example
/// ```
```

### Naming

- `snake_case` for functions and variables
- `CamelCase` for types
- Descriptive names: `spectral_radius` not `spec_r`

---

## TDD Workflow

1. **Write failing test** (verify it fails)
2. **Implement minimum** to pass
3. **Refactor** after green
4. **Verify**: `cargo test --all-features`

---

## Pull Request Process

### Before Submitting

```bash
# Must pass
cargo test --all-features
cargo clippy --all-features -- -D warnings
cargo fmt -- --check
```

### PR Guidelines

1. **One feature per PR** - keep changes focused
2. **Include tests** - new code needs test coverage
3. **Update docs** - if API changes, update USER_GUIDE.md
4. **Add CHANGELOG entry** - for user-visible changes

### Commit Messages

```
feat: add Grover coin operator

- Implement n-dimensional Grover diffusion
- Add tests for 2D and 3D cases
- Update USER_GUIDE with examples
```

Prefixes: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `bench:`

---

## Documentation Standards

See [DOC_POLICY.md](./internal/DOC_POLICY.md) for full policy.

Key points:
- Every claim needs evidence (benchmark, test, or proof)
- Tables > prose; examples > explanations
- No jargon without inline definition

---

## Adding New Features

1. Discuss in issue first (for major features)
2. Feature-gate in `Cargo.toml`
3. Update `docs/DEPENDENCY_GRAPH.md`
4. TDD: failing test → implement → refactor
5. Add example in `examples/`
6. Update USER_GUIDE.md

---

## Extension Points

See [ARCHITECTURE.md](./ARCHITECTURE.md) for:
- Adding new coin operators
- Implementing custom FFT backends
- Extending the physics module

---

## Getting Help

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- For questions, use GitHub Discussions

---

## Related Documents

- [ARCHITECTURE.md](./ARCHITECTURE.md) - System internals
- [USER_GUIDE.md](./USER_GUIDE.md) - API documentation
- [internal/DOC_POLICY.md](./internal/DOC_POLICY.md) - Documentation standards
