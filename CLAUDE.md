# CLAUDE.md - Development Standards

circulant-rs: High-performance block-circulant matrix operations via FFT.

---

## Quick Reference

| Task | Command |
|------|---------|
| Build | `cargo build --all-features` |
| Test | `cargo test --all-features` |
| Lint | `cargo clippy --all-features -- -D warnings` |
| Format | `cargo fmt -- --check` |
| Bench | `cargo bench` |
| Docs | `cargo doc --all-features --no-deps` |
| Example | `cargo run --example quantum_walk_1d` |

---

## Core Standards (Non-Negotiable)

### Error Handling
- **NO** `unwrap()`/`panic!()` in `src/` - use `Result<T>` + `?`
- **Verify:** `grep -r "unwrap()" src/` must return 0 matches
- Tests may use `unwrap()` for brevity

### Documentation
Public items require: `/// summary`, `# Arguments`, `# Returns`, `# Errors`, `# Example`

### Naming
- `snake_case` functions/variables
- `CamelCase` types
- Descriptive names: `spectral_radius` not `spec_r`

---

## Workflow Rules

### TDD Protocol
1. Write failing test (verify it fails)
2. Implement minimum to pass
3. Refactor after green

### Change Rules
- **Additive:** Prefer adding over modifying
- **Minimum scope:** Change ONLY what's necessary
- **Context preservation:** Do not reformat or modify sections outside the task's scope
- **Verify:** `cargo test` after EACH change

See `docs/internal/WORKFLOW_GUIDE.md` for detailed procedures.

---

## Agent Model (Default: 3-Agent)

| Agent | Role | Output |
|-------|------|--------|
| Planner | Design, task breakdown | TASKS.md, NOTES.md |
| Implementer | Code per plan | Source changes |
| Tester | Verify, validate | Test results |

Extended model (Documenter, Reviewer) for releases only.
See `docs/internal/AGENT_PROTOCOL.md` for details.

---

## Context Management

### Budget
- CLAUDE.md: ~1,500 tokens (this file)
- Session target: <40k before `/compact`
- Monitor every 5 interactions

### Before `/clear` or `/compact`
1. Update `docs/internal/NOTES.md`
2. Use: `/compact "Preserve: <context>"`

### File Loading
- Before loading >3 files: check `docs/DEPENDENCY_GRAPH.md`
- Use narrow refs: `@src/physics/coin.rs` not `@src/`

---

## Document References

| Document | Purpose | Location |
|----------|---------|----------|
| TASKS.md | Current task tracking | docs/internal/ |
| NOTES.md | Session context | docs/internal/ |
| DEPENDENCY_GRAPH.md | Module relationships | docs/ |
| ARCHITECTURE.md | System architecture | docs/ |
| USER_GUIDE.md | API tutorials, examples | docs/ |
| WHITEPAPER.md | Mathematical foundations | docs/ |
| WORKFLOW_GUIDE.md | TDD, change management | docs/internal/ |
| AGENT_PROTOCOL.md | Multi-agent coordination | docs/internal/ |
| RELEASE_GUIDE.md | Version procedures | docs/internal/ |
| WORKTREE_GUIDE.md | Git worktrees (optional) | docs/internal/ |
| DOCUMENTATION_GUIDE.md | Doc ecosystem | docs/internal/ |

---

## Adding New Features

1. Update `docs/internal/TASKS.md` with task breakdown
2. Feature-gate in `Cargo.toml` and `lib.rs`
3. Update `docs/DEPENDENCY_GRAPH.md`
4. TDD: failing test -> implement -> refactor
5. Add example in `examples/`
6. Verify: `cargo test --all-features && cargo clippy --all-features -- -D warnings`

### Error Extension Pattern
```rust
#[derive(Error, Debug)]
pub enum CirculantError {
    #[error("New error: {0}")]
    NewVariant(String),
    // ... existing variants
}
```

---

## Verification Checklist

### Per Task
```bash
cargo test --all-features
cargo clippy --all-features -- -D warnings
```

### Pre-Release
See `docs/internal/RELEASE_GUIDE.md` for full checklist.
