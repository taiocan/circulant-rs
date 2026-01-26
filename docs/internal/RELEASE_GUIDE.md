# Release Guide

Version management and release procedures.

---

## Semantic Versioning

| Version | When |
|---------|------|
| **MAJOR** (x.0.0) | Breaking API changes |
| **MINOR** (0.x.0) | New backwards-compatible features |
| **PATCH** (0.0.x) | Backwards-compatible bug fixes |

---

## Release Process

### 1. Pre-Release Verification
```bash
cd ~/projects/circulant-rs
git checkout main && git pull
git merge develop --no-ff
```

### 2. Technical Verification (MUST PASS)
```bash
cargo test --all-features
cargo test --release
cargo clippy --all-features -- -D warnings
cargo fmt -- --check
cargo doc --all-features --no-deps 2>&1 | grep -i warning
cargo bench
cargo run --example quantum_walk_1d
```

### 3. Version Bump
```bash
cargo set-version --bump patch   # 0.1.0 -> 0.1.1
cargo set-version --bump minor   # 0.1.0 -> 0.2.0
cargo set-version --bump major   # 0.1.0 -> 1.0.0
```

### 4. Changelog Update
```bash
# Manual or:
cargo changelog > CHANGELOG.md
```

### 5. Commit and Tag
```bash
git add .
git commit -m "release: v$(cargo pkgid | cut -d# -f2)"
git tag v$(cargo pkgid | cut -d# -f2)
```

### 6. Push
```bash
git push && git push --tags
```

### 7. Publish (when ready)
```bash
cargo publish
```

---

## Documentation Checklist

- [ ] README.md updated
- [ ] USER_GUIDE.md current
- [ ] API documentation complete
- [ ] Examples work with new version
- [ ] CHANGELOG.md updated
- [ ] Breaking changes documented

---

## Task Management Checklist

- [ ] All tasks in TASKS.md completed
- [ ] No blocking issues
- [ ] NOTES.md reviewed
- [ ] Knowledge transferred to appropriate docs

---

## Version Update Checklist

- [ ] **API Review:** Document breaking changes
- [ ] **Migration Guide:** For major versions
- [ ] **Deprecation:** Add `#[deprecated]` with guidance
- [ ] **Examples:** Use current API
- [ ] **MSRV:** Update if Rust version changed

---

## Backward Compatibility Rules

1. **Public API:** Breaking changes = MAJOR bump
2. **Internal API:** Breaking allowed in MINOR with deprecation
3. **Features:** Always additive, never modify existing
4. **Deprecation:** 1 MINOR version minimum before removal

---

## Post-Release

1. Merge main back to develop
2. Update version in develop for next cycle
3. Archive release notes
4. Announce release (if public)
