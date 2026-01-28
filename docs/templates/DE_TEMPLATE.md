# DE-YYYY-NNN: [Title]

**Version Bump:** patch | minor | major
**Impact Tier:** T0 | T1 | T2 | T3
**Status:** draft | in-progress | completed | abandoned

---

## Justification

[Why is this change needed? What problem does it solve?]

---

## Avoidance Analysis (T2+ only)

[What alternatives were considered and rejected?]

| Alternative | Rejected Because |
|-------------|------------------|
| [Option 1] | [Reason] |
| [Option 2] | [Reason] |

---

## Scope

### Files to Modify
- [ ] `path/to/file.rs` - [description of change]

### Files to Create
- [ ] `path/to/new/file.rs` - [purpose]

---

## Document Tasks

### plan_expert
- [x] Create docs/events/DE-YYYY-NNN.md

### test_expert
- [ ] Create/update tests with metadata header
- [ ] Add @math_verified: false to new tests

### math_expert
- [ ] Review tests
- [ ] Set @math_verified: true
- [ ] Add @properties_checked

### code_expert
- [ ] Implement changes
- [ ] Update @version, @event in source metadata

### doc_expert
- [ ] Update docs/USER_GUIDE.md (if T2+)
- [ ] Add CHANGELOG entry
- [ ] Update docs/ARCHITECTURE.md (if structural)
- [ ] Update docs/API_TREE.md (if new public API)

---

## Verification Gates

- [ ] @math_verified: true (for numerical code)
- [ ] `cargo test --all-features` passes
- [ ] `cargo clippy --all-features -- -D warnings` passes
- [ ] `rg "\.unwrap\(\)" src/` returns 0
- [ ] Version bump matches tier

---

## Notes

[Additional context, decisions, or implementation notes]
