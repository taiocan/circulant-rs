# Agent Protocol

Multi-agent coordination patterns for AI-assisted development.

---

## Default Model: 3-Agent System

| Agent | Role | Inputs | Outputs |
|-------|------|--------|---------|
| **Planner** | Design, task breakdown | Requirements | TASKS.md, NOTES.md |
| **Implementer** | Code per plan | Task list | Source changes |
| **Tester** | Verify, validate | Code changes | Test results |

### Extended Model (Releases Only)

| Agent | Role | When Used |
|-------|------|-----------|
| **Documenter** | Update public docs | Major features, releases |
| **Reviewer** | Code review, quality gates | Pre-release, API changes |

---

## Agent Responsibilities

### Planner Agent

**Triggers:** `/plan`, complex feature requests, architecture changes

**Must Check:**
- `docs/ARCHITECTURE.md` for design patterns
- `DEPENDENCY_GRAPH.md` for module relationships
- DRY principle compliance
- TDD compliance

**Outputs:**
- Task breakdown in `TASKS.md`
- Design decisions in `NOTES.md`
- Architecture updates if needed

### Implementer Agent

**Triggers:** Task assignment from Planner

**Must Follow:**
- Naming conventions (snake_case functions, CamelCase types)
- Additive changes preferred
- Minimum scope principle
- Error handling standards (no unwrap in src/)

**Outputs:**
- Source code changes
- Inline documentation
- Unit tests

### Tester Agent

**Triggers:** Implementation completion

**Must Run:**
```bash
cargo test --all-features
cargo clippy --all-features -- -D warnings
```

**Block On:**
- Test failures
- Clippy warnings
- Missing documentation

**Outputs:**
- Test results
- Verification status
- Blockers if any

---

## Agent Workflow

```
User Request
     |
     v
[Planner] ---> TASKS.md, NOTES.md
     |
     v
[Implementer] ---> Source changes
     |
     v
[Tester] ---> Verification
     |
     +---> Pass: Task complete
     |
     +---> Fail: Return to Implementer
```

---

## Handoff Protocol

### Planner -> Implementer
1. Tasks documented in TASKS.md
2. Design decisions in NOTES.md
3. Dependencies identified via DEPENDENCY_GRAPH.md
4. Clear acceptance criteria per task

### Implementer -> Tester
1. Code committed (or staged)
2. Task marked as "ready for test"
3. Relevant test commands specified
4. Expected behavior documented

### Tester -> Planner (on failure)
1. Failure details documented
2. Root cause analysis if clear
3. Task returned to pending
4. Blocker documented in TASKS.md

---

## Persistence Protocol

### Before Session End
1. Update `TASKS.md` with current status
2. Document key decisions in `NOTES.md`
3. Use `/compact "Preserve: <context>"` if continuing

### Between Sessions
- TASKS.md: Source of truth for task status
- NOTES.md: Context for decisions made
- DEPENDENCY_GRAPH.md: Module relationships

### Context Restoration
1. Read TASKS.md for current work
2. Read NOTES.md for recent decisions
3. Check DEPENDENCY_GRAPH.md before loading files

---

## Communication Patterns

### Clear Handoffs
```
[Planner] Task 1 ready for implementation:
- Description: Add hadamard coin operator
- Files: src/physics/coin.rs
- Tests: cargo test coin
- Acceptance: Existing tests pass + new hadamard tests
```

### Status Updates
```
[Implementer] Task 1 complete:
- Added HadamardCoin struct
- Added tests in src/physics/coin.rs
- Ready for verification
```

### Blocking Issues
```
[Tester] Task 1 blocked:
- Failure: test_hadamard_normalization failed
- Cause: Normalization factor incorrect (should be 1/sqrt(2))
- Action: Return to Implementer
```
