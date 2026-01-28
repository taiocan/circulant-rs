# CLAUDE.md

circulant-rs: High-performance block-circulant tensor operations via FFT.

## Quick Commands

| Task | Command |
|------|---------|
| Build | `cargo build --all-features` |
| Test | `cargo test --all-features` |
| Lint | `cargo clippy --all-features -- -D warnings` |

## Hard Rules

1. **No unwrap in src/**: `rg "\.unwrap\(\)" src/` must return 0
2. **Tests before commit**: `cargo test --all-features`
3. **TDD required**: Failing test before implementation
4. **DE required for T1+**: Development Event document for non-trivial changes

## Version Bump Checklist

**MANDATORY for any version bump (T1/T2/T3):**

| File | Update Required |
|------|-----------------|
| `Cargo.toml` | Version number |
| `README.md` | Version, features, examples, roadmap |
| `docs/CHANGELOG.md` | New version entry |
| `docs/USER_GUIDE.md` | API changes, new tutorials |
| `docs/API_TREE.md` | New/changed public API |

**Additional for T2/T3:**
- `docs/math.md` - If numerical algorithms changed
- `docs/ARCHITECTURE.md` - If module structure changed

## EDAD (Expert-Driven Atomic Development)

### Overview

EDAD combines version discipline with expert validation:

1. **Development Event (DE)** - Entry point for T1+ changes
2. **Version Classification** - Determines ceremony level
3. **Expert Routing** - Tasks flow through appropriate experts
4. **Validation Gates** - @math_verified blocks implementation

### Tiered Ceremony

| Tier | Version Bump | Ceremony | Math Gate |
|------|--------------|----------|-----------|
| T0 Hotfix | None | Direct commit, no DE | No |
| T1 Patch | x.y.Z | Lightweight DE | If @owner: math_expert |
| T2 Minor | x.Y.0 | Full DE + avoidance | Required for numerical |
| T3 Major | X.0.0 | Full DE + migration | All experts |

### Validation Chain (T2)

```
plan_expert → test_expert → math_expert → code_expert → test_expert → doc_expert
```

## Expert System

See `.claude/experts/` for specialized agents:

| Expert | Responsibility | Creates/Owns |
|--------|---------------|--------------|
| `orchestrator.md` | Task routing, tier classification | Workflow |
| `plan_expert.md` | Design, DE creation | DE documents |
| `test_expert.md` | TDD, verification | Test code + metadata |
| `math_expert.md` | Numerical validation | docs/math.md, @math_verified |
| `code_expert.md` | Implementation | Source code + metadata |
| `doc_expert.md` | Human-facing docs | README, CHANGELOG, USER_GUIDE, API_TREE |

## Source Metadata

All src/ files have headers for discovery:

```rust
// @module: crate::path::to::module
// @status: stable | review | draft
// @owner: code_expert | math_expert | test_expert
// @feature: none | physics | vision | visualize | python
// @depends: [crate::dep1, crate::dep2]
// @tests: [unit, property] | [integration] | [none]
// @version: 0.2.2
// @event: DE-2026-001
```

## Test Metadata

All test files have headers:

```rust
// @test_for: [crate::module1, crate::module2]
// @math_verified: true | false
// @verified_by: math_expert
// @properties_checked: [property1, property2]
// @version: 0.2.2
// @event: DE-2026-001 | initial
```

## Discovery Commands

### Basic Queries
```bash
# Find files by owner
rg "@owner: code_expert" src/ -l

# Find files by status
rg "@status: draft" src/ -l

# Agent + feature intersection
rg "@owner: math_expert" src/ -l | xargs rg -l "@feature: physics"
```

### EDAD Queries
```bash
# Find tests awaiting math verification
rg "@math_verified: false" tests/ src/ -l

# Find unverified numerical assertions
rg -l "assert_relative_eq" tests/ | xargs rg -L "@math_verified"

# Find active Development Events
ls docs/events/DE-*.md

# Find files touched by specific event
rg "@event: DE-2026-001" src/ tests/ -l
```

## Skills

On-demand procedural workflows loaded via slash commands:

| Skill | Purpose | Used By |
|-------|---------|---------|
| `/discover` | Codebase discovery queries | All experts |
| `/create-de` | Development Event creation | plan_expert |
| `/tdd-cycle` | TDD workflow and test patterns | test_expert |
| `/verify-math` | Mathematical verification | math_expert |
| `/implement` | Implementation workflow | code_expert |
| `/handoff` | Expert handoff protocol | All experts |

Skills are located in `.claude/skills/*/SKILL.md`.

## Reference Docs

| Doc | Purpose | Owner |
|-----|---------|-------|
| README.md | Project overview, quick start | doc_expert |
| docs/ARCHITECTURE.md | System architecture | doc_expert |
| docs/USER_GUIDE.md | API tutorials | doc_expert |
| docs/API_TREE.md | Public API overview | doc_expert |
| docs/math.md | Mathematical foundations | math_expert |
| docs/events/ | Development Events | plan_expert |
| docs/templates/ | DE template | plan_expert |
| docs/archive/ | Deprecated documentation | - |
