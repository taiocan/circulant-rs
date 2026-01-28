# Orchestrator

Routes tasks to appropriate experts based on trigger patterns and enforces EDAD workflow.

---

## EDAD (Expert-Driven Atomic Development) Overview

EDAD combines version discipline with the expert validation chain:

1. **Development Event (DE)** - Entry point for T1+ changes
2. **Version Classification** - Determines ceremony level
3. **Expert Routing** - Tasks flow through appropriate experts
4. **Validation Gates** - @math_verified blocks implementation

---

## Tiered Ceremony

| Tier | Version Bump | Ceremony | Math Gate |
|------|--------------|----------|-----------|
| T0 Hotfix | None | Direct commit, no DE | No |
| T1 Patch | x.y.Z | Lightweight DE (justification only) | If @owner: math_expert |
| T2 Minor | x.Y.0 | Full DE + avoidance analysis | Required for numerical code |
| T3 Major | X.0.0 | Full DE + migration guide | All experts |

### Tier Classification Guide

**T0 Hotfix (no DE):**
- Typo fixes
- Comment updates
- Trivial bug fixes (< 5 lines)

**T1 Patch:**
- Bug fixes
- Small enhancements
- Documentation updates

**T2 Minor:**
- New features
- API additions
- Significant refactors

**T3 Major:**
- Breaking changes
- API removals
- Architectural changes

---

## Trigger Matrix

| Trigger Pattern | Primary Expert | Secondary | DE Required |
|-----------------|----------------|-----------|-------------|
| `/plan`, "design", "architecture" | plan_expert | - | Creates DE |
| "implement", "add feature", "code" | code_expert | - | T1+ |
| "test", "TDD", "verify" | test_expert | - | T1+ |
| "algorithm", "numerical", "math" | math_expert | test_expert | T1+ |
| "refactor" | code_expert | test_expert | T2+ |
| "fix bug" | test_expert | code_expert | T1 |
| "optimize", "performance" | math_expert | code_expert | T2+ |
| "document", "docs" | doc_expert | - | T1+ |

### doc_expert Triggers

Route to doc_expert when:
- User mentions "document", "docs", "documentation"
- After code_expert completion for T2+ changes
- When updating USER_GUIDE, ARCHITECTURE, API_TREE
- For CHANGELOG entries

---

## DE Creation Flow

For T1+ changes:

```
User Request → Tier Classification → plan_expert creates DE → Expert routing
```

### plan_expert Creates DE
1. Classify version impact (T1/T2/T3)
2. Create `docs/events/DE-YYYY-NNN.md`
3. Enumerate document tasks for each expert
4. Route to appropriate experts

---

## Discovery Commands

### Basic Queries
```bash
# Find files by owner
rg "@owner: code_expert" src/ -l

# Find files by status
rg "@status: draft" src/ -l

# Find files by feature
rg "@feature: physics" src/ -l

# Find files needing tests
rg "@tests:.*none" src/ -l
```

### Power Queries (Chained)
```bash
# Agent + feature intersection
rg "@owner: math_expert" src/ -l | xargs rg -l "@feature: physics"

# Feature dependency map
rg -l "@feature: vision" src/ | xargs rg "@depends:"

# Module + status filter
rg -l "@module: fft" src/ | xargs rg -l "@status: review"

# Files needing review for feature
rg -l "@feature: physics" src/ | xargs rg -l "@status: review"
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

---

## Hard Rules (Enforced)

1. **No unwrap in src/**: `rg "\.unwrap\(\)" src/` must return 0
2. **Tests before commit**: `cargo test --all-features`
3. **Failing test before implementation**: TDD mandatory
4. **Metadata required**: All src/ files must have headers
5. **DE required for T1+**: No ad-hoc changes above hotfix level

---

## Validation Chain

### New Feature Flow (T2)
```
plan_expert → test_expert → math_expert → code_expert → test_expert → doc_expert
     │              │              │              │              │            │
  create DE    write test    verify test    implement      verify pass    update docs
               (failing)    @math_verified
```

### The `@math_verified` Gate

**Rule:** code_expert BLOCKED until test has `@math_verified: true`

**Enforcement:**
```bash
rg "@math_verified: true" tests/ -l | xargs rg "test_name"
```
If no match, code_expert refuses and escalates.

---

## Task Handoff Protocol

Use task metadata for inter-agent communication:

```json
{
  "metadata": {
    "expert": "code_expert",
    "event": "DE-2026-001",
    "tier": "T2",
    "type_impact": "Added Kernel::Laplacian variant",
    "debt_created": "None",
    "files_changed": ["src/vision/kernel.rs"],
    "validation_needed": "test_expert",
    "tests_added": ["test_laplacian_kernel"]
  }
}
```

### Metadata Fields
- `expert`: Current owner
- `event`: Development Event ID
- `tier`: T0/T1/T2/T3
- `type_impact`: Changes to structs/traits/interfaces
- `debt_created`: Technical debt (if any)
- `files_changed`: Modified files
- `validation_needed`: Expert to verify
- `tests_added`: New tests written

---

## Verification Tags

| Tag | Set By | Meaning |
|-----|--------|---------|
| `@math_verified: true` | math_expert | Test is mathematically sound |
| `@verified_by: [expert]` | any expert | Who performed verification |
| `@properties_checked: [...]` | math_expert | Validated properties |
| `@implementation_approved: true` | code_expert | Ready for test run |

---

## Expert Roster

| Expert | Primary Responsibility | Creates |
|--------|------------------------|---------|
| plan_expert | Task breakdown, DE creation | DE documents |
| test_expert | TDD, verification | Test code |
| math_expert | Numerical validation | math.md |
| code_expert | Implementation | Source code |
| doc_expert | Human-facing docs | CHANGELOG, USER_GUIDE updates |
