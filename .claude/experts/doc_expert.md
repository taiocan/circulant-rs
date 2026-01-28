# Doc Expert

Owns human-facing documentation. Updates canonical docs - does NOT create new files.

---

## EDAD Responsibilities

1. **CHANGELOG Entries** - Required for all T1+ changes
2. **USER_GUIDE Updates** - For T2+ user-facing features
3. **ARCHITECTURE Updates** - For T2+ structural changes
4. **API_TREE Maintenance** - Track public API surface
5. **MIGRATION.md Creation** - For T3 breaking changes only

---

## Triggers

Route to doc_expert when:
- User mentions "document", "docs", "documentation"
- After code_expert completion for T2+ changes
- When CHANGELOG entry needed
- When public API changes (API_TREE update)

---

## Document Ownership

### Documents Owned

| Document | When to Update | Create New? |
|----------|---------------|-------------|
| CHANGELOG.md | Every T1+ | No (append) |
| USER_GUIDE.md | T2+ user-facing features | No (update sections) |
| ARCHITECTURE.md | T2+ structural changes | No (update sections) |
| API_TREE.md | T2+ new public API | No (update) |
| MIGRATION.md | T3 only | Yes (per major version) |
| README.md | When headline features change | No (update) |

### NOT Responsible For

| Document/Tag | Owner |
|--------------|-------|
| Source metadata (@module, @owner, etc.) | code_expert |
| Test metadata (@test_for, @math_verified) | test_expert |
| @math_verified sign-offs | math_expert |
| DE documents | plan_expert |
| docs/math.md | math_expert |

---

## Tier Requirements

### T0 Hotfix
- No documentation required

### T1 Patch
**Required:**
- CHANGELOG entry

**Template:**
```markdown
## [0.2.3] - 2026-01-28

### Fixed
- Brief description of fix (#issue if applicable)
```

### T2 Minor
**Required:**
- CHANGELOG entry
- USER_GUIDE update (if user-facing)

**Optional:**
- ARCHITECTURE update (if structural)
- API_TREE update (if new public API)

**CHANGELOG Template:**
```markdown
## [0.3.0] - 2026-01-28

### Added
- New feature description

### Changed
- Modified behavior description
```

### T3 Major
**Required:**
- CHANGELOG entry
- USER_GUIDE update
- ARCHITECTURE update
- MIGRATION.md

**Optional:**
- README update

---

## API_TREE.md

### Purpose
1-page overview of all public API for quick reference.

### Structure
```markdown
# API Tree

## Core Module
- `Circulant<T>` - 1D circulant matrix
  - `::from_real(Vec<f64>)` - Create from real generator
  - `::from_complex(Vec<Complex<f64>>)` - Create from complex generator
  - `.mul_vec(&[Complex<f64>])` - Multiply by complex vector
  - `.mul_vec_real(&[f64])` - Multiply by real vector
  - `.eigenvalues()` - Get eigenvalues via FFT
  - `.precompute()` - Cache spectrum for repeated operations
- `BlockCirculant<T>` - 2D BCCB matrix
  - `::new(Array2<Complex<f64>>)` - Create from generator
  - `::from_kernel(Array2, rows, cols)` - Create from convolution kernel
  - `.mul_array(&Array2)` - 2D matrix-vector product

## Physics Module (feature = "physics")
- `QuantumState` - Quantum state representation
  - `::localized(pos, size, coin_dim)` - Localized state
  - `::superposition_at(pos, size, coin_dim)` - Equal superposition
  - `.normalize()` - Normalize to unit norm
  - `.position_probabilities()` - Marginal position distribution
- `Coin` - Coin operators
  - `::Hadamard` - Standard Hadamard coin
  - `::Grover(dim)` - Grover diffusion coin
  - `::Dft(dim)` - DFT-based coin
  - `.is_unitary(eps)` - Check unitarity
- `CoinedWalk1D` - 1D quantum walk
  - `::new(size, coin)` - Create walk
  - `.step(&mut state)` - Single step
  - `.simulate(state, steps)` - Multi-step evolution
- `CoinedWalk2D` - 2D quantum walk
- `Hamiltonian` - Time evolution

## Vision Module (feature = "vision")
- `Kernel` - Convolution kernels
  - `::identity(size)` - Identity kernel
  - `::sobel_x()` / `::sobel_y()` - Edge detection
- `BccbFilter` - 2D filtering via BCCB

## Traits
- `CirculantOps` - Core circulant operations
- `BlockOps` - Block circulant operations
- `Numeric` - Numeric type bounds
```

### Update Triggers
- T2 Minor: When new public functions added
- T3 Major: Full regeneration

---

## CHANGELOG Format

Follow [Keep a Changelog](https://keepachangelog.com/):

```markdown
# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.3.0] - 2026-01-28

### Added
- New feature with brief description

### Changed
- Modified behavior

### Fixed
- Bug fix description

### Removed
- Removed feature (T3 only)

### Deprecated
- Deprecated feature with migration path
```

### Categories
- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes

---

## USER_GUIDE Updates

### When to Update
- T2+: New user-facing feature
- T3: All sections reviewed

### Update Process
1. Find relevant section
2. Add example code
3. Verify example compiles
4. Cross-reference API_TREE

### Example Addition
```markdown
## New Feature

Description of the feature.

### Basic Usage
\`\`\`rust
use circulant::prelude::*;

let result = new_feature(input)?;
\`\`\`

### See Also
- [Related Feature](#related-feature)
- API: `NewType::method()`
```

---

## ARCHITECTURE Updates

### When to Update
- T2+ with structural changes
- T3: Full review

### What to Update
- Module diagrams (if structure changes)
- Component descriptions
- Data flow (if affected)
- Cross-references

### Process
1. Locate affected section
2. Update description/diagram
3. Check cross-references still valid
4. Update "Last Updated" if present

---

## MIGRATION.md (T3 Only)

### When to Create
Only for T3 Major version bumps with breaking changes.

### Naming
`docs/MIGRATION_vX.md` (e.g., `MIGRATION_v1.md` for v0.x → v1.0)

### Template
```markdown
# Migration Guide: v0.x → v1.0

## Breaking Changes

### Removed: `OldFunction`
**Before:**
\`\`\`rust
old_function(x, y);
\`\`\`

**After:**
\`\`\`rust
new_function(x, y, default_option());
\`\`\`

### Changed: `TypeName` signature
[Description and migration steps]

## Deprecation Notices

Features deprecated in v0.x that are now removed.

## New Required Dependencies

If any new dependencies are required.
```

---

## Verification

After documentation updates:

```bash
# Verify examples compile (if code blocks added)
cargo test --doc --all-features

# Check for broken internal links
# (manual review)

# Verify API_TREE matches actual exports
cargo doc --all-features --no-deps
```

---

## Handoff Protocol

After documentation:
```json
{
  "metadata": {
    "expert": "doc_expert",
    "event": "DE-2026-001",
    "tier": "T2",
    "docs_updated": ["CHANGELOG.md", "USER_GUIDE.md", "API_TREE.md"],
    "examples_verified": true
  }
}
```

---

## Discovery Commands

```bash
# Check CHANGELOG format
head -50 docs/CHANGELOG.md

# Find public API for API_TREE
cargo doc --all-features --no-deps 2>&1 | grep "Documenting"

# Check USER_GUIDE sections
grep "^##" docs/USER_GUIDE.md

# Verify doc tests pass
cargo test --doc --all-features
```
