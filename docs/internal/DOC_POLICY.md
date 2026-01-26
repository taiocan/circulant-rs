# Documentation Policy

**Purpose:** Standards for all public documentation in circulant-rs.

---

## The 5C Framework

| Principle | Definition | Implementation |
|-----------|------------|----------------|
| **Concise** | Maximum info per token | Tables > prose; examples > explanations |
| **Clear** | 15-year-old test | No jargon without definition; one concept per paragraph |
| **Credible** | Every claim verifiable | Link to benchmark, test, or mathematical proof |
| **Complete** | No dead ends | Every concept has example; every question has answer |
| **Current** | Version-aligned | Update triggers defined; staleness visible |

---

## Trust Triangle

Every significant claim follows:

```
                    CLAIM
                   /     \
                  /       \
                 /         \
           EVIDENCE ─────── VERIFICATION
```

1. **Claim**: "676× faster than naive O(N²)"
2. **Evidence**: Benchmark table with methodology link
3. **Verification**: `cargo bench` command reader can run

---

## Document Standards

### Universal Structure

Every public document includes:

```markdown
# Document Title

**Version:** 0.1.0 | **Updated:** 2024-XX-XX | **Reading time:** X min

> One-sentence summary of what this document covers.

## Quick Navigation (if >500 lines)

---

[Content sections]

---

## Related Documents
## Verification (for performance docs)
```

### Content Density Rules

| Rule | Rationale |
|------|-----------|
| Tables over prose | 3× more scannable |
| Code before explanation | Show, don't tell |
| One concept per heading | Prevents overwhelm |
| Max 3 levels of nesting | H1 → H2 → H3 only |
| Link claims to evidence | Trust building |

### Accessibility Rules

| Rule | Implementation |
|------|----------------|
| Define jargon inline | "FFT (Fast Fourier Transform) - an algorithm that..." |
| Provide analogies | "Like a recipe that uses only the first row" |
| Use progressive disclosure | Simple example → Advanced options |
| Visual where possible | ASCII diagrams, tables |

---

## Change Management

### Core Rules (Aligned with CLAUDE.md)

- **Additive:** Prefer adding sections over modifying existing
- **Minimum scope:** Update ONLY sections affected by the change
- **Version sync:** Docs update with each release
- **Deprecation over deletion:** Mark outdated content before removing

### Update Triggers

| Event | Documents to Update | Owner |
|-------|---------------------|-------|
| New feature | USER_GUIDE, ARCHITECTURE (if internal), CHANGELOG | Implementer |
| Bug fix | CHANGELOG only (unless user-facing behavior changes) | Implementer |
| Performance improvement | BENCHMARKS, README (if headline number changes), CHANGELOG | Tester |
| API change | USER_GUIDE, CHANGELOG, migration notes | Implementer |
| Release | All version headers, CHANGELOG | Documenter |

### Version Header Protocol

Every public doc includes:
```markdown
**Version:** 0.1.0 | **Updated:** YYYY-MM-DD | **Reading time:** X min
```

Update rules:
- **Version**: Matches library version
- **Updated**: Date of last substantive change (not formatting)
- **Reading time**: Recalculate if content changes significantly

---

## Public Document Set

| Document | Purpose | Audience | Target Tokens |
|----------|---------|----------|---------------|
| README.md | First impression + quick start | Everyone | 3,000 |
| OVERVIEW.md | Value prop + use cases | Decision-makers | 5,000 |
| USER_GUIDE.md | API tutorial | Developers | 8,000 |
| ARCHITECTURE.md | Internals + extension | Contributors | 10,000 |
| WHITEPAPER.md | Math foundations | Researchers | 2,000 |
| BENCHMARKS.md | Performance evidence | Validators | 3,000 |
| CHANGELOG.md | Version history | All | 1,000 |
| CONTRIBUTING.md | How to contribute | Contributors | 1,500 |

---

## Reader Journeys (5-Minute Paths)

```
"Is this useful for me?" (3 min)
└── README.md → OVERVIEW.md "Use Cases" section

"Show me the proof" (5 min)
└── README.md benchmarks → BENCHMARKS.md methodology

"Let me try it" (10 min)
└── README.md quick start → USER_GUIDE.md first example

"How does it work?" (15 min)
└── WHITEPAPER.md → ARCHITECTURE.md "Core Implementation"

"I want to contribute" (5 min)
└── CONTRIBUTING.md → ARCHITECTURE.md extension points
```

---

## What to Avoid

| Anti-Pattern | Why Bad | Instead |
|--------------|---------|---------|
| Superlatives without proof | Appears promotional | Specific numbers with links |
| Hiding limitations | Erodes trust when discovered | Upfront "When NOT to use" |
| Jargon-heavy | Excludes beginners | Inline definitions |
| Wall of text | Skipped content | Tables, code, bullets |
| Broken promises | Kills adoption | Only document what works |

---

## Verification Checklist

Before merging documentation changes:

```bash
# Check all public docs have version headers
grep -l "^\*\*Version:" README.md docs/*.md

# Verify cross-references work
grep -rn "BENCHMARK_PROPOSAL" docs/ README.md  # Should return nothing

# Ensure new files exist where expected
ls docs/CHANGELOG.md docs/CONTRIBUTING.md
```
