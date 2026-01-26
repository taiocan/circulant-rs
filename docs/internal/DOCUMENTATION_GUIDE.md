# Documentation Guide

Document ecosystem organization and maintenance procedures.

---

## Document Types

### Public Documentation (User-Facing)
Explain **WHAT** the project is and **WHY** it matters.

| Document | Purpose | Audience |
|----------|---------|----------|
| README.md | Introduction, quick start | New users |
| WHITEPAPER.md | Mathematical foundations | Researchers |
| USER_GUIDE.md | Tutorials, examples | Developers |
| API_REFERENCE.md | Auto-generated API | Library consumers |
| BENCHMARKS.md | Performance data | Performance-focused users |

### Internal Documentation (Development)
Manage **HOW** development happens.

| Document | Purpose | Update Trigger |
|----------|---------|----------------|
| TASKS.md | Current task tracking | Real-time |
| NOTES.md | Session context | Each session |
| DEPENDENCY_GRAPH.md | Module relationships | Dependency changes |
| WORKFLOW_GUIDE.md | TDD, verification | Process changes |
| AGENT_PROTOCOL.md | Multi-agent patterns | Workflow changes |
| RELEASE_GUIDE.md | Version procedures | Process changes |

---

## Directory Structure

```
docs/
+-- DEPENDENCY_GRAPH.md      # Module relationships (root for visibility)
+-- internal/                # Development docs
|   +-- TASKS.md
|   +-- NOTES.md
|   +-- WORKFLOW_GUIDE.md
|   +-- AGENT_PROTOCOL.md
|   +-- RELEASE_GUIDE.md
|   +-- WORKTREE_GUIDE.md
|   +-- DOCUMENTATION_GUIDE.md
+-- public/                  # (future: user-facing docs)
```

---

## Update Triggers

| Event | Documents to Update |
|-------|---------------------|
| New feature | TASKS.md, NOTES.md, DEPENDENCY_GRAPH.md |
| Bug fix | TASKS.md, NOTES.md |
| Architecture change | DEPENDENCY_GRAPH.md, NOTES.md |
| Release | README.md, CHANGELOG.md |
| Session start | TASKS.md (read), NOTES.md (read) |
| Session end | TASKS.md, NOTES.md |

---

## Review Schedule

| Frequency | Documents |
|-----------|-----------|
| Each session | TASKS.md, NOTES.md |
| Weekly | DEPENDENCY_GRAPH.md |
| Per release | All public docs |
| Quarterly | CLAUDE.md, process docs |

---

## Documentation Standards

### Public Items
Every public function/struct requires:
```rust
/// One-line summary
///
/// Detailed description with mathematical basis
///
/// # Arguments
/// * `arg` - description with constraints
///
/// # Returns
/// Description of return value
///
/// # Errors
/// When each error occurs
///
/// # Example
/// ```rust
/// // working doctest
/// ```
```

### Internal Documents
- Keep concise and actionable
- Use tables for quick reference
- Include commands/code snippets
- Update immediately when relevant

---

## Future Documents (Create When Needed)

Per progressive complexity, create these when pain points emerge:

| Document | Purpose | Create When |
|----------|---------|-------------|
| DECISION_FRAMEWORK.md | Decision criteria | Repeated decision confusion |
| PERFORMANCE_MODEL.md | Expected performance | Optimization work begins |
| VALIDATION_SUITE.md | Test procedures | Complex validation needed |
| KNOWLEDGE_TRANSFER.md | Domain knowledge | Knowledge loss occurs |
| ERROR_CATALOG.md | Error documentation | Error handling expands |
| CONTRIBUTOR_GUIDE.md | Contributor onboarding | External contributors |
