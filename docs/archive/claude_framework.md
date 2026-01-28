# Unified Claude Code Agent Framework

## **1. System Architecture & Governance**

### **1.1 Three-Tier Hierarchical Structure**
- **L0: Orchestrator (`CLAUDE.md`)**
  - Never writes code directly
  - Classifies user intent → delegates to appropriate expert
  - Enforces discipline: prevents reading irrelevant files
  - Maintains metadata mandate enforcement

- **L1: Specialized Experts (`.claude/xxxx_expert.md`)**
  - Each has distinct "personality" and knowledge domain
  - Bootstrap procedure: must query codebase metadata via `rg` commands
  - Strict operational constraints per expert type

- **L2: Shared Communication Buffer (`docs/internal/sync.md`)**
  - Single source of truth for inter-agent communication
  - Structured markdown with explicit tags
  - Must be cleaned regularly to prevent bloat

### **1.2 Metadata Mandate (The "Searchable Database")**
**Rule:** Every source file must include machine-readable metadata header:
```rust
// @status: [Draft | Review | Stable]
// @module: [module_name]
// @feature: [feature_name]
// @owner: [expert_name]
// @depends_on: [file_path]
```

**Enforcement:** Orchestrator rejects any plan creating files without metadata. Experts must verify metadata consistency in their working set.

---

## **2. Orchestrator Protocol (CLAUDE.md)**

### **2.1 Delegation Workflow**
1. **Classification:** Match user request to expert type via intent analysis
2. **Context Discovery:** Generate precise `rg` queries for sub-agent
   - Example: `rg -l "@module: fft_core" | xargs rg -l "#fft_v2"`
3. **Isolation:** Explicitly forbid reading outside discovered scope
   - Exception: files listed in `@depends_on`

### **2.2 Hard Rules**
- **TDD Enforcement:** No code_expert task begins until test_expert commits failing test suite verified by math_expert
- **Scope Verification:** Each expert activation includes verification of their working set via metadata queries
- **Self-Correction:** Every expert must perform final audit against CLAUDE.md before task completion

---

## **3. Expert Specialist Framework**

### **3.1 Bootstrap Procedure (Mandatory First Actions)**
Upon activation, each expert must:
1. Run metadata queries to identify working set
2. Verify `@depends_on` relationships form complete dependency graph
3. Check for stale or incomplete metadata in assigned files

### **3.2 Expert-Specific Constraints**
- **`plan_expert`**: 
  - Output must be `TASK_LIST` with dependency graph
  - Must generate **Task Context Manifest** (specific `rg` commands for next agent)
  - Priority: safety (never break existing functionality)

- **`math_expert`**:
  - "Bible": complex numbers and linear algebra
  - Forbidden from using `f32` (must enforce `Complex<f64>`)
  - Responsible for algorithm verification against mathematical proofs

- **`code_expert`**:
  - Master of Rust traits, generics, and memory efficiency
  - Must modularize any file exceeding 500 lines
  - Must adhere to `ndarray 0.16` (or specified) constraints

- **`test_expert`**:
  - Acts as "adversary" seeking edge cases
  - Must use `approx` crate for floating-point comparisons
  - Creates failing test suites before implementation begins

- **`docs_expert`**:
  - Translates technical debt into usable documentation
  - Must update `dependency_graph.md` when new `@depends_on` links are created
  - Uses `sync.md` as primary source (never reads raw code directly)

### **3.3 Communication Protocol**
**Summary-First Principle:** Each expert must write structured summary to `sync.md` before exiting:

```markdown
<SYNC_ID: task_name>
- CHANGE: brief description of changes
- TYPE_IMPACT: modifications to structs/traits/interfaces
- DEBT_CREATED: temporary hacks or technical debt
- METADATA_UPDATED: changes to file headers
- NEXT_STEPS: required follow-up actions
</SYNC_ID>
```

**Inter-Agent Tags:** Use structured tags for quick scanning:
- `<ARCH_UPDATE>` - Architectural changes requiring review
- `<DEBT_ALERT>` - Technical debt requiring attention
- `<VALIDATION_REQ>` - Requests verification from another expert
- `<MERGE_READY>` - Task complete and verified

---

## **4. Verification & Quality Assurance**

### **4.1 The Hard Rules**
- **Scope Isolation:** Every expert prompt begins with: 
  *"Acting as [Expert_Name], using ONLY resources from [Path_to_Expert_MD] and files discovered via: [rg_command]..."*

- **Dependency Validation:** Experts must verify all `@depends_on` references exist and are up-to-date

- **Output Format Enforcement:**
  - Rust code: mandatory `cargo fmt` and `cargo clippy` before completion
  - Documentation: must follow project templates
  - Tests: must include edge cases and property-based testing where applicable

### **4.2 Continuous Verification**
1. **Pre-Task:** Orchestrator verifies metadata completeness
2. **Mid-Task:** Experts must check for consistency with CLAUDE.md rules
3. **Post-Task:** Orchestrator validates sync.md entry format and completeness

---

## **5. Proposed Improvements & Next Steps**

### **5.1 Immediate Implementation Priorities**
1. **Create `CLAUDE.md`** with the delegation protocol and metadata enforcement
2. **Create `.claude/plan_expert.md`** with Task Context Manifest generation logic
3. **Establish `docs/internal/sync.md`** with template structure
4. **Add metadata headers** to existing critical files

### **5.2 Automation Enhancements**
- **Script metadata validation** (pre-commit hook)
- **Generate expert-specific `rg` query templates** based on common patterns
- **Create sync.md cleanup automation** (removing merged task entries)

### **5.3 Scalability Considerations**
- **Expert composition:** Allow creating composite experts for complex tasks
- **Progressive disclosure:** Experts should initially see minimal context, expanding as needed
- **Audit trail:** Maintain immutable log of all sync.md changes for debugging

### **5.4 Quality Metrics**
- Track token efficiency (work vs. noise ratio)
- Measure metadata coverage percentage
- Monitor task completion time with/without framework

---

## **Ready to Generate?**

**Shall I now generate the complete `CLAUDE.md` and `.claude/plan_expert.md` files with these unified guidelines?** This will establish the operational framework for your `circulant-rs` project with 99% reliability through metadata-driven context management.