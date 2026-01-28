# Git Worktree Guide

**Status:** OPTIONAL - Use for complex parallel feature development.

Git worktrees allow isolated development of multiple features simultaneously without branch switching.

---

## When to Use Worktrees

**Use worktrees when:**
- Developing 2+ features in parallel
- Feature requires significant context switching
- Testing compatibility between features
- Long-running feature development

**Skip worktrees when:**
- Single feature development
- Quick bug fixes
- Documentation updates

---

## Directory Structure

```
~/projects/
+-- circulant-rs/              # MAIN WORKTREE (main/develop)
|   +-- src/
|   +-- docs/
|   +-- .git/
+-- worktrees/
|   +-- features/              # FEATURE WORKTREES
|       +-- quantum-walks-2d/
|       +-- vision-module/
+-- archives/                  # Completed worktrees
```

---

## Commands Cheat Sheet

| Operation | Command |
|-----------|---------|
| Create | `git worktree add -b feature/<name> ../worktrees/features/<name> develop` |
| List | `git worktree list` |
| Remove | `git worktree remove ../worktrees/features/<name>` |
| Update | `git fetch && git merge origin/develop` |

---

## Feature Start Protocol

```bash
# From main worktree
cd ~/projects/circulant-rs
git checkout develop && git pull

# Create worktree
git worktree add -b feature/my-feature ../worktrees/features/my-feature develop
cd ../worktrees/features/my-feature

# Initialize feature docs
echo "# My Feature" > NOTES.md
echo "- [ ] Task 1" > TASKS.md
git add . && git commit -m "feat: start my-feature"
```

---

## Development Cycle

```bash
cd ~/projects/worktrees/features/my-feature

# Sync with develop
git fetch && git merge origin/develop

# TDD cycle
# 1. Write failing test
# 2. Implement
# 3. Verify: cargo test

# Regular commits
git push -u origin feature/my-feature
```

---

## Completion Protocol

```bash
# Final verification
cargo test --all-features --release
cargo clippy --all-features -- -D warnings

# Return to main
cd ~/projects/circulant-rs

# Merge
git checkout develop
git merge feature/my-feature --no-ff

# Cleanup
git worktree remove ../worktrees/features/my-feature
git branch -d feature/my-feature
```

---

## Best Practices

1. **Naming:** `feature/<name>`, `fix/<issue>`, `docs/<topic>`
2. **Location:** Always in `../worktrees/features/`
3. **Backup:** Push branches regularly
4. **Cleanup:** Remove immediately after merge
5. **Docs:** Keep worktree-specific NOTES.md and TASKS.md
