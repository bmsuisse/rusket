# rusket Agent Config

## Branch Strategy

This project works **directly on `main`** — no feature branches, no PRs required.

- `git add -A && git commit -m "..." && git push origin main` is the standard flow.
- Only create a branch if the user explicitly asks for one.

## Quick Reference

| Task | Command |
|------|---------|
| Rust check | `cargo check` |
| Build | `uv run maturin develop --release` |
| Test | `uv run pytest tests/ -x -q` |
| Type check | `uv run pyright` |
