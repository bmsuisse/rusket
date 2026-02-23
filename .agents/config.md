# rusket Agent Config

## Branch Strategy

This project works **directly on `main`** — no feature branches, no PRs required.

- `git add -A && git commit -m "..." && git push origin main` is the standard flow.
- We **only** work on the `main` branch. Do not create feature branches under any circumstances.

## Quick Reference

| Task | Command |
|------|---------|
| Rust check | `cargo check` |
| Build | `uv run maturin develop --release` |
| Test | `uv run pytest tests/ -x -q` |
| Type check | `uv run pyright` |
