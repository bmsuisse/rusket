---
description: Research → Build → Test → Benchmark → Repeat — AI-driven optimisation loop for rusket
---

# Research → Build → Test → Benchmark → Repeat

This is the core AI-driven optimisation loop for rusket.
Each iteration identifies a bottleneck, implements a Rust fix, validates correctness, and measures the gain.

> **Rule:** never skip the Test step. A faster but broken library is worthless.

---

## 🔍 Step 1 — Research

Profile the current hottest path and identify what to improve next.

Run a quick benchmark first to get a baseline and see where time is spent:
```
uv run pytest tests/benchmarks/ -v --benchmark-only --benchmark-sort=mean 2>&1 | head -60
```

If you need function-level profiling, use `py-spy` on a benchmark call:
```
uv run py-spy record -o profile.svg -- python -c "
import rusket
choices = ['New York','Newark','New Orleans','Los Angeles'] * 5000
# replace with the hot function under investigation
"
```

Identify the specific Rust function or Python↔Rust boundary causing overhead.
Document the hypothesis: _"the bottleneck is X because Y"_.

---

## 🦀 Step 2 — Build

Implement the fix in Rust and rebuild the extension module.

Make your Rust changes in `src/`, then build in release dev mode:
// turbo
```
uv run maturin develop --release
```

Check that the Rust code compiles cleanly with no warnings:
// turbo
```
cargo check 2>&1
```

---

## ✅ Step 3 — Test

**All tests must pass before proceeding.** No exceptions.

// turbo
```
uv run pytest tests/ -x -q
```

If any test fails, go back to Step 2 and fix the regression before moving on.

Also run pyright to catch any type errors introduced in Python stubs or wrappers:
// turbo
```
uv run pyright
```

---

## 📊 Step 4 — Benchmark

Run the full benchmark suite and compare against the saved baseline.

```
uv run pytest tests/benchmarks/ -v --benchmark-only --benchmark-compare --benchmark-sort=mean
```

Key metrics to record:

| Metric | Target |
|--------|--------|
| Mean time | ≤ baseline mean |
| Min time | competitive |
| Throughput | improved |

Save the results for comparison in the next iteration (replace `N` with the iteration number):
```
uv run pytest tests/benchmarks/ -v --benchmark-only --benchmark-save=iteration_N
```

Interpret the results:
- **Better** → document the gain, pick the next bottleneck → Step 5.
- **Worse / no change** → revisit the approach in Step 1.

---

## 🔁 Step 5 — Repeat

Pick the next biggest gap from the benchmark output and go back to Step 1.

Useful questions to guide the next Research phase:
- Which function is still the slowest relative to the baseline?
- Is the bottleneck in the algorithm itself, the Python↔Rust boundary, or string/memory allocation?
- Can batch operations be parallelised with Rayon?
- Are we paying for unnecessary cloning or UTF-8 re-encoding on every call?

---

## Quick-reference commands

| Command | Purpose |
|---------|---------|
| `uv run maturin develop --release` | Rebuild Rust extension (optimised) |
| `cargo check` | Fast compile check without linking |
| `uv run pytest tests/ -x -q` | Full test suite, stop on first failure |
| `uv run pyright` | Type-check Python stubs |
| `uv run pytest tests/benchmarks/ -v --benchmark-only` | Full benchmark run |
| `uv run pytest tests/benchmarks/ --benchmark-compare` | Compare vs saved baseline |
| `uv run pytest tests/benchmarks/ --benchmark-save=iteration_N` | Save current results |
