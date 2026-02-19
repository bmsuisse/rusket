"""
rusket — Large-Scale Mining (100k+ rows)
==========================================

Demonstrates rusket on a 100k-row × 1000-item dataset —
the kind of scale that makes mlxtend run out of memory.

Prints timing and memory info at each stage.
"""

import time
import tracemalloc
import numpy as np
import pandas as pd
from rusket import fpgrowth, association_rules


def timed(label: str, fn, *args, **kwargs):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / 1e6
    print(f"  {label:<30} {elapsed:6.2f}s   peak {peak_mb:6.1f} MB")
    return result


# ── 1. Generate dataset ──────────────────────────────────────────────────────

N_ROWS, N_COLS = 100_000, 1_000
MIN_SUPPORT = 0.05

print(f"Dataset: {N_ROWS:,} rows × {N_COLS:,} items  ({N_ROWS*N_COLS/1e6:.0f} MB raw)\n")

rng = np.random.default_rng(42)
rank = np.arange(1, N_COLS + 1, dtype=float)
sup = np.clip(0.6 / rank**0.5, 0.001, 0.6)

print("Generating…")
t0 = time.perf_counter()
matrix = (rng.random((N_ROWS, N_COLS)) < sup).astype(bool)
df = pd.DataFrame(matrix, columns=[f"item_{i:04d}" for i in range(N_COLS)])
print(f"  Generated in {time.perf_counter()-t0:.2f}s\n")


# ── 2. Mine ──────────────────────────────────────────────────────────────────

print(f"Mining (min_support={MIN_SUPPORT})…")
freq = timed("fpgrowth", fpgrowth, df, min_support=MIN_SUPPORT, use_colnames=True)
print(f"  → {len(freq):,} frequent itemsets\n")


# ── 3. Rules ─────────────────────────────────────────────────────────────────

print("Association rules (lift ≥ 1.1)…")
rules = timed(
    "association_rules",
    association_rules,
    freq,
    num_itemsets=N_ROWS,
    metric="lift",
    min_threshold=1.1,
)
print(f"  → {len(rules):,} rules\n")


# ── 4. Top rules ─────────────────────────────────────────────────────────────

top = (
    rules
    .sort_values(["lift", "confidence"], ascending=False)
    .head(10)
    [["antecedents", "consequents", "support", "confidence", "lift"]]
    .reset_index(drop=True)
)
print("Top rules by lift:")
print(top.to_string())
