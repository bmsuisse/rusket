"""
rusket — Polars DataFrame Input
=================================

rusket accepts Polars DataFrames directly via Arrow zero-copy buffers.
No conversion overhead — the numpy bridge re-uses the same memory.

Requires: `uv add polars` or `pip install "rusket[polars]"`
"""

import polars as pl
import numpy as np
from rusket import fpgrowth, association_rules


# ── 1. Create a Polars DataFrame ────────────────────────────────────────────
#    (In practice you'd load this from parquet/csv)

rng = np.random.default_rng(0)
n_rows, n_cols = 20_000, 150
products = [f"product_{i:03d}" for i in range(n_cols)]

# Power-law popularity
support = np.clip(0.5 / np.arange(1, n_cols + 1, dtype=float) ** 0.5, 0.005, 0.5)
matrix = rng.random((n_rows, n_cols)) < support

df_pl = pl.DataFrame({p: matrix[:, i].tolist() for i, p in enumerate(products)})

print(f"Polars DataFrame: {df_pl.shape[0]:,} rows × {df_pl.shape[1]} columns")
print(f"Schema: {dict(list(df_pl.schema.items())[:3])} …\n")


# ── 2. fpgrowth on Polars — same API as pandas ──────────────────────────────

freq = fpgrowth(df_pl, min_support=0.05, use_colnames=True)
print(f"Frequent itemsets: {len(freq):,}")
print(freq.sort_values("support", ascending=False).head(8).to_string(index=False))
print()


# ── 3. Association rules ─────────────────────────────────────────────────────

rules = association_rules(freq, num_itemsets=n_rows, metric="lift", min_threshold=1.1)
print(f"Rules: {len(rules):,}")
print(
    rules[["antecedents", "consequents", "confidence", "lift"]]
    .sort_values("lift", ascending=False)
    .head(6)
    .to_string(index=False)
)
