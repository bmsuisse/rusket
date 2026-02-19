"""
rusket — Sparse Pandas Input
==============================

For very sparse datasets (e.g. e-commerce with thousands of SKUs),
use pandas SparseDtype to minimise memory.  rusket passes the raw
CSR arrays straight to Rust — no densification ever happens.
"""

import pandas as pd
import numpy as np
from rusket import fpgrowth, association_rules


# ── 1. Build a sparse DataFrame ─────────────────────────────────────────────

rng = np.random.default_rng(7)
n_rows, n_cols = 30_000, 500

# Very sparse: average basket size ≈ 3 items out of 500
p_buy = 3 / n_cols
matrix = rng.random((n_rows, n_cols)) < p_buy
products = [f"sku_{i:04d}" for i in range(n_cols)]

df_dense = pd.DataFrame(matrix.astype(bool), columns=products)
df_sparse = df_dense.astype(pd.SparseDtype("bool", fill_value=False))

dense_mb = df_dense.memory_usage(deep=True).sum() / 1e6
sparse_mb = df_sparse.memory_usage(deep=True).sum() / 1e6
print(f"Dense  memory: {dense_mb:.1f} MB")
print(f"Sparse memory: {sparse_mb:.1f} MB  ({dense_mb / sparse_mb:.1f}× smaller)\n")


# ── 2. fpgrowth on sparse DF — exact same API ───────────────────────────────

freq = fpgrowth(df_sparse, min_support=0.01, use_colnames=True)
print(f"Frequent itemsets: {len(freq):,}")
print(freq.sort_values("support", ascending=False).head(5).to_string(index=False))
print()


# ── 3. Rules ─────────────────────────────────────────────────────────────────

rules = association_rules(
    freq, num_itemsets=n_rows, metric="confidence", min_threshold=0.5
)
print(f"Rules (confidence ≥ 0.5): {len(rules):,}")
if not rules.empty:
    print(
        rules[["antecedents", "consequents", "support", "confidence", "lift"]]
        .sort_values("lift", ascending=False)
        .head(5)
        .to_string(index=False)
    )
