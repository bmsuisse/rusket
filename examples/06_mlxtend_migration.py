"""
rusket — mlxtend migration guide
===================================

If you are migrating from mlxtend, this file shows the exact
API equivalents side-by-side.  rusket is a drop-in replacement:
just change the import line.
"""

import pandas as pd
from rusket import fpgrowth, association_rules

# ── mlxtend code (before) ────────────────────────────────────────────────────
#
#   from mlxtend.frequent_patterns import fpgrowth, association_rules
#   freq  = fpgrowth(df, min_support=0.05, use_colnames=True)
#   rules = association_rules(freq, metric="lift", min_threshold=1.2)
#                            # ^ mlxtend does NOT require num_itemsets
#
# ── rusket (after) ──────────────────────────────────────────────────────────
#
#   from rusket import fpgrowth, association_rules                  # ← only change
#   freq  = fpgrowth(df, min_support=0.05, use_colnames=True)        # identical
#   rules = association_rules(freq, num_itemsets=len(df),            # ← add this
#                             metric="lift", min_threshold=1.2)
#
# The only required API difference is `num_itemsets` (total transaction count).
# This makes support calculation explicit and avoids a hidden pandas join.

# ── Concrete example ─────────────────────────────────────────────────────────

data = {
    "milk": [1, 1, 0, 1, 1],
    "bread": [1, 0, 1, 1, 0],
    "butter": [1, 1, 1, 0, 1],
    "eggs": [0, 1, 0, 1, 0],
    "cheese": [0, 0, 1, 0, 0],
}
df = pd.DataFrame(data).astype(bool)

# fpgrowth — identical to mlxtend
freq = fpgrowth(df, min_support=0.4, use_colnames=True)

# association_rules — add num_itemsets=len(df)
rules = association_rules(
    freq,
    num_itemsets=len(df),  # ← the one extra arg vs mlxtend
    metric="confidence",
    min_threshold=0.6,
)

print("Frequent itemsets:")
print(freq.to_string(index=False))
print()
print("Rules:")
print(
    rules[["antecedents", "consequents", "confidence", "lift"]].to_string(index=False)
)

# ── Gotchas ───────────────────────────────────────────────────────────────────
#
# 1. Input must be bool or 0/1 integers.
#    rusket warns if you pass floats and converts them.
#
# 2. Polars is supported natively:
#       import polars as pl
#       freq = fpgrowth(pl.from_pandas(df), min_support=0.4)
#
# 3. Sparse pandas DataFrames work too — and use much less RAM:
#       df_sparse = df.astype(pd.SparseDtype("bool", fill_value=False))
#       freq = fpgrowth(df_sparse, min_support=0.4)
