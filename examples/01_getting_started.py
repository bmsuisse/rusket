"""
rusket — Getting Started
=========================

The simplest possible example: mine frequent itemsets from a
one-hot encoded pandas DataFrame, then generate association rules.
"""

import pandas as pd

from rusket import association_rules, fpgrowth

# A small market-basket dataset (5 transactions, 5 items)
data = {
    "bread": [1, 1, 0, 1, 1],
    "butter": [1, 0, 1, 1, 0],
    "milk": [1, 1, 1, 0, 1],
    "eggs": [0, 1, 1, 0, 1],
    "cheese": [0, 0, 1, 0, 0],
}
df = pd.DataFrame(data).astype(bool)

print("Input DataFrame:")
print(df.to_string())
print()

# ── 1. Mine frequent itemsets ───────────────────────────────────────────────
freq = fpgrowth(df, min_support=0.4, use_colnames=True)

print("Frequent itemsets (min_support=0.4):")
print(freq.sort_values("support", ascending=False).to_string(index=False))
print()

# ── 2. Generate association rules ───────────────────────────────────────────
rules = association_rules(
    freq,
    num_itemsets=len(df),
    metric="confidence",
    min_threshold=0.6,
)

print("Association rules (confidence ≥ 0.6):")
cols = ["antecedents", "consequents", "support", "confidence", "lift"]
print(rules[cols].sort_values("lift", ascending=False).to_string(index=False))
