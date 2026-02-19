"""
rusket — Realistic Market-Basket Analysis with Faker
======================================================

Generates a synthetic e-commerce dataset using Faker
(realistic product names with power-law popularity),
mines frequent itemsets, generates rules, and prints
the "Top 10 must-stock cross-sell opportunities" table.
"""

import numpy as np
import pandas as pd
from faker import Faker
from rusket import fpgrowth, association_rules


# ── 1. Generate dataset with Faker ──────────────────────────────────────────

def make_basket_df(
    n_transactions: int = 50_000,
    n_products: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic market-basket: power-law product popularity via Faker names."""
    fake = Faker()
    Faker.seed(seed)
    rng = np.random.default_rng(seed)

    # Unique product names
    seen: set[str] = set()
    products: list[str] = []
    while len(products) < n_products:
        name = f"{fake.word().capitalize()} {fake.word()}"
        if name not in seen:
            seen.add(name)
            products.append(name)

    # Zipf-like support: top products appear in ~60% of baskets, tail in ~0.1%
    rank = np.arange(1, n_products + 1, dtype=float)
    support = np.clip(0.6 / rank**0.6, 0.001, 0.6)
    matrix = rng.random((n_transactions, n_products)) < support

    return pd.DataFrame(matrix.astype(bool), columns=products)


print("Generating 50k transaction dataset with Faker…")
df = make_basket_df()
print(f"  {len(df):,} transactions × {len(df.columns)} products\n")


# ── 2. Mine frequent itemsets ───────────────────────────────────────────────

print("Mining frequent itemsets (min_support=0.05)…")
freq = fpgrowth(df, min_support=0.05, use_colnames=True)
print(f"  Found {len(freq):,} frequent itemsets\n")


# ── 3. Generate association rules ───────────────────────────────────────────

print("Generating association rules (lift ≥ 1.2)…")
rules = association_rules(
    freq,
    num_itemsets=len(df),
    metric="lift",
    min_threshold=1.2,
)
print(f"  Found {len(rules):,} rules\n")


# ── 4. Top cross-sell opportunities ─────────────────────────────────────────

top = (
    rules
    .query("confidence >= 0.5")
    .sort_values(["lift", "confidence"], ascending=False)
    .head(10)
    [["antecedents", "consequents", "support", "confidence", "lift"]]
    .reset_index(drop=True)
)

print("🛒 Top 10 cross-sell opportunities:")
print(top.to_string())
