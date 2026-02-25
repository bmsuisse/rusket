"""
Example 06 — Hybrid Recommender: CF + Association Rules + Semantic Similarity
============================================================================

Combines ALS collaborative filtering with frequent-pattern association rules
into a unified recommendation engine.  Three placement surfaces are served
from a single Recommender object:

1.  **"For You" homepage** — pure CF or hybrid CF + semantic
2.  **"Frequently Bought Together"** — association-rule cart cross-sell
3.  **Batch scoring** — overnight CRM pipeline for all users
"""

import numpy as np
import pandas as pd
from scipy import sparse as sp

from rusket import ALS, AutoMiner, Recommender

# ── 1. Synthetic purchase history ────────────────────────────────────────────
rng = np.random.default_rng(42)
n_users, n_items = 100, 50

# Sparse purchase matrix (implicit feedback)
purchase_prob = 0.08
purchases = (rng.random((n_users, n_items)) < purchase_prob).astype(np.float32)
csr = sp.csr_matrix(purchases)

print(f"Users: {n_users}  |  Items: {n_items}  |  Purchases: {csr.nnz:,}")

# ── 2. Train ALS model ──────────────────────────────────────────────────────
als = ALS(factors=16, iterations=10, seed=42)
als.fit(csr)

items, scores = als.recommend_items(user_id=0, n=5, exclude_seen=True)
print(f"\n[ALS only] User 0 top-5 items: {items.tolist()}")
print(f"           Scores:             {scores.round(3).tolist()}")

# ── 3. Mine basket association rules ────────────────────────────────────────
basket_df = pd.DataFrame(purchases.astype(bool), columns=[f"item_{i}" for i in range(n_items)])
miner = AutoMiner(basket_df, min_support=0.05)
freq = miner.mine(use_colnames=True)
rules = miner.association_rules(metric="lift", min_threshold=1.0)

print(f"\nFrequent itemsets found: {len(freq):,}")
print(f"Association rules:      {len(rules):,}")

# ── 4. Create Hybrid Recommender ────────────────────────────────────────────
rec = Recommender(model=als, rules_df=rules)

# Homepage recommendation
hybrid_items, hybrid_scores = rec.recommend_for_user(user_id=0, n=5)
print(f"\n[Hybrid] User 0 top-5 items: {hybrid_items.tolist()}")

# Cart cross-sell
cart = [f"item_{i}" for i in items[:2]]
cross_sell = rec.recommend_for_cart(cart_items=list(items[:2]), n=3)
print(f"\n[Cart cross-sell] Cart={cart}  Suggestions={cross_sell}")

# ── 5. Batch scoring ────────────────────────────────────────────────────────
user_df = pd.DataFrame({"user_id": range(10)})
batch = rec.predict_next_chunk(user_df, user_col="user_id", k=3)
print(f"\n[Batch] First 5 rows:\n{batch.head()}")

# ── 6. Save & Load the trained model ────────────────────────────────────────
als.save("als_model.pkl")
loaded = ALS.load("als_model.pkl")
loaded_items, _ = loaded.recommend_items(user_id=0, n=5, exclude_seen=True)
assert np.array_equal(items, loaded_items), "Save/load round-trip failed!"
print("\n✅ Model save/load round-trip verified.")

# Cleanup
import os
os.remove("als_model.pkl")

print("\nDone! 🎉")
