# Recommender Workflows

While `rusket` provides blazing-fast core algorithms like Alternating Least Squares (ALS) and FP-Growth, raw algorithms often require heavy lifting to translate into business outcomes.

To bridge this gap, `rusket` includes a high-level **Business Recommender API** designed for e-commerce, content platforms, and marketing analytics.

---

## Quick Cart Recommendations (Mining + Rules)

The simplest path to cart recommendations uses the OOP mining API. All mining classes inherit `recommend_items` from `RuleMinerMixin`:

```python
import pandas as pd
from rusket import FPGrowth  # or Eclat, AutoMiner

# Build from raw transactional data
model = FPGrowth.from_transactions(
    df,                   # pd.DataFrame with (order_id, item) columns
    min_support=0.05,
)

# Get top-3 suggestions for an active basket
suggestions = model.recommend_items(["bread", "milk"], n=3)
# e.g. ["butter", "eggs", "cheese"]

# Or inspect rules directly
rules = model.association_rules(metric="lift", min_threshold=1.0)
```

`num_itemsets` is inferred automatically — no extra wiring needed.

---

## ALS / BPR Collaborative Filtering

### Fitting

```python
from rusket import ALS, BPR

# Option A: from a scipy CSR user-item matrix
als = ALS(
    factors=64,
    iterations=15,
    cg_iters=3,          # Conjugate Gradient iterations per solve
    use_cholesky=False,  # True = exact Cholesky (better for dense data)
    anderson_m=5,        # Anderson Acceleration — ~30-50% fewer iterations
).fit(user_item_csr)

# Option B: straight from a long-format event log
als = ALS(factors=64).from_transactions(
    df,                       # pd.DataFrame | pl.DataFrame | Spark DataFrame
    user_col="user_id",
    item_col="item_id",
    rating_col=None,          # optional explicit rating column
)

# BPR is a drop-in alternative (optimises ranking, not reconstruction)
bpr = BPR(factors=64, learning_rate=0.05, iterations=150).fit(user_item_csr)
```

### Item and User Recommendations

```python
# Top-N items for a user
items, scores = als.recommend_items(user_id=42, n=10, exclude_seen=True)

# Top-N users for an item (reverse lookup / targeting)
users, scores = als.recommend_users(item_id=99, n=5)
```

---

## The Hybrid Recommender

The `Recommender` class blends **Collaborative Filtering** (ALS/BPR) with **Frequent Pattern Mining** (Association Rules) to provide the "Next Best Action" for any context.

```python
from rusket import ALS, Recommender, mine, association_rules

# 1. Train your CF model
als = ALS(factors=64, iterations=15).fit(user_item_csr)

# 2. Mine association rules from basket data
freq  = mine(basket_matrix, min_support=0.01)
rules = association_rules(freq, num_itemsets=n_transactions)

# 3. Create the Hybrid Engine
rec = Recommender(als_model=als, rules_df=rules)
```

### 1. Personalized Recommendations (CF)

```python
# Pure ALS collaborative filtering
items, scores = rec.recommend_for_user(user_id=42, n=5)
print(f"Recommended: {items}")
```

### 2. Hybrid CF + Semantic Blending

When you have external item embeddings (e.g. from a product description vector index), you can blend CF scores with semantic similarity:

```python
rec = Recommender(als_model=als, rules_df=rules, item_embeddings=embeddings)

# alpha=1.0 → pure CF, alpha=0.0 → pure semantic, values in between = hybrid
items, scores = rec.recommend_for_user(
    user_id=42,
    n=5,
    alpha=0.7,                     # 70% CF + 30% semantic
    target_item_for_semantic=99,   # anchor item for similarity lookup
)
```

### 3. Cart-Based Cross-Selling

When a user adds items to their shopping cart, use deterministic association rules for "Frequently bought together" suggestions:

```python
# User has items 14 and 7 in their cart
suggested = rec.recommend_for_cart([14, 7], n=3)
print(f"Others also bought: {suggested}")
```

### 4. Batch Recommendations

Score all users in a DataFrame at once:

```python
batch_df = rec.predict_next_chunk(user_history_df, user_col="user_id", k=5)
# Returns a DataFrame with columns [user_id, recommended_items]
```

---

## Item-to-Item Similarity (i2i)

For anonymous visitors (no user context), use the latent item factors from ALS/BPR to find conceptually similar products via Cosine Similarity:

```python
from rusket import similar_items, ALS

als = ALS(factors=128).fit(interactions)

# 4 products most similar to product 99
similar, similarities = similar_items(als, item_id=99, n=4)
print(similar)        # [100, 95, 102, 88]
print(similarities)   # [0.98, 0.94, 0.89, 0.85]
```

> **Note:** Because this operates on latent factors, it discovers implicit relationships — e.g. a high-end coffee grinder may be similar to an espresso machine even if they aren't often directly bought together.

---

## Cross-Selling Potential Scoring

Quantify the "missed opportunity" — the probability a user *should* have bought an item by now but hasn't. Perfect for building highly targeted email and retargeting audiences.

```python
from rusket import score_potential

# user_history: list of item IDs each user has already interacted with
potential_matrix = score_potential(
    user_history=[[14, 7], [99], [5, 6, 7]],
    als_model=als,
    target_categories=[101, 102, 103],  # e.g. the "Electronics" category
)
# Shape: (n_users, len(target_categories))
# Items already bought are masked with -infinity.
```

---

## Analytics Helpers

### Find Substitutes / Cannibalizing Products

Items with high individual support but low co-occurrence (lift < 1.0) likely compete with each other:

```python
from rusket import find_substitutes

substitutes = find_substitutes(rules_df, max_lift=0.8)
# Returns a DataFrame sorted ascending by lift (worst cannibalization first)
```

### Customer Saturation

Segment users by their purchase depth within a category or item catalogue and split into deciles:

```python
from rusket import customer_saturation

saturation = customer_saturation(
    df,
    user_col="user_id",
    category_col="category_id",  # or item_col="item_id"
)
# Returns: unique_count, saturation_pct, decile
```

---

## Vector DB Export

Export ALS/BPR item factors as a Pandas DataFrame ready for FAISS / Qdrant / Pinecone:

```python
from rusket import export_item_factors

df_vectors = export_item_factors(als_model, include_labels=True)
# Each row: one item, columns are latent dimensions
```

---

## Graph Analytics

Convert association rules into a NetworkX directed graph for community detection and product clustering:

```python
from rusket.viz import to_networkx

G = to_networkx(rules_df, edge_attr="lift")
```
