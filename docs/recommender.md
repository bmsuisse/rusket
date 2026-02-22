# Recommender Workflows

Three complementary recommendation strategies: cart add-ons, personalised "For You", and hybrid models.

`rusket` provides three complementary recommendation strategies that cover the most common revenue-generating use cases in e-commerce, retail, and content platforms.

| Strategy | Best for | API |
|---|---|---|
| **"Frequently Bought Together"** | Cart add-ons, shelf placement | `FPGrowth` / `AutoMiner` |
| **"For You" (Personalised)** | Homepage, email, loyalty | `ALS` / `BPR` |
| **Hybrid** | Blend both signals | `Recommender` |

---

## "Frequently Bought Together" — Cart Recommendations

```python
import pandas as pd
from rusket import AutoMiner

checkouts = pd.DataFrame({
    "receipt_id": [1, 1, 2, 2, 2, 3, 3, 4, 4, 4],
    "product":    ["espresso_beans", "grinder",
                   "espresso_beans", "milk_frother", "travel_mug",
                   "grinder", "milk_frother",
                   "espresso_beans", "grinder", "descaler"],
})

model = AutoMiner.from_transactions(
    checkouts,
    transaction_col="receipt_id",
    item_col="product",
    min_support=0.3,
)

basket   = ["grinder"]
add_ons  = model.recommend_items(basket, n=3)
rules = model.association_rules(metric="lift", min_threshold=1.0)
```

---

## "For You" — Personalised Recommendations with ALS / BPR

- **ALS** — best for score prediction and serendipitous discovery
- **BPR** — best when you care only about top-N ranking

```python
from rusket import ALS, BPR

purchases = pd.DataFrame({
    "customer_id": [1001, 1001, 1001, 1002, 1002, 1003],
    "sku":         ["A10", "B22", "C15",  "A10", "D33",  "B22"],
    "revenue":     [29.99, 49.00, 9.99,  29.99, 15.00, 49.00],
})

als = ALS(factors=64, iterations=15, alpha=40.0).from_transactions(
    purchases,
    user_col="customer_id",
    item_col="sku",
    rating_col="revenue",
)

bpr = BPR(factors=64, learning_rate=0.05, iterations=150).fit(user_item_csr)
```

### Getting recommendations

```python
items, scores = als.recommend_items(user_id=1001, n=5, exclude_seen=True)
top_customers, scores = als.recommend_users(item_id="D33", n=100)
```

---

## The Hybrid Recommender

```python
from rusket import ALS, Recommender, mine, association_rules

als  = ALS(factors=64, iterations=15).fit(user_item_csr)
freq  = mine(basket_ohe, min_support=0.01)
rules = association_rules(freq, num_itemsets=n_receipts)
rec = Recommender(als_model=als, rules_df=rules)
```

### 1. Personalised homepage ("For You")

```python
items, scores = rec.recommend_for_user(user_id=1001, n=5)
```

### 2. Hybrid — CF + product embeddings

```python
rec = Recommender(als_model=als, rules_df=rules, item_embeddings=product_vectors)

items, scores = rec.recommend_for_user(
    user_id=1001, n=5, alpha=0.7,
    target_item_for_semantic="B22",
)
```

### 3. Cart-based "Frequently Bought Together"

```python
cart = ["espresso_beans", "grinder"]
add_ons = rec.recommend_for_cart(cart, n=3)
```

### 4. Batch scoring — email campaign targeting

```python
batch = rec.predict_next_chunk(user_history_df, user_col="customer_id", k=5)
batch.to_parquet("s3://data-lake/recommendations/daily_picks.parquet")
```

---

## Item-to-Item Similarity — "You May Also Like"

```python
from rusket import similar_items, ALS

als = ALS(factors=128).fit(interactions)
similar_skus, similarity_scores = similar_items(als, item_id="B22", n=4)
```

!!! note "Latent-space similarity"
    Discovers implicit relationships — a premium coffee grinder may cluster tightly with an espresso machine even if they're rarely purchased in the same basket, because the *same type of customer* buys both.

---

## Cross-Selling Potential Scoring

```python
from rusket import score_potential

potential = score_potential(
    user_history=purchase_histories,
    als_model=als,
    target_categories=accessory_skus,
)
```

---

## Analytics Helpers

### Substitute / Cannibalising Products

```python
from rusket import find_substitutes

subs = find_substitutes(rules_df, max_lift=0.8)
```

### Customer Saturation

```python
from rusket import customer_saturation

saturation = customer_saturation(
    purchases_df, user_col="customer_id", category_col="category_id",
)
```

---

## Vector DB Export

```python
from rusket import export_item_factors

df_vectors = export_item_factors(als, include_labels=True)

import lancedb
db    = lancedb.connect("./product_vectors")
table = db.create_table("skus", data=df_vectors, mode="overwrite")
```

---

## Graph Analytics — Product Community Detection

```python
from rusket.viz import to_networkx
import networkx as nx

G = to_networkx(rules_df, edge_attr="lift")
centrality = nx.pagerank(G, weight="lift")
top_gateway = sorted(centrality, key=centrality.get, reverse=True)[:5]
```
