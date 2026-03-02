# Recommender Workflows

Three complementary recommendation strategies: cart add-ons, personalised "For You", and hybrid models.

`rusket` provides three complementary recommendation strategies that cover the most common revenue-generating use cases in e-commerce, retail, and content platforms.

| Strategy | Best for | API |
|---|---|---|
| **"Frequently Bought Together"** | Cart add-ons, shelf placement | `FPGrowth` / `FPGrowth` |
| **"For You" (Personalised)** | Homepage, email, loyalty | `ALS` / `BPR` |
| **Nearest Neighbors** | Simple, strong baselines | `ItemKNN` / `UserKNN` |
| **Hybrid** | Blend both signals | `Recommender` |

---

## "Frequently Bought Together" — Cart Recommendations

```python
import pandas as pd
from rusket import FPGrowth

checkouts = pd.DataFrame({
    "receipt_id": [1, 1, 2, 2, 2, 3, 3, 4, 4, 4],
    "product":    ["espresso_beans", "grinder",
                   "espresso_beans", "milk_frother", "travel_mug",
                   "grinder", "milk_frother",
                   "espresso_beans", "grinder", "descaler"],
})

model = FPGrowth.from_transactions(
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

als = ALS.from_transactions(
    purchases,
    user_col="customer_id",
    item_col="sku",
    rating_col="revenue",
    factors=64,
    iterations=15,
    alpha=40.0,
).fit()

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
from rusket import ALS, FPGrowth, Recommender

als  = ALS(factors=64, iterations=15).fit(user_item_csr)
model = FPGrowth(basket_ohe, min_support=0.01)
freq  = model.mine()
rules = model.association_rules()
rec = Recommender(model=als, rules_df=rules)
```

### 1. Personalised homepage ("For You")

```python
items, scores = rec.recommend_for_user(user_id=1001, n=5)
```

### 2. Hybrid — CF + product embeddings

```python
rec = Recommender(model=als, rules_df=rules, item_embeddings=product_vectors)

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

### 4. Databricks / Batch scoring — high speed cross-sell generation

Avoid slow Python `for` loops by using the Rust-accelerated `batch_recommend` to score all users across all CPU cores simultaneously.

```python
# Returns a native Polars DataFrame instantly: [user_id, item_id, score]
recommendations_pl = als.batch_recommend(n=10, format="polars")

# Need it in Delta Lake? Export factors and scores directly to Spark DataFrames!
user_factors_df = als.export_user_factors(normalize=True, format="spark")
item_factors_df = als.export_factors(normalize=True, format="spark")

# Save to Delta
user_factors_df.write.format("delta").mode("overwrite").saveAsTable("silver_layer.user_embeddings")
item_factors_df.write.format("delta").mode("overwrite").saveAsTable("silver_layer.item_embeddings")
```

---

## Nearest-Neighbor Collaborative Filtering — ItemKNN & UserKNN

Two complementary memory-based methods that consistently rank among the **top performers** in academic benchmarks ([Anelli et al. 2022](https://arxiv.org/abs/2203.01155)).

- **ItemKNN** — Finds items similar to what the user already liked. Computes item-item similarity via `X^T · X`.
- **UserKNN** — Finds users similar to the target and recommends what they liked. Computes user-user similarity via `X · X^T`.

Both support **BM25**, **TF-IDF**, **Cosine**, and raw **Count** weighting. The top-K neighbor pruning runs in parallel Rust.

### ItemKNN — "Customers who bought X also bought Y"

```python
from rusket import ItemKNN

item_knn = ItemKNN.from_transactions(
    purchases, user_col="user_id", item_col="item_id",
    method="bm25", k=100,
).fit()

items, scores = item_knn.recommend_items(user_id=42, n=10, exclude_seen=True)
```

### UserKNN — "Users similar to you enjoyed these"

```python
from rusket import UserKNN

user_knn = UserKNN.from_transactions(
    purchases, user_col="user_id", item_col="item_id",
    method="cosine", k=50,
).fit()

items, scores = user_knn.recommend_items(user_id=42, n=10, exclude_seen=True)
```

### Weighting methods

| Method | Description | Best for |
|---|---|---|
| `"bm25"` | BM25 term-frequency saturation + IDF | Long-tail catalogs (e-commerce) |
| `"tfidf"` | Standard TF-IDF weighting | General purpose |
| `"cosine"` | Row-normalized for cosine similarity | Dense datasets |
| `"count"` | Raw interaction counts (no weighting) | Quick baseline |

!!! tip "Which to choose?"
    Start with `ItemKNN(method="bm25")` — it's fast and robust. Switch to `UserKNN` for denser datasets or more diverse results. Evaluate both with `rusket.evaluate()`.

---

## Item-to-Item Similarity — "You May Also Like"

```python
from rusket import ALS

als = ALS(factors=128).fit(interactions)
similar_skus, similarity_scores = als.similar_items(item_id="B22", n=4)
```

!!! note "Latent-space similarity"
    Discovers implicit relationships — a premium coffee grinder may cluster tightly with an espresso machine even if they're rarely purchased in the same basket, because the *same type of customer* buys both.

---

## Cross-Selling Potential Scoring

```python
from rusket import score_potential

potential = score_potential(
    user_history=purchase_histories,
    model=als,
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
df_vectors = als.export_item_factors(include_labels=True)

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
