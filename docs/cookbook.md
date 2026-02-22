# Cookbook

A hands-on guide to every feature in `rusket` — from market basket analysis to billion-scale collaborative filtering.

---

## Setup

```bash
pip install rusket
```

```python
import numpy as np
import pandas as pd
import polars as pl
from rusket import mine, eclat, association_rules, ALS, BPR
from rusket import prefixspan, sequences_from_event_log, hupm, similar_items, Recommender, score_potential
```

---

## 1. Market Basket Analysis — Grocery Retail

### Business context

A supermarket chain wants to identify which product combinations appear most frequently in customer baskets. The output drives:

- **"Frequently Bought Together"** widgets on the self-checkout screen
- **Shelf adjacency** decisions (place high-lift pairs closer together)
- **Promotional bundles** (discount pairs with high confidence but low current margin)

### Prepare the basket data

```python
import numpy as np
import pandas as pd
from rusket import from_transactions

np.random.seed(42)

categories = {
    "Milk": 0.55, "Bread": 0.52, "Butter": 0.36, "Eggs": 0.41,
    "Cheese": 0.28, "Yogurt": 0.22, "Coffee": 0.31, "Tea": 0.18,
    "Sugar": 0.20, "Apples": 0.25, "Bananas": 0.30, "Oranges": 0.15,
    "Chicken": 0.35, "Pasta": 0.27, "Tomato Sauce": 0.26, "Onions": 0.40,
}

n_receipts = 10_000
df_long = pd.DataFrame(
    [(receipt, product)
     for receipt in range(n_receipts)
     for product, prob in categories.items()
     if np.random.rand() < prob],
    columns=["receipt_id", "product"],
)

basket = from_transactions(df_long, transaction_col="receipt_id", item_col="product")
```

### Find frequent product combinations

```python
from rusket import mine

freq = mine(basket, min_support=0.05, use_colnames=True)
print(f"Found {len(freq):,} frequent product combinations")
top_combos = freq.sort_values("support", ascending=False)
```

### Generate cross-sell rules

```python
from rusket import association_rules

rules = association_rules(freq, num_itemsets=n_receipts, min_threshold=0.3)
actionable = rules[(rules["confidence"] > 0.45) & (rules["lift"] > 1.2)]
print(actionable.sort_values("lift", ascending=False).head(10))
```

### Limit itemset length for large catalogues

```python
freq_pairs = mine(basket, min_support=0.02, max_len=2, use_colnames=True)
```

---

## 2. ECLAT — When to Use vs FPGrowth

ECLAT uses a vertical bitset representation. It is **faster than FPGrowth for sparse datasets**.

```python
freq_ec = eclat(basket, min_support=0.05, use_colnames=True)
```

| Condition | Recommended algorithm |
|---|---|
| Dense dataset, few items | `mine(method="auto")` |
| Sparse dataset, many items, low support | `mine(method="auto")` |
| Very large dataset (100M+ rows) | `FPMiner` with streaming |

---

## 3. Transaction Helpers

### From a Pandas DataFrame

```python
from rusket import from_transactions

orders = pd.DataFrame({
    "order_id": [1, 1, 1, 2, 2, 3],
    "item":     ["Milk", "Bread", "Eggs", "Milk", "Butter", "Eggs"],
})

basket = from_transactions(orders, user_col="order_id", item_col="item")
fi = mine(basket, min_support=0.3, use_colnames=True)
```

### From a Polars DataFrame

```python
orders_pl = pl.DataFrame({
    "order_id": [1, 1, 1, 2, 2, 3],
    "item":     ["Milk", "Bread", "Eggs", "Milk", "Butter", "Eggs"],
})
basket = from_transactions(orders_pl, user_col="order_id", item_col="item")
```

---

## 4. Collaborative Filtering with ALS

### Fit from purchase history

```python
from rusket import ALS

purchases = pd.DataFrame({
    "customer_id": [1001, 1001, 1001, 1002, 1002, 1003, 1003, 1003],
    "sku":         ["A10", "B22", "C15",  "A10", "D33",  "B22", "C15", "E07"],
    "revenue":     [29.99, 49.00, 9.99,  29.99, 15.00, 49.00, 9.99, 22.00],
})

model = ALS(factors=64, iterations=15, alpha=40.0, cg_iters=3, verbose=True)
model = ALS.from_transactions(
    purchases,
    transaction_col="customer_id",
    item_col="sku",
    rating_col="revenue",
)
```

### Get personalised recommendations

```python
skus, scores = model.recommend_items(user_id=1002, n=5, exclude_seen=True)
top_customers, scores = model.recommend_users(item_id="B22", n=100)
```

### Access latent factors directly

```python
print(model.user_factors.shape)  # (n_users, 64)
print(model.item_factors.shape)  # (n_items, 64)
```

---

## 5. Out-of-Core ALS for 1B+ Ratings

### Build the out-of-core CSR matrix

```python
import numpy as np
from scipy import sparse
from pathlib import Path

data_dir = Path("data/ml-1b/ml-20mx16x32")
npz_files = sorted(data_dir.glob("trainx*.npz"))

max_user, max_item, nnz = 0, 0, 0
counts = np.zeros(100_000_000, dtype=np.int64)

for f in npz_files:
    arr = np.load(f)["arr_0"]
    uids, iids = arr[:, 0], arr[:, 1]
    max_user = max(max_user, int(uids.max()))
    max_item = max(max_item, int(iids.max()))
    chunk_counts = np.bincount(uids, minlength=max_user + 1)
    counts[:len(chunk_counts)] += chunk_counts
    nnz += len(uids)

n_users, n_items = max_user + 1, max_item + 1
indptr = np.zeros(n_users + 1, dtype=np.int64)
np.cumsum(counts[:n_users], out=indptr[1:])

mmap_indices = np.memmap("indices.mmap", dtype=np.int32, mode="w+", shape=(nnz,))
mmap_data    = np.memmap("data.mmap",    dtype=np.float32, mode="w+", shape=(nnz,))
```

### Fit ALS on the out-of-core matrix

```python
mat = sparse.csr_matrix((n_users, n_items))
mat.indptr  = indptr
mat.indices = mmap_indices
mat.data    = mmap_data

model = ALS(factors=64, iterations=5, alpha=40.0, verbose=True, cg_iters=3)
model.fit(mat)
```

!!! tip
    On a machine with ≥ 32 GB RAM, each iteration completes in ~5 minutes. On 8 GB RAM, each iteration is disk-bound and takes hours.

---

## 6. Bayesian Personalized Ranking (BPR)

```python
from rusket import BPR
from scipy.sparse import csr_matrix
import numpy as np

rows = np.random.randint(0, 1000, size=5000)
cols = np.random.randint(0, 500, size=5000)
mat = csr_matrix((np.ones(5000), (rows, cols)), shape=(1000, 500))

model = BPR(factors=64, learning_rate=0.01, regularization=0.01, iterations=100, seed=42)
model.fit(mat)

items, scores = model.recommend_items(user_id=10, n=5)
```

---

## 7. Sequential Pattern Mining (PrefixSpan)

```python
import pandas as pd
from rusket import prefixspan, sequences_from_event_log

clickstream = pd.DataFrame({
    "session_id": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4],
    "timestamp":  [10, 20, 30, 15, 25, 5, 15, 35, 10, 18, 40],
    "page": [
        "Home", "Pricing", "Checkout",
        "Home", "Pricing",
        "Features", "Pricing", "Checkout",
        "Home", "Features", "Checkout",
    ],
})

seqs, mapping = sequences_from_event_log(
    clickstream, user_col="session_id", time_col="timestamp", item_col="page"
)
patterns_df = prefixspan(seqs, min_support=2, max_len=4)
patterns_df["path"] = patterns_df["sequence"].apply(
    lambda seq: " → ".join(mapping[s] for s in seq)
)
```

---

## 8. High-Utility Pattern Mining (HUPM)

```python
from rusket import HUPM
import pandas as pd

receipts = pd.DataFrame({
    "receipt_id": [1, 1, 1, 2, 2, 3, 3, 4, 4, 4],
    "product":    ["champagne", "foie_gras", "truffle_oil",
                   "champagne", "truffle_oil",
                   "foie_gras", "truffle_oil",
                   "champagne", "foie_gras", "truffle_oil"],
    "margin":     [18.50, 14.00, 8.00, 18.50, 8.00, 14.00, 8.00, 18.50, 14.00, 8.00],
})

high_value = HUPM.from_transactions(
    receipts,
    transaction_col="receipt_id",
    item_col="product",
    utility_col="margin",
    min_utility=30.0,
).mine()
```

---

## 9. Native Polars Integration

```python
df_pl = pl.from_pandas(df)
fi_pl = mine(df_pl, min_support=0.05, use_colnames=True)
```

---

## 10. Spark / Databricks Integration

### Streaming 1B+ Rows from Spark

```python
from rusket import mine_spark

spark_df = spark.table("silver_transactions")
frequent_itemsets = mine_spark(
    spark_df, n_items=500_000,
    txn_col="transaction_id", item_col="product_id",
    min_support=0.001
)
```

### Distributed Parallel Mining (Grouped)

```python
from rusket.spark import mine_grouped

regional_rules_df = mine_grouped(spark_df, group_col="store_id", min_support=0.05)
```

### Collaborative Filtering (ALS) from Spark

```python
from rusket import ALS

model = ALS.from_transactions(
    spark.table("implicit_ratings"),
    transaction_col="user_id", item_col="item_id", rating_col="clicks",
    factors=64, iterations=10,
)
```

!!! note "Out-of-Core Models"
    For Spark tables spanning >100M rows, use `mine_spark` for Frequent Pattern mining, or export the table to an Out-of-Core disk map (Section 5) for ALS factorisation.

---

## 11. Tuning Guide

### FPGrowth / ECLAT

| Parameter | Default | Effect |
|---|---|---|
| `min_support` | required | Lower → more itemsets, slower |
| `max_len` | None | Cap itemset size — huge speedup on large catalogs |
| `use_colnames` | False | Return column names instead of indices |

### ALS

| Parameter | Default | Notes |
|---|---|---|
| `factors` | 64 | Higher → better quality, more RAM, slower |
| `iterations` | 15 | 5–15 is typical |
| `alpha` | 40.0 | Higher → stronger signal |
| `cg_iters` | 3 | CG solver steps |

---

## 12. Item Similarity and Cross-Selling Potential

```python
from rusket import similar_items

item_ids, match_scores = similar_items(model, item_id=102, n=5)
```

---

## 13. Hybrid Recommender (ALS + Association Rules)

```python
from rusket import Recommender

rec = Recommender(als_model=model, rules_df=strong_rules)
item_ids, scores = rec.recommend_for_user(user_id=125, n=5)
suggested_additions = rec.recommend_for_cart([10, 15], n=3)
```

---

## 14. GenAI / LLM Stack Integration

```python
import lancedb
from rusket import export_item_factors

df_vectors = export_item_factors(als_model)
db = lancedb.connect("./lancedb")
table = db.create_table("item_embeddings", data=df_vectors, mode="overwrite")
```

---

## 15. Visualizing Latent Spaces (PCA)

```python
import numpy as np
import plotly.express as px

item_factors = model.item_factors
item_norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
item_factors_norm = item_factors / np.clip(item_norms, a_min=1e-10, a_max=None)

def compute_pca_3d(data):
    data_centered = data - np.mean(data, axis=0)
    U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
    return np.dot(data_centered, Vt[:3].T)

item_pca = compute_pca_3d(item_factors_norm)
fig = px.scatter_3d(x=item_pca[:, 0], y=item_pca[:, 1], z=item_pca[:, 2])
fig.show()
```
