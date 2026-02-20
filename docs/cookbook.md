# Rusket Cookbook

A hands-on guide to every feature in `rusket` — from market basket analysis to billion-scale collaborative filtering.

---

## Setup

```python
pip install rusket
```

```python
import numpy as np
import pandas as pd
import polars as pl
from rusket import fpgrowth, eclat, association_rules, ALS
```

---

## 1. Market Basket Analysis with FPGrowth

### Generate a synthetic retail dataset

```python
np.random.seed(42)

items = [
    "Milk", "Bread", "Butter", "Eggs", "Cheese", "Yogurt",
    "Coffee", "Tea", "Sugar", "Apples", "Bananas", "Oranges",
    "Chicken", "Beef", "Fish", "Rice", "Pasta", "Tomato Sauce",
    "Onions", "Garlic",
]

n_transactions = 10_000
probs = np.power(np.arange(1, len(items) + 1, dtype=float), -0.7)
probs = np.clip(probs / probs.max() * 0.3, 0.01, 0.8)

df = pd.DataFrame(
    np.random.rand(n_transactions, len(items)) < probs,
    columns=items,
)
```

### Find frequent itemsets

```python
fi = fpgrowth(df, min_support=0.05, use_colnames=True)
# Returns a Pandas DataFrame with columns: support, itemsets
print(f"Found {len(fi)} frequent itemsets")
fi.sort_values("support", ascending=False).head(10)
```

### Generate association rules

```python
rules = association_rules(fi, num_itemsets=len(df), min_threshold=0.3)
# Returns: antecedents, consequents, support, confidence, lift, ...
strong = rules[(rules["confidence"] > 0.4) & (rules["lift"] > 1.2)]
strong.sort_values("lift", ascending=False).head(10)
```

### Limit itemset length

```python
# Only find pairs and triples — much faster for large catalogs
fi_pairs = fpgrowth(df, min_support=0.02, max_len=2, use_colnames=True)
```

---

## 2. ECLAT — When to Use vs FPGrowth

ECLAT uses a vertical bitset representation. It is **faster than FPGrowth for sparse datasets** with many unique items and a low support threshold.

```python
fi_ec = eclat(df, min_support=0.05, use_colnames=True)
```

**Rule of thumb:**

| Condition | Recommended algorithm |
|---|---|
| Dense dataset, few items | `fpgrowth` |
| Sparse dataset, many items, low support | `eclat` |
| Very large dataset (100M+ rows) | `fpgrowth` with streaming |

---

## 3. Transaction Helpers

Convert long-format order data (e.g., from a database) to the one-hot boolean matrix format required by `fpgrowth` and `eclat`.

### From a Pandas DataFrame

```python
from rusket import from_transactions

orders = pd.DataFrame({
    "order_id": [1, 1, 1, 2, 2, 3],
    "item":     ["Milk", "Bread", "Eggs", "Milk", "Butter", "Eggs"],
})

# Converts long-format → wide boolean matrix
basket = from_transactions(orders, user_col="order_id", item_col="item")
fi = fpgrowth(basket, min_support=0.3, use_colnames=True)
```

### From a Polars DataFrame

```python
orders_pl = pl.DataFrame({
    "order_id": [1, 1, 1, 2, 2, 3],
    "item":     ["Milk", "Bread", "Eggs", "Milk", "Butter", "Eggs"],
})

basket = from_transactions(orders_pl, user_col="order_id", item_col="item")
```

### From a Spark DataFrame

```python
# Works with PySpark DataFrames via .toPandas() under the hood
basket = from_transactions(spark_df, user_col="order_id", item_col="item")
```

---

## 4. Collaborative Filtering with ALS

`ALS` (Alternating Least Squares) learns user and item embeddings from implicit feedback (clicks, plays, purchases) and enables personalised recommendations.

### Prepare the interaction matrix

```python
from scipy.sparse import csr_matrix

# Build a user × item interaction matrix (1 = interacted)
n_users, n_items = 10_000, 5_000
rows = np.random.randint(0, n_users, size=200_000)
cols = np.random.randint(0, n_items, size=200_000)

mat = csr_matrix(
    (np.ones(len(rows), dtype=np.float32), (rows, cols)),
    shape=(n_users, n_items),
)
```

### Fit the model

```python
model = ALS(
    factors=64,        # latent dimension
    iterations=15,     # ALS alternating steps
    alpha=40.0,        # confidence scaling
    regularization=0.01,
    verbose=True,      # print per-iteration timing
    cg_iters=3,        # CG solver steps (3 is usually optimal)
)
model.fit(mat)
```

### Get recommendations

```python
# Top-10 items for user 0, excluding already-seen items
item_ids, scores = model.recommend_items(user_id=0, n=10, exclude_seen=True)

# Top-10 users likely to enjoy item 5
user_ids, scores = model.recommend_users(item_id=5, n=10)
```

### Fit from transaction data

```python
purchases = pd.DataFrame({
    "user_id": [1, 1, 2, 3, 3, 3],
    "item_id": [101, 102, 101, 103, 104, 101],
})

model = ALS(factors=32, iterations=10, verbose=True)
model.fit_transactions(purchases, user_col="user_id", item_col="item_id")
```

### Access latent factors directly

```python
# NumPy arrays (n_users × factors) and (n_items × factors)
print(model.user_factors.shape)  # (10000, 64)
print(model.item_factors.shape)  # (5000, 64)
```

---

## 5. Out-of-Core ALS for 1B+ Ratings

When the interaction matrix exceeds available RAM, use the out-of-core streaming loader. The CSR matrix is stored on SSD and the OS pages data into RAM on demand.

### Download and prepare the MovieLens 1B dataset

```python
import urllib.request, tarfile, os

# Download (~1.4 GB)
url = "https://files.grouplens.org/datasets/movielens/ml-20mx16x32.tar"
urllib.request.urlretrieve(url, "ml-1b.tar")

with tarfile.open("ml-1b.tar") as t:
    t.extractall("data/ml-1b/")
```

### Build the out-of-core CSR matrix

```python
import numpy as np
from scipy import sparse
from pathlib import Path

data_dir = Path("data/ml-1b/ml-20mx16x32")
npz_files = sorted(data_dir.glob("trainx*.npz"))

# Pass 1 — count ratings per user
max_user, max_item, nnz = 0, 0, 0
counts = np.zeros(100_000_000, dtype=np.int64)  # pre-allocate for max users

for f in npz_files:
    arr = np.load(f)["arr_0"]          # shape (N, 2) — [user_id, item_id]
    uids, iids = arr[:, 0], arr[:, 1]
    max_user = max(max_user, int(uids.max()))
    max_item = max(max_item, int(iids.max()))
    chunk_counts = np.bincount(uids, minlength=max_user + 1)
    counts[:len(chunk_counts)] += chunk_counts
    nnz += len(uids)

n_users, n_items = max_user + 1, max_item + 1
indptr = np.zeros(n_users + 1, dtype=np.int64)
np.cumsum(counts[:n_users], out=indptr[1:])

# Pass 2 — write indices/data to SSD memory maps
mmap_indices = np.memmap("indices.mmap", dtype=np.int32, mode="w+", shape=(nnz,))
mmap_data    = np.memmap("data.mmap",    dtype=np.float32, mode="w+", shape=(nnz,))
pos = indptr[:-1].copy()

for f in npz_files:
    arr = np.load(f)["arr_0"]
    uids, iids = arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int32)
    for u, i in zip(uids, iids):
        p = pos[u]
        mmap_indices[p] = i
        mmap_data[p]    = 1.0
        pos[u] += 1

mmap_indices.flush()
mmap_data.flush()
```

### Fit ALS on the out-of-core matrix

```python
# Bypass scipy's int32 limits by direct property assignment
mat = sparse.csr_matrix((n_users, n_items))
mat.indptr  = indptr
mat.indices = mmap_indices
mat.data    = mmap_data

model = ALS(
    factors=64,
    iterations=5,      # fewer iterations for 1B — each takes hours on SSD
    alpha=40.0,
    verbose=True,
    cg_iters=3,
)
model.fit(mat)
```

!!! tip "Hardware sizing"
    On a machine with ≥ 32 GB RAM the mmap working set stays hot in OS page cache and each iteration completes in ~5 minutes. On 8 GB RAM each iteration is disk-bound and takes hours.

---

## 6. Native Polars Integration

`rusket` returns itemsets as zero-copy **PyArrow `ListArray`** structures, making Polars interoperability very efficient.

```python
df_pl = pl.from_pandas(df)
fi_pl = fpgrowth(df_pl, min_support=0.05, use_colnames=True)

# LazyFrame works too:
lazy = df_pl.lazy()
# (convert to eager first before passing to fpgrowth)
fi_pl2 = fpgrowth(lazy.collect(), min_support=0.05, use_colnames=True)
```

### Query itemsets with PyArrow compute

```python
import pyarrow.compute as pc

# Find itemsets containing "Milk" as first element
contains_milk = pc.list_element(fi["itemsets"].array, 0) == "Milk"
fi[contains_milk].head()
```

### Convert to Python sets (only for small subsets)

```python
top_10 = fi.head(10).copy()
top_10["sets"] = top_10["itemsets"].apply(set)
```

---

## 7. Spark / Databricks Integration

```python
from rusket import from_transactions
from rusket import ALS

# FPGrowth from Spark
fi = fpgrowth(spark_df.toPandas(), min_support=0.05, use_colnames=True)

# ALS from Spark ratings table
ratings_spark = spark.table("ratings")  # user_id, item_id, rating
model = ALS(factors=64, iterations=10, verbose=True)
model.fit_transactions(ratings_spark, user_col="user_id", item_col="item_id", rating_col="rating")
```

!!! note "Large Spark tables"
    For tables with >100M rows, collect a representative sample or use the out-of-core loader from Section 5.

---

## 8. Tuning Guide

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
| `iterations` | 15 | 5–15 is typical; diminishing returns after 20 |
| `alpha` | 40.0 | Higher → stronger signal from implicit feedback |
| `regularization` | 0.01 | Increase if overfitting; decrease for denser data |
| `cg_iters` | 3 | CG solver steps per ALS step — 3 is almost always optimal |
| `verbose` | False | Set `True` to print per-iteration timing |

### Recommendation quality tips

- Use `regularization=0.1` for very sparse matrices (< 5 interactions/user)
- `alpha=10` works better for rating-weighted data vs binary implicit feedback
- For the best cold-start handling, combine ALS with popularity-based fallback
- Lower `cg_iters` (e.g., 1–2) for faster but noisier convergence on huge datasets
