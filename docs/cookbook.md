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
from rusket import mine, eclat, association_rules, ALS, BPR
from rusket import prefixspan, sequences_from_event_log, hupm, similar_items, Recommender, score_potential
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
fi = mine(df, min_support=0.05, use_colnames=True)
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
fi_pairs = mine(df, min_support=0.02, max_len=2, use_colnames=True)
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
| Dense dataset, few items | `mine(method="auto")` |
| Sparse dataset, many items, low support | `mine(method="auto")` |
| Very large dataset (100M+ rows) | `FPMiner` with streaming |

---

## 3. Transaction Helpers

Convert long-format order data (e.g., from a database) to the one-hot boolean matrix format required by `mine` (which automatically routes to `fpgrowth` or `eclat`).

### From a Pandas DataFrame

```python
from rusket import from_transactions

orders = pd.DataFrame({
    "order_id": [1, 1, 1, 2, 2, 3],
    "item":     ["Milk", "Bread", "Eggs", "Milk", "Butter", "Eggs"],
})

# Converts long-format → wide boolean matrix
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

---

## 6. Bayesian Personalized Ranking (BPR)

Unlike ALS which tries to reconstruct the full interaction matrix, BPR explicitly optimizes the model to rank positive observed items higher than unobserved items. This makes BPR excellent for top-N ranking tasks on implicit data (like clicks or views).

```python
from rusket import BPR
from scipy.sparse import csr_matrix
import numpy as np

# Prepare sparse interaction matrix
rows = np.random.randint(0, 1000, size=5000)
cols = np.random.randint(0, 500, size=5000)
mat = csr_matrix((np.ones(5000), (rows, cols)), shape=(1000, 500))

# Initialize and fit BPR with Hogwild! parallel SGD
model = BPR(
    factors=64,
    learning_rate=0.01,
    regularization=0.01,
    iterations=100,
    seed=42,
)
model.fit(mat)

# Recommend items just like ALS
items, scores = model.recommend_items(user_id=10, n=5)
```

---

## 7. Sequential Pattern Mining (PrefixSpan)

PrefixSpan discovers frequent sequences of events over time. Unlike standard market basket analysis where subsets within a single transaction are mined, PrefixSpan finds patterns *across ordered transactions*.

```python
import pandas as pd
from rusket import prefixspan, sequences_from_event_log

# 1. Start with an event log (e.g. clickstream)
events = pd.DataFrame({
    "user_id": [1, 1, 1, 2, 2, 3, 3, 3],
    "timestamp": [10, 20, 30, 15, 25, 5, 15, 35],
    "page": ["Home", "Product", "Cart", "Home", "Cart", "Product", "Cart", "Checkout"],
})

# 2. Convert to the nested list format expected by the Rust miner
seqs, mapping = sequences_from_event_log(
    events, user_col="user_id", time_col="timestamp", item_col="page"
)

# 3. Mine sequential patterns (min_support = number of sequences)
patterns_df = prefixspan(seqs, min_support=2, max_len=3)

# 4. Map the integer item IDs back to human-readable labels
patterns_df["sequence_labels"] = patterns_df["sequence"].apply(
    lambda seq: [mapping[item] for item in seq]
)
print(patterns_df.head())
```

---

## 8. High-Utility Pattern Mining (HUPM)

Frequent itemsets aren't always the most profitable. High-Utility Pattern Mining (HUPM) accounts for the utility (e.g., profit margin or revenue) of items to find sets that generate the *highest total value* across all transactions, regardless of frequency.

```python
from rusket import hupm

# Transactions (lists of item IDs) and their corresponding utilities (profit)
transactions = [
    [1, 2, 3],  # Transaction 1: Items 1, 2, 3
    [1, 3],     # Transaction 2: Items 1, 3
    [2, 3],     # Transaction 3: Items 2, 3
]

# The profit of each item inside that specific transaction
utilities = [
    [5.0, 2.0, 1.0], # Profits for items 1, 2, 3 in T1
    [5.0, 1.0],      # Profits for items 1, 3 in T2
    [2.0, 1.0],      # Profits for items 2, 3 in T3
]

# Find itemsets with a total global utility >= 7.0
high_value_patterns = hupm(transactions, utilities, min_utility=7.0)

# Output contains the 'utility' and 'itemset'
print(high_value_patterns)
```

---

## 9. Native Polars Integration

`rusket` returns itemsets as zero-copy **PyArrow `ListArray`** structures, making Polars interoperability very efficient.

```python
df_pl = pl.from_pandas(df)
fi_pl = mine(df_pl, min_support=0.05, use_colnames=True)

# LazyFrame works too:
lazy = df_pl.lazy()
# (convert to eager first before passing to fpgrowth)
fi_pl2 = mine(lazy.collect(), min_support=0.05, use_colnames=True)
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

## 10. Spark / Databricks Integration

```python
from rusket import from_transactions
from rusket import ALS

# FPGrowth from Spark
fi = mine(spark_df.toPandas(), min_support=0.05, use_colnames=True)

# ALS from Spark ratings table
ratings_spark = spark.table("ratings")  # user_id, item_id, rating
model = ALS(factors=64, iterations=10, verbose=True)
model.fit_transactions(ratings_spark, user_col="user_id", item_col="item_id", rating_col="rating")
```

!!! note "Large Spark tables"
    For tables with >100M rows, collect a representative sample or use the out-of-core loader from Section 5.

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
| `iterations` | 15 | 5–15 is typical; diminishing returns after 20 |
| `alpha` | 40.0 | Higher → stronger signal from implicit feedback |
| `regularization` | 0.01 | Increase if overfitting; decrease for denser data |
| `cg_iters` | 3 | CG solver steps per ALS step — 3 is almost always optimal |
| `verbose` | False | Set `True` to print per-iteration timing |

### BPR

| Parameter | Default | Notes |
|---|---|---|
| `factors` | 64 | Higher → better quality, more RAM. BPR requires more factors than ALS typically |
| `iterations` | 100 | BPR uses SGD and requires more iterations than ALS. Try 100-500. |
| `learning_rate` | 0.01 | SGD learning rate. Decrease if unstable, increase if slow convergence. |
| `regularization` | 0.01 | Increase if overfitting |

### PrefixSpan & HUPM

| Parameter | Default | Notes |
|---|---|---|
| `min_support` | required | Defines frequency for PrefixSpan or total value for HUPM |
| `max_len` | None | Cap itemset/sequence size to avoid combinatorial explosions |

### Recommendation quality tips

- Use `regularization=0.1` for very sparse matrices (< 5 interactions/user)
- `alpha=10` works better for rating-weighted data vs binary implicit feedback for ALS
- For top-N ranking optimization directly, use **BPR** instead of **ALS**. ALS is better for score prediction and serendipity.
- For the best cold-start handling, combine ALS/BPR with popularity-based fallback
- Lower `cg_iters` (e.g., 1–2) for faster but noisier convergence on huge ALS datasets

---

## 9. Sequential Pattern Mining (PrefixSpan)

When the **order** of events matters (e.g., website navigation paths, sequential purchases over time), use PrefixSpan instead of FP-Growth.

### Prepare the Event Log

```python
from rusket import prefixspan, sequences_from_event_log

events = pd.DataFrame({
    "user_id": [1, 1, 1, 2, 2, 3, 3, 3],
    "timestamp": [
        "2024-01-01 10:00", "2024-01-01 10:05", "2024-01-01 10:10",
        "2024-01-02 11:00", "2024-01-02 11:05",
        "2024-01-03 09:00", "2024-01-03 09:05", "2024-01-03 09:10"
    ],
    "page_id": ["home", "products", "checkout", "home", "products", "home", "products", "checkout"]
})

# Convert timestamp to datetime for correct sorting
events["timestamp"] = pd.to_datetime(events["timestamp"])

# Convert to sequence format required by prefixspan
sequences, idx_to_item = sequences_from_event_log(
    events, 
    user_col="user_id", 
    time_col="timestamp", 
    item_col="page_id"
)
```

### Mine Sequential Patterns

```python
# min_support is an absolute number of sequences
patterns = prefixspan(sequences, min_support=2, max_len=3)

# Map integer IDs back to original page names
patterns["sequence_names"] = patterns["sequence"].apply(
    lambda seq: [idx_to_item[idx] for idx in seq]
)

print(patterns[["support", "sequence_names"]])
#    support               sequence_names
# 0        3                       [home]
# 1        3                   [products]
# 2        3             [home, products]
# 3        2                   [checkout]
# 4        2             [home, checkout]
# 5        2         [products, checkout]
# 6        2   [home, products, checkout]
```

---

## 10. High-Utility Pattern Mining (HUPM)

Standard Frequent Itemset Mining (FP-Growth/Eclat) treats all items equally. HUPM considers the **profit** or **utility** of items, discovering itemsets that generate high total revenue even if they are bought infrequently.

```python
from rusket import hupm

# Item IDs bought
transactions = [
    [1, 2, 3], 
    [1, 3],    
    [2, 3]     
]

# Profit (or quantity * price) of each item in the respective transaction
utilities = [
    [5.0, 10.0, 2.0], # Transaction 1 profits
    [5.0, 2.0],       # Transaction 2 profits
    [10.0, 2.0]       # Transaction 3 profits
]

# Mine itemsets with at least 15.0 total utility
high_utility_itemsets = hupm(transactions, utilities, min_utility=15.0, max_len=3)
print(high_utility_itemsets)
#    utility    itemset
# 0     20.0       [2]
# 1     24.0    [2, 3]
# 2     17.0 [1, 2, 3]
```

---

## 11. Bayesian Personalized Ranking (BPR)

BPR is a matrix factorization model that optimizes for **ranking metrics** rather than reconstruction error (like ALS). It works by sampling positive (interacted) and negative (unseen) items and ensuring the positive items are ranked higher. Use it when interaction data is purely binary implicit feedback.

```python
from rusket import BPR
from scipy.sparse import csr_matrix
import numpy as np

# Create an implicit feedback matrix (users x items)
mat = csr_matrix((np.ones(10), ([0,0,0,1,1,2,2,3,3,4], [1,3,4,1,2,2,3,1,4,4])), shape=(5, 5))

model = BPR(
    factors=32,
    learning_rate=0.05,
    iterations=200,
    regularization=0.01,
    seed=42
)

# Fit the BPR model
model.fit(mat)

# Recommend 3 items for user 0, excluding items they already interacted with
item_ids, scores = model.recommend_items(user_id=0, n=3, exclude_seen=True)
print(item_ids, scores)
```

---

## 12. Item Similarity and Cross-Selling Potential

Once you have fitted an ALS or BPR model, the learned latent factors are incredibly useful for measuring item similarity and predicting missed cross-sell opportunities.

### Find Similar Products (Item-to-Item)

```python
from rusket import similar_items

# Given an ALS model fitted on purchases
item_ids, match_scores = similar_items(model, item_id=102, n=5)

print(f"Items similar to {102}: {item_ids}")
# => Items similar to 102: [105, 99, 110, 87, 10]
```

### Calculate Cross-Selling Potential

Identify the probability that a user *should* have bought an item by now, but hasn't (`score_potential`).

```python
from rusket import score_potential

user_purchase_history = [
    [0, 1, 5], # User 0 bought items 0, 1, 5
    [1, 3],    # User 1 bought items 1, 3
    [0]        # User 2 bought item 0
]

# Provide specific categories/items you want to cross-sell
target_items = [2, 4, 6]

# Matrix of shape (n_users, len(target_items))
scores = score_potential(
    user_purchase_history, 
    als_model=model, 
    target_categories=target_items
)

# The highest scores correspond to the users most primed to buy those specific targets
print("Cross-sell potential scores:")
print(scores)
```

---

## 13. Hybrid Recommender (ALS + Association Rules)

The `Recommender` workflow class wraps both your collaborative filtering models (ALS) and Frequent Pattern Mining rules into a single API. This easily enables the two most common placement strategies in e-commerce: **"For You"** (ALS) and **"Frequently Bought Together"** (Association Rules).

```python
from rusket import Recommender

# Initialize with both your fitted ALS model and Rules DataFrame
rec = Recommender(als_model=model, rules_df=strong_rules)

# 1. "For You" (Personalized cross-selling based on user history)
item_ids, scores = rec.recommend_for_user(user_id=125, n=5)

# 2. "Frequently Bought Together" (Cart-based additions)
active_cart = [10, 15] # User just added items 10 and 15
suggested_additions = rec.recommend_for_cart(active_cart, n=3)
```
