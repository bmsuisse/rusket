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

## 1. Market Basket Analysis — Grocery Retail

### Business context

A supermarket chain wants to identify which product combinations appear most frequently in customer baskets across all checkout terminals. The output drives:

- **"Frequently Bought Together"** widgets on the self-checkout screen
- **Shelf adjacency** decisions (place high-lift pairs closer together)
- **Promotional bundles** (discount pairs with high confidence but low current margin)

### Prepare the basket data

In practice this comes from your POS system as a long-format order table. For demonstration we build a plausible synthetic dataset:

```python
import numpy as np
import pandas as pd
from rusket import from_transactions

np.random.seed(42)

# Realistic grocery catalogue with purchase probabilities (power-law distributed)
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
print(f"Simulated {n_receipts:,} receipts, {len(df_long):,} line items")

# Convert to one-hot basket matrix
basket = from_transactions(df_long, transaction_col="receipt_id", item_col="product")
```

### Find frequent product combinations

```python
from rusket import mine

# Keep combinations appearing in ≥5% of receipts
freq = mine(basket, min_support=0.05, use_colnames=True)
print(f"Found {len(freq):,} frequent product combinations")

top_combos = freq.sort_values("support", ascending=False)
print(top_combos.head(10))
# e.g. (Milk,) 55%, (Bread,) 52%, (Milk, Bread) 28% ...
```

### Generate cross-sell rules

```python
from rusket import association_rules

rules = association_rules(freq, num_itemsets=n_receipts, min_threshold=0.3)

# Filter for actionable rules: high confidence AND lift (better than random)
actionable = rules[(rules["confidence"] > 0.45) & (rules["lift"] > 1.2)]
print(actionable.sort_values("lift", ascending=False).head(10))
# antecedents     consequents   confidence  lift
# (Butter,)       (Bread,)      0.72        1.38   → 72% of Butter buyers also buy Bread
```

### Limit itemset length for large catalogues

```python
# For a full supermarket with 5000 SKUs: cap at pairs and triples to avoid explosion
freq_pairs = mine(basket, min_support=0.02, max_len=2, use_colnames=True)
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

## 4. Collaborative Filtering with ALS — "For You" Personalisation

`ALS` (Alternating Least Squares) learns latent user and item embeddings from **implicit feedback** (purchases, clicks, plays) and enables personalised "For You" recommendations.

### Business context

An e-commerce platform wants to show a personalised homepage to each logged-in user. ALS learns from past purchase history which categories of products each user affinity group prefers — without any explicit ratings.

### Fit from purchase history (event log)

```python
from rusket import ALS

# Purchase history from your order management system
purchases = pd.DataFrame({
    "customer_id": [1001, 1001, 1001, 1002, 1002, 1003, 1003, 1003],
    "sku":         ["A10", "B22", "C15",  "A10", "D33",  "B22", "C15", "E07"],
    "revenue":     [29.99, 49.00, 9.99,  29.99, 15.00, 49.00, 9.99, 22.00],
})

# revenue used as confidence weight (alpha scaling)
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
# Top-5 SKUs for customer 1002, excluding already-purchased items
skus, scores = model.recommend_items(user_id=1002, n=5, exclude_seen=True)
print(f"Recommended SKUs for customer 1002: {skus}")

# Target: which customers should receive a promo for SKU B22 (high-margin item)?
top_customers, scores = model.recommend_users(item_id="B22", n=100)
print(f"Top customers to target with B22 promo: {top_customers[:5]}")
```

### Fit from transaction data

```python
# If you have pre-built purchase integers from your warehouse:
model2 = ALS(factors=32, iterations=10, verbose=True)
model2 = ALS.from_transactions(purchases, transaction_col="customer_id", item_col="sku")
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

PrefixSpan discovers frequent sequences of events over time. Unlike standard market basket analysis (which looks at what is bought *together*), PrefixSpan finds patterns *across ordered events* — ideal for customer journey analysis, funnel optimisation, and churn prediction.

**Business scenario:** A SaaS company wants to understand which product page navigation sequences lead customers to checkout. Which paths are most common? Where do users drop off?

```python
import pandas as pd
from rusket import prefixspan, sequences_from_event_log

# 1. Website clickstream log — each row is one page visit
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

# 2. Convert to sequence format expected by the Rust miner
seqs, mapping = sequences_from_event_log(
    clickstream, user_col="session_id", time_col="timestamp", item_col="page"
)

# 3. Mine navigation sequences seen in ≥2 sessions (absolute count)
patterns_df = prefixspan(seqs, min_support=2, max_len=4)

# 4. Map integer IDs back to readable page names
patterns_df["path"] = patterns_df["sequence"].apply(
    lambda seq: " → ".join(mapping[s] for s in seq)
)
print(patterns_df[["support", "path"]].sort_values("support", ascending=False).head(10))
# support  path
# 3        Home
# 3        Pricing
# 3        Checkout
# 3        Home → Pricing
# 3        Pricing → Checkout
# 2        Home → Checkout                  ← users who skipped Pricing
# 2        Features → Pricing → Checkout   ← high-intent funnel path
```


---

## 8. High-Utility Pattern Mining (HUPM) — Profit-Driven Bundle Discovery

Frequent itemsets aren't always the most profitable. HUPM accounts for the **utility** (e.g., margin, revenue, quantity × price) of items to find sets that generate the *highest total business value* — even if they appear infrequently.

**Business scenario:** A wine shop wants to identify high-margin product bundles for a "Sommelier's Selection" gift box. Standard FP-Growth would surface budget items (e.g., sparkling water) because they're bought often. HUPM surfaces the highest-revenue product combinations instead:

```python
from rusket import HUPM

# Receipt data from the EPOS system — product_id and margin per line item
import pandas as pd

receipts = pd.DataFrame({
    "receipt_id": [1, 1, 1, 2, 2, 3, 3, 4, 4, 4],
    "product":    ["champagne", "foie_gras", "truffle_oil",
                   "champagne", "truffle_oil",
                   "foie_gras", "truffle_oil",
                   "champagne", "foie_gras", "truffle_oil"],
    "margin":     [18.50, 14.00, 8.00,   # receipt 1 margins
                   18.50, 8.00,            # receipt 2
                   14.00, 8.00,            # receipt 3
                   18.50, 14.00, 8.00],    # receipt 4
})

# Discover all product bundles generating ≥ €30 total margin
high_value = HUPM.from_transactions(
    receipts,
    transaction_col="receipt_id",
    item_col="product",
    utility_col="margin",
    min_utility=30.0,
).mine()

print(high_value.sort_values("utility", ascending=False))
# utility   itemset
# 122.0     [champagne, foie_gras, truffle_oil]  ← ideal gift box
# 74.0      [champagne, foie_gras]
# 62.0      [champagne, truffle_oil]
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

`rusket` provides native integration with PySpark out of the box, leaning heavily on **Apache Arrow** to completely bypass Python-to-JVM serialization and Pandas memory bloat.

### 10.1 Streaming 1B+ Rows from Spark (Zero-Copy)

Instead of using `.toPandas()` (which will OOM driver nodes instantly on large tables), use `mine_spark`, which streams Arrow partitions dynamically.

```python
from rusket import mine_spark

spark_df = spark.table("silver_transactions")

# Streams Arrow RecordBatches directly to the Rust backend
frequent_itemsets = mine_spark(
    spark_df, 
    n_items=500_000, 
    txn_col="transaction_id", 
    item_col="product_id", 
    min_support=0.001
)
```

### 10.2 Distributed Parallel Mining (Grouped)

If you have multi-tenant data (e.g., you want to mine *distinct* association rules per region, store, or customer segment), you can distribute `rusket` across your entire Databricks cluster using PySpark's `applyInArrow`.

```python
from rusket.spark import mine_grouped

spark_df = spark.table("retail_transactions")

# Rusket maps the workload across executor nodes. Each node runs pure Rust 
# on its Spark partition and yields the regional itemsets back.
regional_rules_df = mine_grouped(
    spark_df, 
    group_col="store_id", 
    min_support=0.05
)

display(regional_rules_df)
```

### 10.3 Collaborative Filtering (ALS) from Spark

For recommendation workloads, you can seamlessly feed Spark DataFrames containing `(user, item, rating)` triplets into ALS.

```python
from rusket import ALS

ratings_spark = spark.table("implicit_ratings") 
model = ALS(factors=64, iterations=10, verbose=True)

# Coerces to Pandas internally for fit (only for tables that fit in driver RAM)
model = ALS.from_transactions(
    ratings_spark, 
    transaction_col="user_id", 
    item_col="item_id", 
    rating_col="clicks",
    factors=64,
    iterations=10,
    verbose=True
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

`rusket` has native, out-of-the-box integration for routing sequence generation across **Pandas**, **Polars**, and **PySpark** `DataFrame`s. It utilizes lightning-fast internal grouped maps to feed the integers into the Rust parser.

### Prepare the Event Log (Pandas, Polars, or Spark)

```python
from rusket import prefixspan, sequences_from_event_log
import pandas as pd

# Let's say this is your PySpark, Polars, or Pandas DataFrame
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
# Passes natively to Arrow if events is a PySpark or Polars dataframe!
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
    lambda seq: [idx_to_item[idx] for seq]
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

---

## 14. GenAI / LLM Stack Integration

`rusket` provides native utilities to export its learned representations and rules into the modern Generative AI and graph analytics stack.

### Vector Export & Vector Databases (LanceDB)

You can easily export ALS latent user or item factors as vector embeddings to power RAG (Retrieval-Augmented Generation) or fast semantic similarity search in vector databases like LanceDB, FAISS, or Qdrant.

```python
import lancedb
from rusket import export_item_factors

# Export ALS item factors to a Pandas DataFrame
# Returns columns: ['item_id', 'vector'] (and 'item_label' if available)
df_vectors = export_item_factors(als_model)

# Connect to a local LanceDB instance
db = lancedb.connect("./lancedb")

# Ingest the embeddings into a table
table = db.create_table("item_embeddings", data=df_vectors, mode="overwrite")

# Perform a vector similarity search (e.g., finding items similar to a given query embedding)
query_vector = df_vectors.iloc[0]["vector"]
results = table.search(query_vector).limit(5).to_pandas()
print(results)
```

### Fast Item-to-Item Similarity

If you don't need a full vector database and just want fast, in-memory cosine similarity between items based on their ALS embeddings:

```python
from rusket import similar_items

# Find the top 5 most similar items to item_id=42 using Cosine Similarity
similar_ids, similarity_scores = similar_items(als_model, item_id=42, n=5)

print(f"Similar items: {similar_ids}")
print(f"Cosine similarities: {similarity_scores}")
```

### Graph Generation for Community Detection

Frequent Pattern Mining rules can be naturally represented as a directed graph. You can automatically convert them into a `networkx` graph to run community detection (like Louvain) and discover "Product Clusters" or "Categories".

```python
import networkx as nx
from rusket.viz import to_networkx

# rules is a Pandas DataFrame from rusket.association_rules()
# We use 'lift' as the edge weight connecting antecedents to consequents
G = to_networkx(rules_df, source_col="antecedents", target_col="consequents", edge_attr="lift")

# Run basic graph analytics
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# E.g., calculate PageRank to find the most influential products
centrality = nx.pagerank(G, weight='weight')
top_items = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top central products:", top_items)
```

---

## 15. Visualizing Latent Spaces (PCA)

When using `ALS`, raw embeddings capture both **magnitude** (how frequently an item is bought) and **direction** (the "taste" or behavioral profile of who buys it).

To map these multidimensional factors down to a 3D Plotly visualization for dashboarding, we apply **L2 Normalization** (to focus solely on Cosine Similarity / direction) followed by **PCA** (Principal Component Analysis).

Because `rusket` exposes ALS factors directly as NumPy arrays, you can do this **without adding dependencies like `scikit-learn` or PySpark**:

```python
import pandas as pd
import numpy as np
import plotly.express as px
from rusket import ALS

# 1. Load the Online Retail dataset
url = "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/refs/heads/master/data/retail-data/all/online-retail-dataset.csv"
df_purchases = pd.read_csv(url)
df_purchases = df_purchases.dropna(subset=["CustomerID", "Description"])

# 2. Fit an ALS model
model = ALS.from_transactions(df_purchases, transaction_col="CustomerID", item_col="StockCode", factors=64, iterations=15, alpha=40.0, seed=42)

# 3. L2 Normalization (Unit Sphere Projection) using pure NumPy
# Divide each latent factor row by its L2 norm (magnitude)
item_factors = model.item_factors
item_norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
item_factors_norm = item_factors / np.clip(item_norms, a_min=1e-10, a_max=None)

# 4. PCA Reduction (e.g. 64D -> 3D) using Singular Value Decomposition
def compute_pca_3d(data):
    # Mean centering
    data_centered = data - np.mean(data, axis=0)
    
    # SVD
    U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
    
    # Extract the top 3 principal components map
    components = Vt[:3]
    return np.dot(data_centered, components.T)

item_pca = compute_pca_3d(item_factors_norm)

# 5. Bind arrays back to a Pandas DataFrame for Plotly
df_viz = pd.DataFrame({
    "StockCode": model._item_labels, # The original dataset IDs mapped back
    "pca_1": item_pca[:, 0],
    "pca_2": item_pca[:, 1],
    "pca_3": item_pca[:, 2]
})

# Merge descriptions back in for hover labels
df_items = df_purchases[["StockCode", "Description"]].drop_duplicates("StockCode")
df_viz = df_viz.merge(df_items, on="StockCode", how="inner")

fig = px.scatter_3d(
    df_viz, x="pca_1", y="pca_2", z="pca_3",
    hover_name="Description",
    title="ALS Latent Space (3D PCA Mapping)"
)
fig.update_traces(marker=dict(size=3, opacity=0.7))
fig.show()
```

This workflow matches Spark MLlib's dimensionality reduction pipelines seamlessly while executing locally in microseconds.

---

## 16. Translating Spark MLlib to `rusket`

For users migrating from Databricks or PySpark, `rusket` offers a highly similar API without the distributed computing overhead. 

This example translates the famous Recommendation example from Chapter 28 of **Spark: The Definitive Guide** directly into `rusket` using pure Python and Pandas.

### Spark Version (Original)

```python
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

ratings = spark.read.text("/data/sample_movielens_ratings.txt") \
  .selectExpr("split(value, '::') as col") \
  .selectExpr(
      "cast(col[0] as int) as userId",
      "cast(col[1] as int) as movieId",
      "cast(col[2] as float) as rating"
  )

training, test = ratings.randomSplit([0.8, 0.2])

als = ALS().setMaxIter(5).setRegParam(0.01) \
  .setUserCol("userId").setItemCol("movieId").setRatingCol("rating")

alsModel = als.fit(training)
predictions = alsModel.transform(test)

evaluator = RegressionEvaluator().setMetricName("rmse") \
  .setLabelCol("rating").setPredictionCol("prediction")

print("RMSE =", evaluator.evaluate(predictions))
```

### `rusket` Version (Equivalent)

```python
import pandas as pd
import numpy as np
from rusket import ALS

# 1. Load the data using Pandas
url = "https://raw.githubusercontent.com/apache/spark/master/data/mllib/als/sample_movielens_ratings.txt"
ratings = pd.read_csv(url, sep="::", engine="python", 
                      names=["userId", "movieId", "rating", "timestamp"])

# 2. Random Split (80/20)
shuffled = ratings.sample(frac=1.0, random_state=42)
split_idx = int(len(shuffled) * 0.8)
training = shuffled.iloc[:split_idx]
test = shuffled.iloc[split_idx:]

# 3. Initialize and Fit the ALS Model
# Note: rusket uses `factors` instead of `rank`, and `iterations` instead of `maxIter`.
model = ALS.from_transactions(training, transaction_col="userId", item_col="movieId", rating_col="rating", factors=10, iterations=5, regularization=0.01, seed=42)

# 4. Generate Predictions for the test set
# rusket has a built-in vectorized score_potential helper for evaluating target vectors
from rusket.recommend import score_potential

# We reconstruct the user's history from the training set to mask known interactions
user_histories = training.groupby("userId")["movieId"].apply(list).to_dict()
# Ensure all users in the test set exist in our history mapping, even if empty
history_list = [user_histories.get(uid, []) for uid in range(model._n_users)]

# Calculate raw prediction scores across all users and all items
all_predictions = score_potential(history_list, model)

# 5. Evaluate RMSE
# Extract only the actual ratings we care about from the test set
test_users = test["userId"].values
test_movies = test["movieId"].values
actual_ratings = test["rating"].values

# Map the raw pandas IDs to rusket's internal 0-indexed matrix IDs
try:
    internal_user_ids = np.array([model._user_labels.index(u) for u in test_users])
    internal_movie_ids = np.array([model._item_labels.index(str(m)) for m in test_movies])
    
    # Extract predicted ratings
    predicted_ratings = all_predictions[internal_user_ids, internal_movie_ids]
    
    # Calculate RMSE
    valid_mask = ~np.isinf(predicted_ratings) & ~np.isnan(predicted_ratings)
    rmse = np.sqrt(np.mean((predicted_ratings[valid_mask] - actual_ratings[valid_mask]) ** 2))
    print(f"Root-mean-square error = {rmse:.4f}")
    
except ValueError as e:
    # Handle cold-start users/items in the test set not seen in training
    print("Cold start warning: Some users/items in test set were not in training.")
```

### Explanation of Key Differences
1. **No Distributed Execution:** PySpark builds physical query plans (`.transform()`, `.show()`). `rusket` executes completely eagerly in memory, heavily relying on Rust arrays and `numpy` for C-level vector operations.
2. **Cold Starts:** `rusket` is designed for implicit feedback recommendations, and its `transform`/prediction step expects `user_id` and `item_id` values to have been seen during `.fit()`. Proper production code should handle cold starts with popularity backups instead of omitting them.
3. **Implicit vs Explicit Feedback:** The Databricks exact example uses ALS for *explicit* ratings out of 5 stars to calculate Regression Error (RMSE). `rusket` focuses entirely on *implicit* feedback (clicks, purchases), so it's optimized for calculating Ranking Metrics (like Precision@K) rather than Regression Error. The math works identically, but it scales differently.
