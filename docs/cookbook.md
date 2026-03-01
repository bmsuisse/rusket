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
from rusket import FPGrowth, Eclat, FPGrowth, association_rules
from rusket import ALS, eALS, BPR, PrefixSpan, HUPM, Recommender
```

---

## 1. Market Basket Analysis — Grocery Retail

### Business context

A supermarket chain wants to identify which product combinations appear most frequently in customer baskets. The output drives:

- **"Frequently Bought Together"** widgets on the self-checkout screen
- **Shelf adjacency** decisions (place high-lift pairs closer together)
- **Promotional bundles** (discount pairs with high confidence but low current margin)

### Prepare the basket data and find frequent combinations

```python
import numpy as np
import pandas as pd
from rusket import FPGrowth

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

miner = FPGrowth.from_transactions(
    df_long,
    transaction_col="receipt_id",
    item_col="product",
    min_support=0.05,
    use_colnames=True,
)
freq = miner.mine()
print(f"Found {len(freq):,} frequent product combinations")
top_combos = freq.sort_values("support", ascending=False)
```

### Generate cross-sell rules

```python
# Rules are now accessible directly from the miner instance
rules = miner.association_rules(min_threshold=0.3)
actionable = rules[(rules["confidence"] > 0.45) & (rules["lift"] > 1.2)]
print(actionable.sort_values("lift", ascending=False).head(10))
```

### Limit itemset length for large catalogues

```python
miner_pairs = FPGrowth.from_transactions(
    df_long,
    transaction_col="receipt_id",
    item_col="product",
    min_support=0.02,
    max_len=2,
    use_colnames=True,
)
freq_pairs = miner_pairs.mine()
```

---

## 2. ECLAT — When to Use vs FPGrowth

ECLAT uses a vertical bitset representation. It is **faster than FPGrowth for sparse datasets**.

```python
from rusket import Eclat

freq_ec = Eclat.from_transactions(
    df_long,
    transaction_col="receipt_id",
    item_col="product",
    min_support=0.05,
    use_colnames=True,
).mine()
```

| Condition | Recommended class |
|---|---|
| Dense dataset, few items | `FPGrowth` |
| Sparse dataset, many items, low support | `Eclat` |
| Very large dataset (100M+ rows) | `FPMiner` with streaming |

---

## 3. Transaction Input Formats

### From a Pandas DataFrame

```python
import pandas as pd
from rusket import FPGrowth

orders = pd.DataFrame({
    "order_id": [1, 1, 1, 2, 2, 3],
    "item":     ["Milk", "Bread", "Eggs", "Milk", "Butter", "Eggs"],
})

freq = FPGrowth.from_transactions(
    orders,
    transaction_col="order_id",
    item_col="item",
    min_support=0.3,
    use_colnames=True,
).mine()
```

### From a Polars DataFrame

```python
import polars as pl
from rusket import FPGrowth

orders_pl = pl.DataFrame({
    "order_id": [1, 1, 1, 2, 2, 3],
    "item":     ["Milk", "Bread", "Eggs", "Milk", "Butter", "Eggs"],
})

freq = FPGrowth.from_transactions(
    orders_pl,
    transaction_col="order_id",
    item_col="item",
    min_support=0.3,
    use_colnames=True,
).mine()
```

### From a list of lists

```python
from rusket import FPGrowth

baskets = [["Milk", "Bread"], ["Milk", "Eggs", "Butter"], ["Bread", "Eggs"]]
freq = FPGrowth(baskets, min_support=0.5, use_colnames=True).mine()
```

---

## 4. Collaborative Filtering with ALS

### Fit from purchase history

```python
import pandas as pd
from rusket import ALS

purchases = pd.DataFrame({
    "customer_id": [1001, 1001, 1001, 1002, 1002, 1003, 1003, 1003],
    "sku":         ["A10", "B22", "C15",  "A10", "D33",  "B22", "C15", "E07"],
    "revenue":     [29.99, 49.00, 9.99,  29.99, 15.00, 49.00, 9.99, 22.00],
})

model = ALS.from_transactions(
    purchases,
    transaction_col="customer_id",
    item_col="sku",
    rating_col="revenue",
    factors=64,
    iterations=15,
    alpha=40.0,
    cg_iters=3,
).fit()
```

### Get personalised recommendations

```python
skus, scores = model.recommend_items(user_id=1002, n=5, exclude_seen=True)
top_customers, scores = model.recommend_users(item_id="B22", n=100)
```

### Access latent factors (item embeddings) directly

```python
# NumPy arrays (n_users x factors) and (n_items x factors)
print(model.user_factors.shape)  # (n_users, 64)
print(model.item_factors.shape)  # (n_items, 64)

# Semantic alias for LLM/GenAI workflows
embeddings = model.item_embeddings
```

---

## 4b. Element-wise ALS (eALS) — Faster Default

`eALS` is a convenience wrapper that sets `use_eals=True` by default. It updates factors element-by-element and is typically **faster and more memory-efficient** than the standard CG solver.

```python
from rusket import eALS

model = eALS.from_transactions(
    purchases,
    transaction_col="customer_id",
    item_col="sku",
    rating_col="revenue",
    factors=64,
    iterations=15,
    alpha=40.0,
).fit()

# Exact same API as ALS — all methods work identically
skus, scores = model.recommend_items(user_id=1002, n=5)
```

!!! tip
    `eALS` and `ALS(use_eals=True)` produce identical results.  Use `eALS` for a cleaner API.

## 5. Out-of-Core ALS for 1B+ Ratings

### Build the out-of-core CSR matrix

```python
import numpy as np
from scipy import sparse
from pathlib import Path

data_dir = Path("data/ml-1b/ml-20mx16x32")
npz_files = sorted(data_dir.glob("trainx*.npz"))

# ... (out of core logic) ...
```

### Fit ALS on the out-of-core matrix

```python
from rusket import ALS

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
import numpy as np
import pandas as pd
from rusket import BPR

purchases = pd.DataFrame({
    "user_id": np.random.randint(0, 1000, size=5000),
    "item_id": np.random.randint(0, 500, size=5000),
})

model = BPR.from_transactions(
    purchases,
    transaction_col="user_id",
    item_col="item_id",
    factors=64,
    learning_rate=0.01,
    regularization=0.01,
    iterations=100,
    seed=42,
).fit()

items, scores = model.recommend_items(user_id=10, n=5)
```

---

## 7. Sequential Pattern Mining (PrefixSpan)

```python
import pandas as pd
from rusket import PrefixSpan

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

miner = PrefixSpan.from_transactions(
    clickstream,
    user_col="session_id",
    time_col="timestamp",
    item_col="page",
    min_support=2,
    max_len=4,
)
patterns_df = miner.mine()
print(patterns_df.head(10))
```

---

## 8. High-Utility Pattern Mining (HUPM)

```python
import pandas as pd
from rusket import HUPM

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

All miners accept Polars DataFrames directly — no conversion needed:

```python
import polars as pl
from rusket import FPGrowth

df_pl = pl.read_parquet("transactions.parquet")

freq = FPGrowth.from_transactions(
    df_pl,
    transaction_col="order_id",
    item_col="product_id",
    min_support=0.05,
    use_colnames=True,
).mine()
```

---

## 10. Spark / Databricks Integration

### Streaming 1B+ Rows from Spark

```python
from rusket import FPMiner

spark_df = spark.table("silver_transactions")
frequent_itemsets = FPMiner(
    spark_df,
    n_items=500_000,
    txn_col="transaction_id",
    item_col="product_id",
    min_support=0.001,
).mine()
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
    transaction_col="user_id",
    item_col="item_id",
    rating_col="clicks",
    factors=64,
    iterations=10,
).fit()
```

---

## 11. Databricks: High-Speed Cross-Sell Generation

When working in Databricks with millions of users, Python `for` loops are a massive bottleneck. Use `batch_recommend` to leverage Rust's parallel iterators (Rayon) and return native Spark or Polars DataFrames instantly.

```python
from rusket import ALS
import polars as pl
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
purchases = spark.table("bronze_layer.customer_transactions")

# 1. Train the model using the fast Polars bridge
als = ALS.from_transactions(
    purchases.toPandas(), # Or pass Polars directly if memory allows
    transaction_col="customer_id",
    item_col="product_id",
    rating_col="sales_amount",
    factors=128,
    iterations=15,
).fit()

# 2. Score ALL users simultaneously across all CPU cores (Rust Rayon)
#    Returns a fast Polars DataFrame: [user_id, item_id, score]
recommendations_pl = als.batch_recommend(n=10, format="polars")

# 3. Export L2-normalized item and user factors directly to Spark for Delta tables
user_factors_df = als.export_user_factors(normalize=True, format="spark")
item_factors_df = als.export_factors(normalize=True, format="spark")

# 4. Save to Delta
user_factors_df.write.format("delta").mode("overwrite").saveAsTable("silver_layer.user_embeddings")
item_factors_df.write.format("delta").mode("overwrite").saveAsTable("silver_layer.item_embeddings")
```

---

## 12. Tuning Guide

### FPGrowth / Eclat / FPGrowth

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
| `use_eals` | False | Use eALS solver (faster, less memory) |
| `eals_iters` | 1 | Inner iterations for eALS |
| `anderson_m` | 0 | Anderson acceleration history (5 recommended) |

---

## 13. Item Similarity and Cross-Selling Potential

```python
# Now part of the Model class
item_ids, match_scores = model.similar_items(item_id=102, n=5)
```

---

## 14. Hybrid Recommender (ALS + Association Rules)

```python
from rusket import Recommender

rec = Recommender(model=model, rules_df=rules)
item_ids, scores = rec.recommend_for_user(user_id=125, n=5)
suggested_additions = rec.recommend_for_cart([10, 15], n=3)
```

---

## 15. GenAI / LLM Stack Integration

```python
import lancedb
# Direct export from model
df_vectors = model.export_factors()
db = lancedb.connect("./lancedb")
table = db.create_table("item_embeddings", data=df_vectors, mode="overwrite")
```

---

## 16. Visualizing Latent Spaces (PaCMAP)

```python
# Built-in interactive 2D PaCMAP visualization via fluent API
fig = model.fit().pacmap2().plot(title="Latent Item Space")
fig.show()
```

---

## 17. Dealing with Cold Starts

The **"cold start" problem** is one of the most common challenges in building recommender systems. It occurs when a system cannot draw accurate inferences because it hasn't yet gathered enough data about a user or an item.

Here is how `rusket` addresses the three main types of cold starts:

### 1. Handling User Cold Starts (The "Folding In" Strategy)

If you have an existing ALS model and a new user signs up and clicks on a few items, you don't need to retrain the entire matrix. You can instantly "fold in" their early interactions (e.g. from an onboarding flow) to compute their latent factors on the fly:

```python
import rusket

# Assume model is already fitted on millions of users
model = rusket.ALS(factors=64).fit(X)

# A new user views items [3, 105, 992]
new_user_items = [3, 105, 992]

# Instantly compute their 64-dimensional latent factor vector
user_factors = model.recalculate_user(new_user_items)

# You can now multiply this vector against model.item_factors to score all items
scores = model.item_factors.dot(user_factors)
top_items = scores.argsort()[::-1][:10]
```

### 2. Handling System Cold Starts (Knowledge & Context-Aware)

If you want to recommend items to a user based purely on their demographics or context *before* they even make a single click, you should use **Factorization Machines (`rusket.FM`)**.

FM allows you to use a sparse feature matrix (one-hot encoded categories, ages, locations, time of day) instead of just user and item IDs. By learning the pairwise interactions between these features, FM can recommend items based entirely on metadata.

```python
import rusket
from scipy.sparse import csr_matrix
import numpy as np

# Format: [User=Alice, Item=Laptop, Age=25-34, Category=Electronics]
# 1 represents the presence of that categorical feature
X = csr_matrix([
    [1, 1, 1, 1], # Alice bought Laptop
    [0, 1, 0, 1]  # Bob did not buy Laptop
], dtype=np.float32)

y = np.array([1.0, 0.0])

model = rusket.FM(factors=8, iterations=100)
model.fit(X, y)

# Predict CTR for a new user based on demographics
# X_new = [User=Charlie, Item=Laptop, Age=25-34, Category=Electronics]
X_new = csr_matrix([[0, 1, 1, 1]], dtype=np.float32)

ctr_prob = model.predict_proba(X_new)
```

Alternatively, `rusket`'s Association Rule mining (`FPGrowth`, `Eclat`) can act as a knowledge-based fallback. `Recommender.recommend_for_cart()` uses explicit `IF (A) THEN (B)` rules to suggest items without relying on converged user factors.

### 3. Handling Item Cold Starts (Content-Based Hybrid)

When a new product is added to the catalog and no one has interacted with it yet, `rusket.Recommender` can fall back to semantic similarity.

By providing an `item_embeddings` matrix (e.g., dense vectors generated from OpenAI based on product descriptions), the `Recommender` intelligently blends behavioral CF scores with semantic similarity:

```python
from rusket import Recommender

# Set alpha=0.0 for pure Content-Based semantic recommender, or alpha=0.5 for Hybrid
rec = Recommender(
    model=als_model, 
    item_embeddings=llm_text_embeddings
)

items, scores = rec.recommend_for_user(user_id=123, alpha=0.5)
```

