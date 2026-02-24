<p align="center">
  <img src="docs/assets/logo_wide.svg" alt="rusket logo" width="520" height="200" />
</p>

<p align="center">
  <strong>Ultra-fast Recommender Engines & Market Basket Analysis for Python, written in Rust.</strong><br>
  <em>Made with ❤️ by the Data & AI Team.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/rusket/"><img src="https://img.shields.io/pypi/v/rusket?color=%2334D058&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.83%2B-orange?logo=rust" alt="Rust"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://bmsuisse.github.io/rusket/"><img src="https://img.shields.io/badge/docs-Zensical-blue" alt="Docs"></a>
</p>

---

> **⚠️ Note:** `rusket` is currently under heavy construction. The API will probably change in upcoming versions.

**rusket** is a modern, Rust-powered library for Market Basket Analysis and Recommender Engines. It delivers significant speed-ups and lower memory usage compared to traditional Python implementations, while natively supporting Pandas, Polars, and Spark out of the box.

**Zero runtime dependencies.** No TensorFlow, no PyTorch, no JVM — just `pip install rusket` and go. The entire engine is compiled Rust, distributed as a single ~3 MB wheel.

It features Collaborative Filtering (ALS, BPR, SVD, LightGCN, ItemKNN, EASE) and Pattern Mining (FP-Growth, Eclat, HUPM, PrefixSpan) with high performance and low memory footprints. Both functional and OOP APIs are available for seamless integration.

---

## ✨ Highlights

| | `rusket` | `LibRecommender` | `implicit` | `pyspark.ml` |
|---|---|---|---|---|
| **Core language** | Rust (PyO3) | TF + PyTorch + Cython | Cython / C++ | Scala / Java (JVM) |
| **Runtime deps** | **0** | TF + PyTorch + gensim (~2 GB) | OpenBLAS / MKL | JVM + Spark |
| **Install size** | ~3 MB | ~2 GB | ~50 MB | ~300 MB |
| **Algorithms** | ALS, BPR, SVD, LightGCN, ItemKNN, EASE, FP-Growth, Eclat, HUPM, PrefixSpan | ALS, BPR, SVD, LightGCN, ItemCF, FM, DeepFM, ... | ALS, BPR | ALS, FP-Growth, PrefixSpan |
| **Recommender API** | ✅ Hybrid Engine + i2i Similarity | ✅ | ✅ | ✅ (ALS only) |
| **Graph & Embeddings** | ✅ NetworkX Export, Vector DB Export | ❌ | ❌ | ❌ |
| **OOP class API** | ✅ `ALS.from_transactions(df)` | ✅ | ✅ | ✅ |
| **Pandas / Polars / Spark** | ✅ / ✅ / ✅ | ✅ / ❌ / ❌ | ❌ / ❌ / ❌ | ❌ / ❌ / ✅ |
| **Parallel execution** | ✅ Rayon work-stealing | ✅ TF/PyTorch threads | ✅ OpenMP | ✅ Spark Cluster |
| **Memory** | Low (native Rust buffers) | High (TF/PyTorch graphs) | Low (C++ arrays) | High (JVM overhead) |

---

## 📦 Installation

```bash
pip install rusket
# or with uv:
uv add rusket
```

**Optional extras:**

```bash
# Polars support
pip install "rusket[polars]"

# Pandas/NumPy support (usually already installed)
pip install "rusket[pandas]"
```

---

## 🚀 Quick Start

### "Frequently Bought Together" — Grocery Checkout Data

Identify which products co-occur most in customer baskets — the foundation of cross-sell widgets, promotional bundles, and shelf placement decisions.

```python
import pandas as pd
from rusket import AutoMiner

# One week of supermarket checkout data (1 row = 1 receipt, 1 col = 1 SKU)
receipts = pd.DataFrame({
    "milk":         [1, 1, 0, 1, 1, 0, 1],
    "bread":        [1, 0, 1, 1, 0, 1, 1],
    "butter":       [1, 0, 1, 0, 0, 1, 0],
    "eggs":         [0, 1, 1, 0, 1, 0, 1],
    "coffee":       [0, 1, 0, 0, 1, 1, 0],
    "orange_juice": [1, 0, 0, 1, 0, 0, 1],
}, dtype=bool)

# Step 1 — which SKU combinations appear in ≥40% of receipts?
# AutoMiner selects FP-Growth or Eclat based on catalogue density
model = AutoMiner(receipts, min_support=0.4)
freq = model.mine(use_colnames=True)

# Step 2 — keep rules with ≥60% confidence
rules = model.association_rules(metric="confidence", min_threshold=0.6)

# Lift > 1 means customers buy these together more than chance alone
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]]
      .sort_values("lift", ascending=False))
```

---

### 🛒 E-Commerce Order Lines (Long Format)

Real-world data arrives as `(order_id, sku)` rows from a database — not one-hot matrices.

All mining algorithms expose a class-based API that goes straight from order lines to recommendations:

```python
import pandas as pd
from rusket import AutoMiner

# Order line export from your e-commerce backend
orders = pd.DataFrame({
    "order_id": [1001, 1001, 1001, 1002, 1002, 1003, 1003],
    "sku":      ["HDPHONES", "USB_DAC", "AUX_CABLE",
                 "HDPHONES", "CARRY_CASE",
                 "USB_DAC",  "AUX_CABLE"],
})

model = AutoMiner.from_transactions(
    orders,
    transaction_col="order_id",
    item_col="sku",
    min_support=0.3,
)

freq  = model.mine(use_colnames=True)
rules = model.association_rules(metric="confidence", min_threshold=0.6)

# Which accessories should be suggested when headphones are in the cart?
suggestions = model.recommend_items(["HDPHONES"], n=3)
# → e.g. ["USB_DAC", "AUX_CABLE", "CARRY_CASE"]
```

Or use the explicit type variants:

```python
from rusket import AutoMiner

ohe = AutoMiner.from_pandas(orders, transaction_col="order_id", item_col="sku")
ohe = AutoMiner.from_polars(pl_orders, transaction_col="order_id", item_col="sku")
ohe = AutoMiner.from_transactions([["HDPHONES", "USB_DAC"], ["HDPHONES", "CARRY_CASE"]])  # list of lists
```

> **Spark** is also supported: `AutoMiner.from_spark(spark_df)` calls `.toPandas()` internally.

---

### ⚡ Eclat — Large SKU Catalogues

`eclat` uses vertical bitset representation + hardware `popcnt` for fast support counting. Ideal for **large SKU catalogues** where baskets contain only a handful of items out of thousands (low density, typically < 0.15).

```python
import pandas as pd
from rusket import Eclat

# Fashion e-tailer: 5 receipts, basket contains only a subset of the catalogue
baskets = pd.DataFrame({
    "jeans":    [True, True, False, True, True],
    "t_shirt":  [True, False, True,  True, False],
    "sneakers": [True, True, True,  False, True],
    "belt":     [False, True, True,  False, True],
})

# Eclat — same API as AutoMiner, typically faster on sparse catalogues
model = Eclat(baskets, min_support=0.4)
freq  = model.mine(use_colnames=True)
rules = model.association_rules(min_threshold=0.6)
print(rules)
```

#### When to use which?

You almost always want to use `AutoMiner`. This evaluates the density of your dataset `nnz / (rows * cols)` using the [Borgelt heuristic (2003)](https://borgelt.net/doc/eclat/eclat.html) to pick the best algorithm under the hood:

| Scenario | Algorithm chosen by `AutoMiner` |
|---|---|
| Large SKU catalogue, small basket size (density < 0.15) | `Eclat` (bitset/SIMD intersections) |
| Smaller catalogue, dense baskets (density > 0.15) | `FPGrowth` (FP-tree traversals) |

---

### 🐻‍❄️ Polars Input — Reading from Data Lake Parquet

For teams running a modern data stack with Parquet files on S3/GCS/Azure Blob, `rusket` natively accepts [Polars](https://pola.rs/) DataFrames. Data is transferred via Arrow zero-copy buffers — **no conversion overhead**.

The fastest path from a data lake to "Frequently Bought Together" rules:

```python
import polars as pl
from rusket import AutoMiner

# ── 1. Read a one-hot basket matrix directly from S3/GCS/local Parquet ──
# Columns = SKUs (bool), rows = receipts — produced by your dbt or Spark pipeline
baskets = pl.read_parquet("s3://data-lake/gold/basket_ohe.parquet")
print(f"Loaded {baskets.shape[0]:,} receipts × {baskets.shape[1]} SKUs")

# ── 2. Instantiate AutoMiner (zero-copy from Polars) ─────────────────
model = AutoMiner(baskets, min_support=0.02, max_len=3)

# ── 3. Mine frequent combinations ────────────────────────────────────
freq = model.mine(use_colnames=True)
print(f"Found {len(freq):,} frequent itemsets")
print(freq.sort_values("support", ascending=False).head(10))

# ── 4. Generate cross-sell rules ────────────────────────────────────
rules = model.association_rules(metric="lift", min_threshold=1.2)
print(f"Rules with lift > 1.2: {len(rules):,}")
print(
    rules[["antecedents", "consequents", "confidence", "lift"]]
    .sort_values("lift", ascending=False)
    .head(8)
)
```

> **How it works under the hood:**  
> Polars → Arrow buffer → `np.uint8` (zero-copy) → Rust `fpgrowth_from_dense`

---

### 💎 High-Utility Pattern Mining (HUPM) — Profit-Driven Bundle Discovery

Frequent items aren't always the most profitable. HUPM finds product combinations that generate the **highest total gross margin** — even if they appear rarely. `rusket` implements the state-of-the-art **EFIM** algorithm in Rust.

```python
import pandas as pd
from rusket import HUPM

# Specialty foods retailer: receipt line items with gross margin per unit sold
orders = pd.DataFrame({
    "receipt_id": [1, 1, 1, 2, 2, 3, 3],
    "product": ["aged_cheese", "wine_flight", "charcuterie",
                "aged_cheese", "charcuterie",
                "wine_flight", "charcuterie"],
    "margin": [8.50, 12.00, 6.50,   # receipt 1 — margin per item
               8.50, 6.50,           # receipt 2
               12.00, 6.50],         # receipt 3
})

# Find all product bundles generating ≥ €20 total margin across all receipts
high_margin = HUPM.from_transactions(
    orders,
    transaction_col="receipt_id",
    item_col="product",
    utility_col="margin",
    min_utility=20.0,
).mine()
print(high_margin.head())
# e.g. aged_cheese + wine_flight + charcuterie → total margin 81.0
```

---

### 📊 Sparse Pandas Input

For very sparse datasets (e.g. e-commerce with thousands of SKUs), use Pandas `SparseDtype` to minimize memory. `rusket` passes the raw CSR arrays straight to Rust — **no densification ever happens**.

```python
import pandas as pd
import numpy as np
from rusket import AutoMiner

rng = np.random.default_rng(7)
n_rows, n_cols = 30_000, 500

# Very sparse: average basket size ≈ 3 items out of 500
p_buy = 3 / n_cols
matrix = rng.random((n_rows, n_cols)) < p_buy
products = [f"sku_{i:04d}" for i in range(n_cols)]

df_dense = pd.DataFrame(matrix.astype(bool), columns=products)
df_sparse = df_dense.astype(pd.SparseDtype("bool", fill_value=False))

dense_mb = df_dense.memory_usage(deep=True).sum() / 1e6
sparse_mb = df_sparse.memory_usage(deep=True).sum() / 1e6
print(f"Dense  memory: {dense_mb:.1f} MB")
print(f"Sparse memory: {sparse_mb:.1f} MB  ({dense_mb / sparse_mb:.1f}× smaller)")

# Same API, same results — just faster and lighter
freq = AutoMiner(df_sparse, min_support=0.01).mine(use_colnames=True)
print(f"Frequent itemsets: {len(freq):,}")
```

> **How it works under the hood:**  
> Sparse DataFrame → COO → CSR → `(indptr, indices)` → Rust `fpgrowth_from_csr`

---

### 🌊 Out-of-Core Processing (FPMiner Streaming)

For datasets scaling to **Billion-row** sizes that don't fit in memory, use the `FPMiner` accumulator. It accepts chunks of `(txn_id, item_id)` pairs, sorting them in-place immediately, and uses a memory-safe **k-way merge** across all chunks to build the CSR matrix on the fly avoiding massive memory spikes.

```python
import numpy as np
from rusket import FPMiner

n_items = 5_000
miner = FPMiner(n_items=n_items)

# Feed chunks incrementally (e.g. from Parquet/CSV/SQL)
for chunk in dataset:
    txn_ids = chunk["txn_id"].to_numpy(dtype=np.int64)
    item_ids = chunk["item_id"].to_numpy(dtype=np.int32)
    
    # Fast O(k log k) per-chunk sort
    miner.add_chunk(txn_ids, item_ids)

# Stream k-way merge and mine in one pass!
# Returns a DataFrame with 'support' and 'itemsets' just like fpgrowth()
freq = miner.mine(min_support=0.001, max_len=3)
```

**Memory efficiency:** The peak memory overhead at `mine()` time is just $O(k)$ for the cursors (where $k$ is the number of chunks), plus the final compressed CSR allocation. 

---

### 🌩️ Distributed Computing with Apache Spark

`rusket` ships a full Spark integration layer in `rusket.spark`. All algorithms run as **Native Arrow UDFs** via `applyInArrow` — Rust is called directly on each executor, with zero Python overhead per row.

#### How it works

```
PySpark DataFrame
  └─► groupby(group_col).applyInArrow(...)
        └─► Arrow Table (per partition / per group)
              └─► Polars zero-copy conversion
                    └─► rusket Rust extension (on the executor)
                          └─► results → PyArrow → PySpark DataFrame
```

#### Full Example — Retail Basket Analysis per Store

```python
from pyspark.sql import SparkSession
from rusket.spark import mine_grouped, rules_grouped

spark = SparkSession.builder.appName("rusket-demo").getOrCreate()

# ── 1. Load your OHE transaction table (one row = one basket) ──────────────
#    Schema: store_id (string), bread (bool), butter (bool), milk (bool), ...
spark_df = spark.read.parquet("s3://data/baskets/")

# ── 2. Mine frequent itemsets per store in parallel ──────────────────────────
#    Each Spark task calls the Rust FP-Growth/Eclat engine on its Arrow batch.
freq_df = mine_grouped(
    spark_df,
    group_col="store_id",
    min_support=0.05,    # 5% support per store
    method="auto",       # auto-selects FP-Growth or Eclat
)
# freq_df schema: store_id | support (double) | itemsets (array<string>)

# ── 3. Count transactions per store (needed for rule support) ────────────────
from pyspark.sql import functions as F
counts = (
    spark_df.groupby("store_id")
    .agg(F.count("*").alias("n"))
    .rdd.collectAsMap()          # {"store_1": 12000, "store_2": 8500, ...}
)

# ── 4. Generate association rules per store ──────────────────────────────────
rules_df = rules_grouped(
    freq_df,
    group_col="store_id",
    num_itemsets=counts,         # pass per-group counts as a dict
    metric="confidence",
    min_threshold=0.6,
)
# rules_df schema: store_id | antecedents | consequents | confidence | lift | ...

rules_df.orderBy("lift", ascending=False).show(10, truncate=False)
```

#### Sequential Patterns per Category

```python
from rusket.spark import prefixspan_grouped

# event_log schema: category_id, user_id, item_id, event_ts
event_log = spark.read.parquet("s3://data/events/")

seq_df = prefixspan_grouped(
    event_log,
    group_col="category_id",   # mine independently per product category
    user_col="user_id",        # sequence identifier within the group
    time_col="event_ts",       # ordering column
    item_col="item_id",
    min_support=50,            # absolute count: pattern must appear in ≥50 sessions
    max_len=4,
)
# seq_df schema: category_id | support (long) | sequence (array<string>)
seq_df.show(5, truncate=False)
```

#### High-Utility Patterns per Region

```python
from rusket.spark import hupm_grouped

# profit_log schema: region_id, txn_id, item_id, profit
profit_log = spark.read.parquet("s3://data/profit/")

utility_df = hupm_grouped(
    profit_log,
    group_col="region_id",
    transaction_col="txn_id",
    item_col="item_id",
    utility_col="profit",
    min_utility=500.0,         # only itemsets with combined profit ≥ €500
)
# utility_df schema: region_id | utility (double) | itemset (array<long>)
utility_df.show(5, truncate=False)
```

#### Batch Recommendations across the Cluster

```python
from rusket.spark import recommend_batches
from rusket import ALS

# 1. Train an ALS model locally (or load a pre-trained one)
als = ALS(factors=64, iterations=15).from_transactions(
    events_pd,
    user_col="user_id",
    item_col="item_id",
)

# 2. Scale-out scoring: one recommendation row per user
user_df = spark.read.parquet("s3://data/users/").select("user_id")

recs_df = recommend_batches(user_df, model=als, user_col="user_id", k=10)
# recs_df schema: user_id (string) | recommended_items (array<int>)
recs_df.show(5, truncate=False)
```

> **Tip — Databricks / Delta Lake:** All functions return a standard PySpark DataFrame, so you can write results back with `.write.format("delta").save(...)` or `.saveAsTable(...)` directly.

---

## 📖 API Reference

### OOP Class API

Every algorithm in `rusket` exposes a **class-based API** in addition to the functional helpers. All classes share a unified interface inherited from `BaseModel`:

| Class | Inherits from | Description |
|-------|--------------|-------------|
| `FPGrowth` | `Miner`, `RuleMinerMixin` | FP-Tree parallel mining |
| `Eclat` | `Miner`, `RuleMinerMixin` | Vertical bitset mining |
| `AutoMiner` | `Miner`, `RuleMinerMixin` | Auto-selects FP-Growth or Eclat |
| `HUPM` | `Miner` | High-Utility Pattern Mining (EFIM) |
| `PrefixSpan` | `Miner` | Sequential pattern mining |
| `ALS` | `ImplicitRecommender` | Alternating Least Squares CF |
| `BPR` | `ImplicitRecommender` | Bayesian Personalized Ranking CF |
| `SVD` | `ImplicitRecommender` | Funk SVD (biased SGD) |
| `LightGCN` | `ImplicitRecommender` | Graph Convolutional CF |
| `ItemKNN` | `ImplicitRecommender` | Item-based k-NN CF |
| `EASE` | `ImplicitRecommender` | Embarrassingly Shallow Autoencoders |

All classes share the following data-ingestion class methods inherited from `BaseModel`:

```python
# Load from long-format (transaction_id, item_id) DataFrame or list of lists
model = FPGrowth.from_transactions(df, transaction_col="order_id", item_col="item", min_support=0.3)

# Typed convenience aliases — same result
model = FPGrowth.from_pandas(df,  ...)
model = FPGrowth.from_polars(pl_df, ...)
model = FPGrowth.from_spark(spark_df, ...)
```

`Miner` subclasses (`FPGrowth`, `Eclat`, `AutoMiner`) additionally expose `RuleMinerMixin`, giving a fluent pipeline:

```python
model  = AutoMiner.from_transactions(df, min_support=0.3)
freq   = model.mine(use_colnames=True)             # pd.DataFrame [support, itemsets]
rules  = model.association_rules(metric="lift")    # pd.DataFrame [antecedents, consequents, ...]
recs   = model.recommend_items(["bread", "milk"])  # list of suggested items
```

`ImplicitRecommender` subclasses (`ALS`, `BPR`) expose:

```python
model = ALS(factors=64, iterations=15).fit(user_item_csr)
# — or directly from an event log —
model = ALS(factors=64).from_transactions(df, user_col="user_id", item_col="item_id")

items, scores = model.recommend_items(user_id=42, n=10, exclude_seen=True)
users, scores = model.recommend_users(item_id=99, n=5)
```



## 🧠 Advanced Pattern & Recommendation Algorithms

`rusket` provides more than just basic market basket analysis. It includes an entire suite of modern algorithms and a high-level Business Recommender API.

### 🎯 ALS & BPR Collaborative Filtering

Both models learn user and item embeddings from **implicit feedback** (purchases, clicks, plays) and power personalised recommendations at scale. Use **ALS** for broad serendipitous discovery; use **BPR** when you care only about top-N ranking.

```python
from rusket import ALS, BPR

# ── "For You" homepage — music streaming platform ────────────────────
# event log: user_id | track_id | plays (optional weight)
plays = pd.DataFrame({
    "user_id":  [101, 101, 102, 102, 103, 103, 103],
    "track_id": ["T01", "T03", "T01", "T05", "T02", "T03", "T05"],
    "plays":    [12, 5, 8, 3, 20, 1, 7],  # play count as confidence weight
})

als = ALS(factors=64, iterations=15, alpha=40.0).from_transactions(
    plays, user_col="user_id", item_col="track_id", rating_col="plays"
)

# Top-10 tracks for user 101, excluding already-played tracks
tracks, scores = als.recommend_items(user_id=101, n=10, exclude_seen=True)

# Which users are most likely to enjoy track T05? — useful for email campaigns
users, scores = als.recommend_users(item_id="T05", n=50)

# BPR — optimise ranking directly rather than reconstruction
bpr = BPR(factors=64, learning_rate=0.05, iterations=150).fit(user_item_csr)
```

### 🎯 Hybrid Recommender API

Combine **Collaborative Filtering** (ALS/BPR) with **Frequent Pattern Mining** to cover every placement surface — personalised homepage ("For You") and active cart ("Frequently Bought Together") — in a single engine.

```python
from rusket import ALS, Recommender, AutoMiner

# 1. Train on purchase history (implicit feedback)
als = ALS(factors=64, iterations=15).fit(user_item_csr)

# 2. Mine co-purchase rules from basket data
miner = AutoMiner(basket_ohe, min_support=0.01)
freq  = miner.mine()
rules = miner.association_rules()

# 3. Create the Hybrid Engine
rec = Recommender(model=als, rules_df=rules)

# "For You" homepage — personalised for customer 1001
items, scores = rec.recommend_for_user(user_id=1001, n=5)

# Blend CF + product embeddings (e.g. from a PIM or sentence-transformer)
items, scores = rec.recommend_for_user(user_id=1001, n=5, alpha=0.7,
                                       target_item_for_semantic="HDPHONES")

# Active cart cross-sell — "Frequently Bought Together"
add_ons = rec.recommend_for_cart(["USB_DAC", "AUX_CABLE"], n=3)

# Overnight batch — score all customers, write to CRM
batch_df = rec.predict_next_chunk(user_history_df, user_col="customer_id", k=5)
```

### 🔍 Analytics Helpers

```python
from rusket import find_substitutes, customer_saturation

# Identify cannibalizing SKUs (lift < 1.0) for assortment rationalisation
subs = find_substitutes(rules_df, max_lift=0.8)
#  antecedents  consequents  lift
#  (Cola A,)    (Cola B,)    0.61   ← these products hurt each other's sales

# Segment customers by category penetration (decile 10 = buy everything; 1 = barely engaged)
saturation = customer_saturation(
    purchases_df, user_col="customer_id", category_col="category_id"
)
```

### 📈 BPR & Sequential Patterns

- **BPR (Bayesian Personalized Ranking):** Directly optimises ranking of positive interactions over negative ones — ideal for newsfeeds, playlists, and app recommendation surfaces that prioritise top-N precision.
- **Sequential Pattern Mining (PrefixSpan):** Discovers ordered patterns across time (e.g., "Subscriber signed up for broadband → mobile plan → premium bundle" or "Customer viewed Camera → 2 weeks later bought Lens"). 

`rusket` natively extracts PrefixSpan sequences from **Pandas, Polars, and PySpark** event logs with zero-copy Arrow mapping:

```python
from rusket import PrefixSpan

# Telco product adoption journeys — what sequence of subscriptions do customers follow?
# df: customer_id | subscription_date | product_id
model = PrefixSpan.from_transactions(
    subscription_events,
    transaction_col="customer_id",
    item_col="product_id",
    time_col="subscription_date",
    min_support=50,    # at least 50 customers follow this path
    max_len=4,
)
freq_seqs = model.mine()
# e.g. [broadband] → [mobile] → [tv_bundle] appears in 312 journeys
```



### 🕸️ Graph Analytics & Embeddings

Integrate natively with the modern GenAI/LLM stack:

- **Vector Export:** Export user/item factors to a Pandas `DataFrame` ready for FAISS/Qdrant using `model.export_item_factors()`.
- **Item-to-Item Similarity:** Fast Cosine Similarity on embeddings using `model.similar_items(item_id)`.
- **Graph Generation:** Automatically convert association rules into a `networkx` directed Graph for community detection using `rusket.viz.to_networkx(rules)`.

---

## ⚡ Benchmarks

> **Benchmark environment:** Apple Silicon MacBook Air (M-series, arm64, 8 GB RAM). All timings are single-run wall-clock measurements.

### Scale Benchmarks (1M → 200M rows)

> **What's measured:** `from_transactions()` converts long-format `(txn_id, item_id)` rows into a sparse OHE matrix. `fpgrowth()` then mines that matrix. Both steps have the same Rust mining cost — the only difference at large scale is whether you pay the conversion cost upfront.

| Scale | `from_transactions` (conversion) | `fpgrowth` (mining) | **Total** |
|---|:---:|:---:|:---:|
| 1M rows | 4.9s | **0.1s** | **5.0s** |
| 10M rows | 23.2s | **1.2s** | **24.4s** |
| 50M rows | 59.1s | **4.0s** | **63.1s** |
| 100M rows (20M txns × 200k items) | 124.1s | **10.1s** | **134.2s** |
| **200M rows** (40M txns × 200k items) | 229.2s | **17.6s** | **246.8s** |

The mining step is fast — the bottleneck at scale is the long-format → sparse-matrix conversion. If your pipeline already produces a CSR/sparse matrix (e.g., from a Parquet/warehouse export), you skip the conversion entirely and only pay the mining cost.

#### Power-user path: Direct CSR → Rust

```python
import numpy as np
from scipy import sparse as sp
from rusket import AutoMiner

# Build CSR directly from integer IDs (no pandas!)
csr = sp.csr_matrix(
    (np.ones(len(txn_ids), dtype=np.int8), (txn_ids, item_ids)),
    shape=(n_transactions, n_items),
)
freq = AutoMiner(csr, item_names=item_names).mine(
    min_support=0.001, max_len=3, use_colnames=True
)
```

> At 100M rows, the mining step itself takes **10.1 seconds**. Building the CSR directly skips the `from_transactions` conversion cost (~124s) but does not change the mining time.

### Real-World Datasets

| Dataset | Transactions | Items | `rusket` |
|---------|:----------:|:-----:|:--------:|
| [andi_data.txt](https://github.com/andi611/Apriori-and-Eclat-Frequent-Itemset-Mining) | 8,416 | 119 | **9.7 s** (22.8M itemsets) |
| [andi_data2.txt](https://github.com/andi611/Apriori-and-Eclat-Frequent-Itemset-Mining) | 540,455 | 2,603 | **7.9 s** |

Run benchmarks yourself:

```bash
uv run pytest benchmarks/bench_scale.py -v -s   # Scale benchmark
uv run python benchmarks/bench_realworld.py     # Real-world datasets
uv run pytest tests/test_benchmark.py -v -s      # pytest-benchmark
```

### Recommender Benchmarks vs LibRecommender

> **Measured with `pytest-benchmark`** (5 rounds, warmed up, GC disabled). MovieLens 100k dataset (943 users, 1,682 items, 100k ratings). Only `model.fit()` is timed — no startup or data loading overhead.

| Benchmark | rusket | LibRecommender | **Speedup** |
|---|:---:|:---:|:---:|
| **ALS** (64 factors, 15 epochs) | **427 ms** | 1,324 ms | **3.1×** |
| **BPR** (64 factors, 10 epochs) | **33 ms** | 681 ms | **20.4×** |
| **ItemKNN** (k=100) | **55 ms** | 287 ms | **5.2×** |
| **SVD** (64 factors, 20 epochs) | **55 ms** | ❌ TF-only (broken) | — |
| **EASE** | **71 ms** | *N/A* | — |

> **Note:** LibRecommender requires TensorFlow + PyTorch + gensim + Cython (~2 GB of dependencies). rusket has **zero runtime dependencies**.

```bash
uv run pytest benchmarks/bench_pytest_librecommender.py -v --benchmark-columns=mean,stddev,rounds
```

---

## 🏗 Architecture

### Data Flow

```
pandas dense         ──► np.uint8 array (C-contiguous)  ──► Rust fpgrowth_from_dense
pandas Arrow backend ──► Arrow → np.uint8 (zero-copy)   ──► Rust fpgrowth_from_dense
pandas sparse        ──► CSR int32 arrays               ──► Rust fpgrowth_from_csr
polars               ──► Arrow → np.uint8 (zero-copy)   ──► Rust fpgrowth_from_dense
numpy ndarray        ──► np.uint8 (C-contiguous)        ──► Rust fpgrowth_from_dense
```

All mining and rule generation happens **inside Rust**. No Python loops, no round-trips.

### The 1 Billion Row Architecture

To pass the "1 Billion Row" threshold without OOM crashes, `rusket` employs a zero-allocation mining loop:
- **Eclat Scratch Buffers:** `intersect_count_into` writes intersections directly into thread-local pre-allocated memory bytes and computes `popcnt` in a single pass. It implements **early-exit** loop termination the moment it proves a combination cannot reach `min_support`.
- **FPGrowth Parallel Tree Build:** Conditional FP-trees are collected concurrently inside the rayon parallel mining step, replacing the standard sequential loop and eliminating memory contention bottlenecks.
- **`AHashMap` Deduplication:** Extremely fast O(N) duplicate basket counting replaces standard O(N log N) unstable sorts in the core pipeline.


---

## 🧑‍💻 Development

### Prerequisites

- **Rust** 1.83+ (`rustup update`)
- **Python** 3.10+
- [**uv**](https://docs.astral.sh/uv/) (recommended package manager)

### Getting Started

```bash
# Clone
git clone https://github.com/bmsuisse/rusket.git
cd rusket

# Build Rust extension in dev mode
uv run maturin develop --release

# Run the full test suite
uv run pytest tests/ -x -q

# Type-check the Python layer
uv run pyright rusket/

# Cargo check (Rust)
cargo check
```

### Run Examples

```bash
# Getting started
uv run python examples/01_getting_started.py

# Market basket analysis with Faker
uv run python examples/02_market_basket_faker.py

# Polars input
uv run python examples/03_polars_input.py

# Sparse input
uv run python examples/04_sparse_input.py

# Large-scale mining (100k+ rows)
uv run python examples/05_large_scale.py

```

---

## 🤖 AI Disclosure

A large part of this library — including the Rust core algorithms, the Python wrappers, the OOP class hierarchy, and the Spark integration layer — was written with substantial assistance from **AI pair-programming tools** (specifically [Google Gemini / Antigravity](https://deepmind.google/technologies/gemini/)). Human review, benchmarking, and architectural decisions were applied throughout.

We believe in transparency about AI-assisted development. The algorithms are correct, the tests pass, and the performance numbers are real — but if you find a bug or a piece of "AI slop", please open an issue!

---

## 📜 License

[MIT License](LICENSE)
