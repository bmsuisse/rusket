<p align="center">
  <img src="assets/logo_single.svg" alt="rusket logo" width="320" />
</p>

# rusket
> Ultra-fast Recommender Engines & Market Basket Analysis for Python, written in Rust.
> 
> *Made with ❤️ by the Data & AI Team.*

[![PyPI](https://img.shields.io/pypi/v/rusket.svg)](https://pypi.org/project/rusket/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://pypi.org/project/rusket/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

---

## What is rusket?

`rusket` turns raw transaction logs into **revenue intelligence** — "frequently bought together" rules, personalised recommendations, high-profit bundle discovery, and sequential customer journey analysis.

The core algorithms run entirely in **Rust** (via [PyO3](https://pyo3.rs)) and accept Pandas, Polars, and Spark DataFrames natively with zero-copy Arrow transfers:

| Input | Rust path | Notes |
|---|---|---|
| Dense pandas DataFrame | `fpgrowth_from_dense` | Flat `uint8` buffer — zero-copy |
| Sparse pandas DataFrame | `fpgrowth_from_csr` | Raw CSR arrays — zero-copy |
| Polars DataFrame | `fpgrowth_from_dense` | Arrow-backed `numpy` buffer |

## 🎯 Goals

| Goal | Details |
|---|---|
| ⚡ **Blazing fast** | Compiled Rust with Rayon multi-threading and SIMD kernels. ALS is **3×**, BPR **20×** faster than Python equivalents. |
| 📦 **Zero dependencies** | No TensorFlow, no PyTorch, no JVM — just `pip install rusket` and go. A single ~3 MB wheel. |
| 🧑‍💻 **Easy to use** | Common cases are one-liners: `model.recommend_items(user_id)`, `model.recommend_users(item_id)`, `model.export_item_factors()` for vector/embedding export. No boilerplate. |
| 🏗️ **Modern data stack** | Native Pandas, Polars, and Apache Spark support with zero-copy Arrow. Works with Delta Lake, Databricks, Snowflake, and dbt/Parquet pipelines out of the box. |

## Why rusket?

**Zero runtime dependencies.** No TensorFlow, no PyTorch, no JVM — just `pip install rusket`. The entire engine is compiled Rust (~3 MB wheel).

| Feature | rusket | LibRecommender |
|---|---|---|
| **Runtime deps** | **0** | TF + PyTorch + gensim (~2 GB) |
| **ALS fit (ML-100k)** | **427 ms** | 1,324 ms (3.1× slower) |
| **BPR fit (ML-100k)** | **33 ms** | 681 ms (20.4× slower) |
| **ItemKNN fit (ML-100k)** | **55 ms** | 287 ms (5.2× slower) |
| Polars / Spark support | ✅ / ✅ | ❌ / ❌ |
| Pattern Mining | FP-Growth, Eclat, HUPM, PrefixSpan | ❌ |

> Benchmarks: `pytest-benchmark`, 5 rounds, warmed up, GC disabled. MovieLens 100k.

## Quick Example — "Frequently Bought Together"

```python
import pandas as pd
from rusket import FPGrowth

receipts = pd.DataFrame({
    "milk":    [1, 1, 0, 1, 0, 1],
    "bread":   [1, 0, 1, 1, 1, 0],
    "butter":  [1, 0, 1, 0, 0, 1],
    "eggs":    [0, 1, 1, 0, 1, 1],
    "coffee":  [0, 1, 0, 0, 1, 1],
}, dtype=bool)

model = FPGrowth(receipts, min_support=0.4)
freq = model.mine(use_colnames=True)
rules = model.association_rules(metric="confidence", min_threshold=0.6)
print(rules[["antecedents", "consequents", "confidence", "lift"]]
      .sort_values("lift", ascending=False).to_markdown(index=False))
```

<!-- output -->

| antecedents   | consequents   |   confidence |   lift |
|:--------------|:--------------|-------------:|-------:|
| ('coffee',)   | ('eggs',)     |         1    |    1.5 |
| ('eggs',)     | ('coffee',)   |         0.75 |    1.5 |
<!-- /output -->

## Quick Example — Personalised Recommendations

Collaborative filtering with Implicit Feedback (e.g. tracking clicks, views, or purchase amounts) is extremely fast with the SIMD-accelerated ALS and BPR engines.

```python
import pandas as pd
from rusket import ALS

purchases = pd.DataFrame({
    "customer_id": [1001, 1001, 1001, 1002, 1002, 1003],
    "sku":         ["A10", "B22", "C15",  "A10", "D33",  "B22"],
    "revenue":     [29.99, 49.00, 9.99,  29.99, 15.00, 49.00],
})

# Fit an Implicit ALS model in milliseconds
model = ALS(factors=64, iterations=15, alpha=40.0)
model.fit(purchases, user_col="customer_id", item_col="sku", rating_col="revenue")

# Get top-3 recommendations for a customer
items, scores = model.recommend_items(user_id=1001, n=3, exclude_seen=True)
print(items)
```

<!-- output -->
```text
['D33']
```
<!-- /output -->

## Built for the Modern Data Stack

`rusket` natively accepts Polars and Apache Spark DataFrames, backing straight into Rust memory with zero-copy Arrow buffers.

```python
import polars as pl
from rusket import fpgrowth

# 10M+ rows load instantly with Polars
df = pl.scan_parquet("s3://datalake/market_baskets/*.parquet").collect()

# Native execution on Arrow buffers, no pandas fallback
freq = fpgrowth(df, min_support=0.01, use_colnames=True)
```

---

- [Get Started](quickstart.md) — Install rusket and run your first analysis in minutes
- [API Reference](api-reference.md) — Full parameter documentation for all functions
- [GitHub](https://github.com/bmsuisse/rusket) — Source code, issues, and contributions
