<p align="center">
  <img src="https://raw.githubusercontent.com/bmsuisse/rusket/main/docs/assets/logo.svg" alt="rusket logo" width="200" height="200" />
</p>

<p align="center">
  <strong>Blazing-fast FP-Growth, Eclat, and Association Rules for Python, powered by Rust.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/rusket/"><img src="https://img.shields.io/pypi/v/rusket?color=%2334D058&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.83%2B-orange?logo=rust" alt="Rust"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-green" alt="License"></a>
  <a href="https://bmsuisse.github.io/rusket/"><img src="https://img.shields.io/badge/docs-MkDocs-blue" alt="Docs"></a>
</p>

---

`rusket` is a **drop-in replacement** for [`mlxtend`](https://rasbt.github.io/mlxtend/)'s `fpgrowth` and `association_rules` — backed by a **Rust core** (via [PyO3](https://pyo3.rs/)) that delivers **2–15× speed-ups** and dramatically lower memory usage. It includes both **FP-Growth** (parallel via Rayon) and **Eclat** (vertical bitset mining) algorithms. Natively supports **Pandas** (including Arrow backend), **Polars**, and **sparse DataFrames** out of the box.

---

## ✨ Highlights

| | `rusket` | `mlxtend` |
|---|---|---|
| **Core language** | Rust (PyO3) | Pure Python |
| **Algorithms** | FP-Growth + Eclat | FP-Growth only |
| **Pandas dense input** | ✅ C-contiguous `np.uint8` | ✅ |
| **Pandas Arrow backend** | ✅ Arrow zero-copy (pandas 2.0+) | ❌ Not supported |
| **Pandas sparse input** | ✅ Zero-copy CSR → Rust | ❌ Densifies first |
| **Polars input** | ✅ Arrow zero-copy | ❌ Not supported |
| **Parallel mining** | ✅ Rayon work-stealing | ❌ Single-threaded |
| **Memory** | Low (native Rust buffers) | High (Python objects) |
| **API compatibility** | ✅ Drop-in replacement | — |
| **Metrics** | 12 built-in metrics | 9 |

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

### Basic — Pandas

```python
import pandas as pd
from rusket import fpgrowth, association_rules

# One-hot encoded boolean DataFrame
data = {
    "bread":  [1, 1, 0, 1, 1],
    "butter": [1, 0, 1, 1, 0],
    "milk":   [1, 1, 1, 0, 1],
    "eggs":   [0, 1, 1, 0, 1],
    "cheese": [0, 0, 1, 0, 0],
}
df = pd.DataFrame(data).astype(bool)

# 1. Mine frequent itemsets
freq = fpgrowth(df, min_support=0.4, use_colnames=True)

# 2. Generate association rules
rules = association_rules(
    freq,
    num_itemsets=len(df),
    metric="confidence",
    min_threshold=0.6,
)

print(rules[["antecedents", "consequents", "support", "confidence", "lift"]]
      .sort_values("lift", ascending=False))
```

---

### 🛒 Transaction Data (Long Format)

Real-world data comes as `(transaction_id, item)` rows — not one-hot matrices. Use the built-in helpers to convert:

```python
import pandas as pd
from rusket import from_transactions, fpgrowth

# Long-format transactional data
df = pd.DataFrame({
    "order_id": [1, 1, 1, 2, 2, 3],
    "item":     [3, 4, 5, 3, 5, 8],
})

# Convert to one-hot boolean matrix
ohe = from_transactions(df)

# Mine!
freq = fpgrowth(ohe, min_support=0.3, use_colnames=True)
print(freq)
```

Or use the explicit helpers for type clarity:

```python
from rusket import from_pandas, from_polars

ohe = from_pandas(df)                      # Pandas DataFrame
ohe = from_polars(pl_df)                   # Polars DataFrame
ohe = from_transactions([[3, 4], [3, 5]])  # list of lists
```

> **Spark** is also supported: `from_spark(spark_df)` calls `.toPandas()` internally.

---

### ⚡ Eclat — Vertical Mining

`eclat` uses vertical bitset representation + hardware `popcnt` for fast support counting. Ideal for **sparse retail basket** data.

```python
import pandas as pd
from rusket import eclat, association_rules

df = pd.DataFrame({
    "bread":  [True, True, False, True, True],
    "butter": [True, False, True, True, False],
    "milk":   [True, True, True, False, True],
    "eggs":   [False, True, True, False, True],
})

# Eclat — same API as fpgrowth
freq = eclat(df, min_support=0.4, use_colnames=True)
rules = association_rules(freq, num_itemsets=len(df), min_threshold=0.6)
print(rules)
```

#### When to use which?

| Scenario | Recommended |
|---|---|
| Very sparse data (retail baskets, e-commerce) | `eclat` or `fpgrowth` — both fast |
| Dense data (many items per transaction) | `fpgrowth` |
| General-purpose / don't know | `fpgrowth` (default) |

---

### 🐻‍❄️ Polars Input

`rusket` natively accepts [Polars](https://pola.rs/) DataFrames. Data is transferred via Arrow zero-copy buffers — **no conversion overhead**.

```python
import polars as pl
import numpy as np
from rusket import fpgrowth, association_rules

# ── 1. Create a Polars DataFrame ────────────────────────────────────
rng = np.random.default_rng(0)
n_rows, n_cols = 20_000, 150
products = [f"product_{i:03d}" for i in range(n_cols)]

# Power-law popularity: top products appear often, tail products are rare
support = np.clip(0.5 / np.arange(1, n_cols + 1, dtype=float) ** 0.5, 0.005, 0.5)
matrix = rng.random((n_rows, n_cols)) < support

df_pl = pl.DataFrame({p: matrix[:, i].tolist() for i, p in enumerate(products)})
print(f"Polars DataFrame: {df_pl.shape[0]:,} rows × {df_pl.shape[1]} columns")

# ── 2. fpgrowth — same API as pandas ────────────────────────────────
freq = fpgrowth(df_pl, min_support=0.05, use_colnames=True)
print(f"Frequent itemsets: {len(freq):,}")
print(freq.sort_values("support", ascending=False).head(8))

# ── 3. Association rules ────────────────────────────────────────────
rules = association_rules(freq, num_itemsets=n_rows, metric="lift", min_threshold=1.1)
print(f"Rules: {len(rules):,}")
print(
    rules[["antecedents", "consequents", "confidence", "lift"]]
    .sort_values("lift", ascending=False)
    .head(6)
)
```

Or more concisely — just read a Parquet file:

```python
import polars as pl
from rusket import fpgrowth

df = pl.read_parquet("transactions.parquet")
freq = fpgrowth(df, min_support=0.05, use_colnames=True)
```

> **How it works under the hood:**  
> Polars → Arrow buffer → `np.uint8` (zero-copy) → Rust `fpgrowth_from_dense`

---

### 📊 Sparse Pandas Input

For very sparse datasets (e.g. e-commerce with thousands of SKUs), use Pandas `SparseDtype` to minimize memory. `rusket` passes the raw CSR arrays straight to Rust — **no densification ever happens**.

```python
import pandas as pd
import numpy as np
from rusket import fpgrowth

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
freq = fpgrowth(df_sparse, min_support=0.01, use_colnames=True)
print(f"Frequent itemsets: {len(freq):,}")
```

> **How it works under the hood:**  
> Sparse DataFrame → COO → CSR → `(indptr, indices)` → Rust `fpgrowth_from_csr`

---

### 🔄 Migrating from mlxtend

`rusket` is a **drop-in replacement**. The only API difference is `num_itemsets`:

```diff
- from mlxtend.frequent_patterns import fpgrowth, association_rules
+ from rusket import fpgrowth, association_rules

  freq  = fpgrowth(df, min_support=0.05, use_colnames=True)        # identical

- rules = association_rules(freq, metric="lift", min_threshold=1.2)
+ rules = association_rules(freq, num_itemsets=len(df),             # ← add this
+                           metric="lift", min_threshold=1.2)
```

> **Why `num_itemsets`?** This makes support calculation explicit and avoids a hidden internal pandas join that `mlxtend` performs.

**Gotchas:**
1. Input must be `bool` or `0/1` integers — `rusket` warns if you pass floats
2. Polars is supported natively — just pass the DataFrame directly
3. Sparse pandas DataFrames work too — and use much less RAM

---

## 📖 API Reference

### `fpgrowth`

```python
rusket.fpgrowth(
    df,
    min_support: float = 0.5,
    null_values: bool = False,
    use_colnames: bool = False,
    max_len: int | None = None,
    verbose: int = 0,
) -> pd.DataFrame
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` \| `pl.DataFrame` \| `np.ndarray` | One-hot encoded input (bool / 0-1). Dense, sparse, or Polars. |
| `min_support` | `float` | Minimum support threshold in `(0, 1]`. |
| `null_values` | `bool` | Allow NaN values in `df` (pandas only). |
| `use_colnames` | `bool` | Return column names instead of integer indices in itemsets. |
| `max_len` | `int \| None` | Maximum itemset length. `None` = unlimited. |
| `verbose` | `int` | Verbosity level (kept for API compatibility with mlxtend). |

**Returns** a `pd.DataFrame` with columns `['support', 'itemsets']`.

---

### `eclat`

```python
rusket.eclat(
    df,
    min_support: float = 0.5,
    null_values: bool = False,
    use_colnames: bool = False,
    max_len: int | None = None,
    verbose: int = 0,
) -> pd.DataFrame
```

Same parameters as `fpgrowth`. Uses vertical bitset representation (Eclat algorithm) instead of FP-Tree.

**Returns** a `pd.DataFrame` with columns `['support', 'itemsets']`.

---

### `association_rules`

```python
rusket.association_rules(
    df,
    num_itemsets: int,
    metric: str = "confidence",
    min_threshold: float = 0.8,
    support_only: bool = False,
    return_metrics: list[str] = [...],  # all 12 metrics by default
) -> pd.DataFrame
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Output from `fpgrowth()`. |
| `num_itemsets` | `int` | Number of transactions in the original dataset (`len(df_original)`). |
| `metric` | `str` | Metric to filter rules on (see table below). |
| `min_threshold` | `float` | Minimum value of `metric` to include a rule. |
| `support_only` | `bool` | Only compute support; fill other columns with `NaN`. |
| `return_metrics` | `list[str]` | Subset of metrics to include in the result. |

**Returns** a `pd.DataFrame` with columns `antecedents`, `consequents`, plus all requested metric columns.

#### Available Metrics

| Metric | Formula / Description |
|--------|----------------------|
| `support` | P(A ∪ B) |
| `confidence` | P(B \| A) |
| `lift` | confidence / P(B) |
| `leverage` | support − P(A)·P(B) |
| `conviction` | (1 − P(B)) / (1 − confidence) |
| `zhangs_metric` | Symmetrical correlation measure |
| `jaccard` | Jaccard similarity between A and B |
| `certainty` | Certainty factor |
| `kulczynski` | Average of P(B\|A) and P(A\|B) |
| `representativity` | Rule coverage across transactions |
| `antecedent support` | P(A) |
| `consequent support` | P(B) |

---

### `from_transactions`

```python
rusket.from_transactions(
    data,
    transaction_col: str | None = None,
    item_col: str | None = None,
) -> pd.DataFrame
```

Converts long-format transactional data to a one-hot boolean matrix. Accepts Pandas DataFrames, Polars DataFrames, Spark DataFrames, or `list[list[...]]`.

### `from_pandas` / `from_polars` / `from_spark`

Explicit typed variants of `from_transactions` for specific DataFrame types:

```python
rusket.from_pandas(df, transaction_col=None, item_col=None) -> pd.DataFrame
rusket.from_polars(df, transaction_col=None, item_col=None) -> pd.DataFrame
rusket.from_spark(df, transaction_col=None, item_col=None)  -> pd.DataFrame
```

---

## ⚡ Benchmarks

### Scale Benchmarks (1M → 200M rows)

| Scale | `from_transactions` → fpgrowth | Direct CSR → Rust | **Speedup** |
|---|:---:|:---:|:---:|
| 1M rows | 5.0s | **0.1s** | **50×** |
| 10M rows | 24.4s | **1.7s** | **14×** |
| 50M rows | 63.1s | **10.9s** | **6×** |
| 100M rows (20M txns × 200k items) | 134.2s | **25.9s** | **5×** |
| 200M rows (40M txns × 200k items) | 246.8s | **73.1s** | **3×** |

#### Power-user path: Direct CSR → Rust

```python
import numpy as np
from scipy import sparse as sp
from rusket import fpgrowth

# Build CSR directly from integer IDs (no pandas!)
csr = sp.csr_matrix(
    (np.ones(len(txn_ids), dtype=np.int8), (txn_ids, item_ids)),
    shape=(n_transactions, n_items),
)
freq = fpgrowth(csr, min_support=0.001, max_len=3,
                use_colnames=True, column_names=item_names)
```

> At 100M rows, the mining step takes **1.3 seconds** — the bottleneck is entirely the CSR build.

### Real-World Datasets

| Dataset | Transactions | Items | `rusket` | `mlxtend` | Speedup |
|---------|:----------:|:-----:|:--------:|:---------:|:-------:|
| [andi_data.txt](https://github.com/andi611/Apriori-and-Eclat-Frequent-Itemset-Mining) | 8,416 | 119 | **9.7 s** (22.8M itemsets) | **TIMEOUT** 💥 | ∞ |
| [andi_data2.txt](https://github.com/andi611/Apriori-and-Eclat-Frequent-Itemset-Mining) | 540,455 | 2,603 | **7.9 s** | 16.2 s | **2×** |

Run benchmarks yourself:

```bash
uv run python benchmarks/bench_scale.py       # Scale benchmark + Plotly chart
uv run python benchmarks/bench_realworld.py   # Real-world datasets
uv run pytest tests/test_benchmark.py -v -s   # pytest-benchmark
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

### Project Structure

```
├── src/                          # Rust core (PyO3)
│   ├── lib.rs                    # Module root & Python bindings
│   ├── fpgrowth.rs               # FP-Tree construction + FP-Growth mining (Rayon parallel)
│   ├── eclat.rs                  # Eclat vertical mining (bitset intersection + popcnt)
│   └── association_rules.rs      # Rule generation + 12 metrics (Rayon parallel)
│
├── rusket/                       # Python wrappers & validation
│   ├── __init__.py               # Package root
│   ├── fpgrowth.py               # FP-Growth input dispatch (dense / sparse / Polars)
│   ├── eclat.py                  # Eclat input dispatch (dense / sparse / Polars)
│   ├── association_rules.py      # Label mapping + Rust call + result assembly
│   ├── transactions.py           # from_transactions / from_pandas / from_polars / from_spark
│   ├── _validation.py            # Input validation
│   └── _rusket.pyi               # Type stubs for Rust extension
│
├── tests/                        # Comprehensive test suite
├── benchmarks/                   # Real-world benchmark scripts
├── docs/                         # MkDocs documentation
└── pyproject.toml                # Build config (maturin)
```

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

# mlxtend migration guide
uv run python examples/06_mlxtend_migration.py
```

---

## 📜 License

[BSD 3-Clause](LICENSE)
