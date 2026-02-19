# rusket 🦀⚡

> **Blazing-fast FP-Growth and Association Rules for Python, powered by Rust.**

`rusket` is a drop-in replacement for `mlxtend`'s `fpgrowth` and `association_rules` — with a Rust core that delivers 5–10× speed-ups and dramatically lower memory usage.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/rust-1.83%2B-orange?logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)

---

## Why rusket?

| | `rusket` | `mlxtend` |
|---|---|---|
| **Core** | Rust (PyO3) | Pure Python |
| **Pandas sparse input** | ✅ Zero-copy CSR | ❌ Densifies first |
| **Polars input** | ✅ Arrow zero-copy | ❌ |
| **Parallel mining** | ✅ Rayon | ❌ |
| **Memory** | Low (native buffers) | High (Python objects) |
| **API compatibility** | ✅ Drop-in | — |

---

## Installation

```bash
pip install rusket
# or with uv:
uv add rusket
```

**Optional — Polars support:**

```bash
pip install "rusket[polars]"
```

---

## Quick Start

```python
import pandas as pd
from rusket import fpgrowth, association_rules

# One-hot encoded boolean / 0-1 DataFrame
df = pd.read_csv("transactions.csv").astype(bool)

# 1. Mine frequent itemsets
freq = fpgrowth(df, min_support=0.05, use_colnames=True)

# 2. Generate association rules
rules = association_rules(freq, num_itemsets=len(df), metric="lift", min_threshold=1.2)

print(rules.sort_values("lift", ascending=False).head())
```

Output looks exactly like `mlxtend`:

```
       antecedents    consequents  support  confidence   lift  ...
0  (bread, butter)       (milk,)     0.07        0.92   2.41  ...
1         (milk,)  (bread, eggs)     0.06        0.78   1.89  ...
```

### Polars input

```python
import polars as pl
from rusket import fpgrowth

df_pl = pl.read_parquet("transactions.parquet")
freq = fpgrowth(df_pl, min_support=0.05, use_colnames=True)
```

Polars DataFrames are converted via Arrow zero-copy buffers — no Python overhead.

### Sparse Pandas input

```python
df_sparse = df.astype(pd.SparseDtype("bool", fill_value=False))
freq = fpgrowth(df_sparse, min_support=0.05, use_colnames=True)
```

Sparse frames are passed to Rust as raw CSR arrays — no densification, minimal memory.

---

## API Reference

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
| `df` | `pd.DataFrame` \| `pl.DataFrame` | One-hot encoded input (bool / 0-1). Dense, sparse, or Polars. |
| `min_support` | `float` | Minimum support in `(0, 1]`. |
| `null_values` | `bool` | Allow NaN values in `df` (pandas only). |
| `use_colnames` | `bool` | Return column names instead of indices in itemsets. |
| `max_len` | `int \| None` | Maximum itemset length. `None` = unlimited. |
| `verbose` | `int` | Verbosity (kept for API compatibility). |

**Returns** `pd.DataFrame` with columns `['support', 'itemsets']`.  
Each itemset is a `frozenset` of column indices (or names when `use_colnames=True`).

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
| `df` | `pd.DataFrame` | Output of `fpgrowth()`. |
| `num_itemsets` | `int` | Number of transactions in the original dataset. |
| `metric` | `str` | Metric to filter rules on (see table below). |
| `min_threshold` | `float` | Minimum value of `metric` to include a rule. |
| `support_only` | `bool` | Only compute support; fill other columns with `NaN`. |
| `return_metrics` | `list[str]` | Subset of metrics to include in the result. |

**Returns** `pd.DataFrame` with columns `antecedents`, `consequents`, plus all requested metric columns.

#### Available metrics

| Metric | Description |
|--------|-------------|
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

## Benchmarks

Measured on Apple M-series hardware. `mlxtend` 0.23, `rusket` 0.1.

| Dataset | `rusket` | `mlxtend` | Speedup |
|---------|-----------|-----------|---------|
| Small (1k × 50) | ~2 ms | ~15 ms | **~8×** |
| Medium (10k × 400) | ~0.4 s | ~4 s | **~10×** |
| Large (100k × 1 000) | ~3 s | OOM / very slow | **N/A** |

> Memory usage at large scale is proportionally lower due to native Rust buffers (no Python object overhead).

---

## How It Works

```
rusket/
├── src/                          # Rust (PyO3)
│   ├── lib.rs                    # Module root & Python bindings
│   ├── fpgrowth.rs               # FP-Tree construction + FP-Growth mining
│   ├── association_rules.rs      # Rule generation + 12 metrics (Rayon parallel)
│   └── common.rs                 # Shared helpers
│
└── python/rusket/               # Thin Python wrappers & validation
    ├── fpgrowth.py               # Input dispatch (dense / sparse / Polars)
    ├── association_rules.py      # Label mapping + Rust call + result assembly
    └── _validation.py            # Input validation
```

### Data paths

```
pandas dense  ──► np.uint8 array (C-contiguous) ──► Rust fpgrowth_from_dense
pandas sparse ──► CSR int32 arrays              ──► Rust fpgrowth_from_csr
polars        ──► Arrow → np.uint8 (zero-copy)  ──► Rust fpgrowth_from_dense
```

All mining and rule generation happens inside Rust. No Python loops, no round-trips.

---

## Development

### Prerequisites

- Rust 1.83+ (`rustup update`)
- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/)

### Getting started

```bash
# Clone
git clone https://github.com/your-org/rusket
cd rusket

# Build Rust extension in dev mode
uv run maturin develop --release

# Run the full test suite
uv run pytest tests/ -x -q

# Type-check the Python layer
uv run pyright python/

# Cargo check (Rust)
cargo check
```

### Benchmarks

```bash
# Run pytest-benchmark suite
uv run pytest tests/test_benchmark.py -v -s

# Generate a full benchmark report (vs mlxtend)
uv run python tests/generate_benchmark_report.py
```

---

## License

[BSD 3-Clause](LICENSE)
