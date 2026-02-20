# Benchmarks

rusket includes two mining algorithms (**FP-Growth** and **Eclat**), both implemented in Rust. These benchmarks compare rusket against **mlxtend** (pure Python) on synthetic and real-world datasets.

Measured on Apple M-series (arm64).

---

## Million-Scale Sparse Benchmarks 🚀

The killer use case: sparse retail basket data with millions of transactions, fed through `from_transactions` → sparse CSR → Rust.

### Interactive Chart

<iframe src="../assets/scale_benchmark.html" width="100%" height="550px" style="border:none; border-radius:8px;"></iframe>

| Dataset | `rusket` | `mlxtend` | **Speedup** | RAM |
|---------|:--------:|:---------:|:-----------:|:---:|
| 1M txns × 10k items (5 items/basket) | **0.49 s** | 18.4 s | **38×** | 25 MB |
| 2M txns × 10k items (5 items/basket) | **0.68 s** | 45.2 s | **66×** | 50 MB |

!!! success "Key insight"
    With sparse CSR-backed DataFrames, rusket uses **only 25–50 MB of RAM** for millions of transactions. The sparse path passes CSR `indptr`/`indices` arrays directly to Rust — **zero densification, zero copy**.

---

## Synthetic Dense Benchmarks

| Dataset | `rusket` (fpgrowth) | `rusket` (eclat) | `mlxtend` | **Speedup** |
|---------|:-------------------:|:----------------:|:---------:|:-----------:|
| 100k txns × 1k items | **0.4 s** | 1.1 s | 4.6 s | **12×** |
| 100k txns × 5k items | **3.6 s** | 4.8 s | 6.4 s | **1.8×** |
| 500k txns × 5k items | **27.7 s** | 31.2 s | 41.8 s | **1.5×** |

---

## Real-World Datasets

Datasets from [andi611/Apriori-and-Eclat-Frequent-Itemset-Mining](https://github.com/andi611/Apriori-and-Eclat-Frequent-Itemset-Mining).

| Dataset | Transactions | Items | Avg basket | `rusket` | `mlxtend` | **Speedup** |
|---------|:----------:|:-----:|:----------:|:--------:|:---------:|:-----------:|
| andi_data.txt | 8,416 | 119 | 23.0 | **9.7 s** (22.8M itemsets) | **TIMEOUT** 💥 | ∞ |
| andi_data2.txt | 540,455 | 2,603 | 4.4 | **7.9 s** | 16.2 s | **2×** |

!!! warning "Dense data"
    On the dense `andi_data.txt` dataset (~23 items per basket), `mlxtend` can't even finish within 60 seconds.
    `rusket` mines **22.8 million itemsets in under 10 seconds**. 🚀

---

## FP-Growth vs Eclat

Both algorithms produce identical results. Choice depends on data shape:

| Data shape | Recommended | Why |
|---|---|---|
| Sparse retail baskets | Either — both fast | FP-Growth slightly ahead at scale |
| Dense data (many items/txn) | **FP-Growth** | Tree compression shines |
| General purpose | **FP-Growth** (default) | Proven, well-studied |

---

## The Full Pipeline: `from_transactions` at Scale

For real-world retail data (100M+ transactions × 200k+ items), use `from_transactions` which now produces **sparse CSR-backed DataFrames**:

```python
import pandas as pd
from rusket import from_transactions, fpgrowth

# Long-format: 100M rows of (transaction_id, item)
df = pd.read_parquet("orders.parquet")
ohe = from_transactions(df)   # → sparse CSR DataFrame (fits in memory!)
freq = fpgrowth(ohe, min_support=0.001, use_colnames=True, max_len=3)
```

Memory for 100M transactions × 200k items with ~5 items/basket:
- **Dense**: ~20 TB (impossible)
- **Sparse CSR**: ~4 GB (fits on a laptop)

---

## Running the benchmarks yourself

```bash
# Build optimized Rust extension
uv run maturin develop --release

# pytest-benchmark suite
uv run pytest tests/test_benchmark.py -v -s

# Real-world dataset benchmark (auto-downloads datasets)
uv run python benchmarks/bench_realworld.py
```

---

## Why is rusket faster?

### 1. Zero-copy data transfer

The Python-to-Rust boundary is a pointer hand-off, not a copy:

- **Dense path** — `df.values` as contiguous `uint8` array via `PyReadonlyArray2<u8>`.
- **Sparse path** — CSR `indptr` and `indices` arrays via `PyReadonlyArray1<i32>`.
- **Polars path** — Arrow-backed NumPy buffer from `df.to_numpy()`.

### 2. No Python loops

FP-Tree construction, recursive pattern mining, and metric computations happen entirely within Rust.

### 3. Parallel mining (Rayon)

Conditional pattern base mining is distributed across CPU threads via [Rayon](https://docs.rs/rayon).

### 4. Arena-based FP-Tree

FP-Tree nodes use a flat children arena (not per-node heap allocations), making the tree cache-friendly. `is_path()` is tracked incrementally — no full-tree scans.

### 5. Eclat: hardware popcount

Eclat stores transactions as dense `Vec<u64>` bitsets. Support counting uses bitwise intersection + `popcnt` — billions of operations per second on modern CPUs.

### 6. Sparse `from_transactions`

The `from_transactions` helper uses `pd.Categorical` codes + scipy COO→CSR — **no Python loops over rows**. Produces `SparseDtype("bool")` DataFrames that keep memory at the theoretical minimum.
