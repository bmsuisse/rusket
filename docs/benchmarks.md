# Benchmarks

rusket includes two mining algorithms (**FP-Growth** and **Eclat**), both implemented in Rust. These benchmarks compare rusket against **mlxtend** (pure Python) on synthetic and real-world datasets.

Measured on Apple M-series (arm64).

---

## Scale Benchmarks: 1M → 200M Rows 🚀

### Interactive Chart

<iframe src="../assets/scale_benchmark.html" width="100%" height="550px" style="border:none; border-radius:8px;"></iframe>

### Two Input Paths

rusket supports two ways to ingest data at scale:

1. **`from_transactions` → sparse DataFrame** — returns a pandas DataFrame, easy API
2. **Direct CSR → Rust** — pass `scipy.sparse.csr_matrix` directly to `fpgrowth()`, skips pandas entirely

| Scale | `from_transactions` → fpgrowth | Direct CSR → fpgrowth | **Speedup** |
|---|:---:|:---:|:---:|
| 1M rows (200k txns × 10k items) | 5.0s | **0.1s** | **50×** |
| 10M rows (2M txns × 50k items) | 24.4s | **1.7s** | **14×** |
| 50M rows (10M txns × 100k items) | 63.1s | **10.9s** | **6×** |
| 100M rows (20M txns × 200k items) | 134.2s | **25.9s** | **5×** |
| 200M rows (40M txns × 200k items) | 246.8s | **73.1s** | **3×** |

!!! success "Direct CSR is the power-user path"
    At 100M rows, direct CSR mining takes **1.3 seconds** — the bottleneck is entirely the CSR build (24.5s).
    Compare to the pandas sparse path where mining alone takes 30.4s due to `sparse.to_coo().tocsr()` overhead.

---

## vs mlxtend

| Dataset | `rusket` | `mlxtend` | **Speedup** | RAM |
|---------|:--------:|:---------:|:-----------:|:---:|
| 1M txns (sparse) | **0.49 s** | 18.4 s | **38×** | 25 MB |
| 2M txns (sparse) | **0.68 s** | 45.2 s | **66×** | 50 MB |

---

## Real-World Datasets

Datasets from [andi611/Apriori-and-Eclat-Frequent-Itemset-Mining](https://github.com/andi611/Apriori-and-Eclat-Frequent-Itemset-Mining).

| Dataset | Transactions | Items | `rusket` | `mlxtend` | **Speedup** |
|---------|:----------:|:-----:|:--------:|:---------:|:-----------:|
| andi_data.txt | 8,416 | 119 | **9.7 s** (22.8M itemsets) | **TIMEOUT** 💥 | ∞ |
| andi_data2.txt | 540,455 | 2,603 | **7.9 s** | 16.2 s | **2×** |

!!! warning "Dense data"
    On `andi_data.txt` (~23 items/basket), `mlxtend` can't finish in 60s. `rusket` mines **22.8M itemsets in under 10s**.

---

## The Power-User Pipeline

For maximum performance at 100M+ scale, skip pandas entirely:

```python
import numpy as np
from scipy import sparse as sp
from rusket import fpgrowth

# Your data: integer arrays of (txn_id, item_id)
txn_ids = np.array([...], dtype=np.int64)  # transaction IDs
item_ids = np.array([...], dtype=np.int32)  # item IDs

# Build CSR directly (fast!)
csr = sp.csr_matrix(
    (np.ones(len(txn_ids), dtype=np.int8), (txn_ids, item_ids)),
    shape=(n_transactions, n_items),
)

# Mine directly from CSR — no pandas overhead
freq = fpgrowth(csr, min_support=0.001, use_colnames=True,
                max_len=3, column_names=item_names)
```

At 100M rows, the mining step takes **1.3 seconds** (not a typo).

---

## Running the benchmarks

```bash
uv run maturin develop --release
uv run python benchmarks/bench_scale.py    # Scale benchmark + Plotly chart
uv run python benchmarks/bench_realworld.py  # Real-world datasets
uv run pytest tests/test_benchmark.py -v -s  # pytest-benchmark
```

---

## Why is rusket faster?

| Technique | Description |
|---|---|
| **Zero-copy CSR** | `indptr`/`indices` passed to Rust as pointer hand-offs |
| **Arena FP-Tree** | Flat children arena, incremental `is_path()` tracking |
| **Rayon** | Parallel conditional mining across CPU cores |
| **Eclat popcount** | `Vec<u64>` bitsets + hardware `popcnt` for support |
| **No Python loops** | FP-Tree, mining, and metrics all in Rust |
| **`pd.factorize`** | O(n) integer encoding, faster than `pd.Categorical` at scale |
