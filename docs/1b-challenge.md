# The 1B Challenge

At the core of `rusket` is the philosophy that frequent itemset mining, association rules, and recommendation algorithms (ALS) should scale seamlessly on a single node. The **1 Billion Row Challenge** is our benchmark to ensure the library can process massive, enterprise-scale datasets without Out-Of-Memory (OOM) crashes or relying on distributed clusters.

---

## Out-Of-Core (OOC) FPMiner 🚀

Memory is finite, but transactional data can be virtually limitless. When processing datasets scaling to hundreds of millions or billions of rows (like retail transactions), loading everything into RAM at once is not feasible.

`FPMiner` is `rusket`’s streaming solution. It leverages a hybrid memory/disk spilling strategy to handle datasets of any size, constrained only by disk space.

### The Strategy

1. **Chunked Ingestion:** Python pipelines (like a `pandas.read_parquet` iterator) feed sparse `(transaction_id, item_id)` integer arrays to Rust one chunk at a time. The peak Python memory cost is strictly the size of the current chunk.
2. **Dynamic RAM Limits:** The Python wrapper automatically detects your system's total memory using `psutil`. It caps the maximum RAM allocation (`max_ram_mb`) to leave enough headroom for the OS and the final CSR matrix allocation, preventing the system from freezing or OOM-killing the process.
3. **Per-Chunk Sorter:** Rust aggressively accumulates and sorts these chunks in memory.
4. **Disk Spilling:** Once the memory budget is reached, Rust spills the sorted intermediate state to disk.
5. **K-Way Merge:** Finally, during the `.mine()` call, Rust performs an efficient k-way merge of the disk-spilled segments, continuously feeding the FP-Tree without materializing the full merged dataset.

### Example: Streaming 1B rows

```python
import pandas as pd
from rusket import FPMiner

# Miner automatically caps max RAM dynamically
miner = FPMiner(n_items=500_000)

# Stream a massive 1B row dataset in small chunks
for chunk in pd.read_parquet("massive_orders.parquet", chunksize=10_000_000):
    txn = chunk["txn_id"].to_numpy(dtype="int64")
    item = chunk["item_idx"].to_numpy(dtype="int32")
    
    # Send pointers to Rust - memory strictly bounded
    miner.add_chunk(txn, item)

# K-way merge and mine the frequent itemsets
freq = miner.mine(min_support=0.001, max_len=3, method="fpgrowth")
```

---

## ALS & The MovieLens 1B Dataset

Collaborative filtering via Alternating Least Squares (ALS) requires large sparse matrices. Processing the **[MovieLens 1 Billion](https://grouplens.org/datasets/movielens/1b/)** interaction dataset introduces challenges beyond memory limits: SciPy's defaults and `dtype` compatibility at scale.

### The Memory-Mapped Approach

To handle the 1B interaction dataset, we use numpy memory-mapped arrays (`np.memmap`). This allows us to load massive binary files directly from disk into virtual memory without consuming active RAM.

### Zero-Copy Pointer Handoffs

SciPy's `csr_matrix` struggles with datasets exceeding 2 billion Non-Zero (NNZ) interactions unless explicitly forced into 64-bit indices (`int64`). Furthermore, constructing the sparse matrix purely in Python is incredibly slow.

`rusket`’s `als_fit` accepts a SciPy `csc_matrix` directly. The critical advantage is **Zero-Copy Pointer Handoffs**. `rusket` bypasses SciPy's internal data manipulation by directly exposing the underlying memory-mapped `indptr` and `indices` `int64` array buffers to Rust.

```python
import numpy as np
from scipy.sparse import csr_matrix
from rusket import als_fit

# Load memory-mapped 64-bit indices
indptr = np.memmap('indptr.dat', dtype=np.int64, mode='r')
indices = np.memmap('indices.dat', dtype=np.int64, mode='r')
data = np.memmap('data.dat', dtype=np.float32, mode='r')

# Construct the SciPy sparse matrix shell
sparse_matrix = csr_matrix((data, indices, indptr), shape=(n_users, n_items))

# Direct pointer handoff to Rust, no copying overhead
user_factors, item_factors = als_fit(
    sparse_matrix.tocsc(),  # ALS is column-oriented
    n_factors=64,
    n_iterations=15,
    regularization=0.1,
    alpha=10.0,
    verbose=1
)
```

By strictly managing memory bounds and enforcing typed, zero-copy interactions between Python and Rust, `rusket` achieves enterprise-grade reliability and performance on laptops and servers alike.
