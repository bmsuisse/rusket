# The 1B Challenge

*A story of memory management, algorithmic trade-offs, and the satisfying click of watching numbers improve.*

---

At the core of `rusket` is the belief that frequent itemset mining should scale on a **single machine**. No Spark cluster, no distributed coordination — just one process, tuned to be as efficient as possible.

The **1 Billion Row Challenge** is how we hold ourselves accountable.

---

## Where We Started: Sorted Chunks + K-Way Merge

The first streaming design took inspiration from external-sort databases:

1. Each call to `add_chunk()` receives `(txn_id, item_id)` arrays from Python.
2. Rust sorts each chunk in-place using `rayon::par_sort_unstable()`.
3. Chunks exceeding the RAM budget are spilled to anonymous `tempfile` files on disk.
4. When `.mine()` is called, a **k-way heap merge** streams all sorted chunks in order, building the CSR matrix on the fly.

```
Memory: O(chunk_size)
mine() memory: O(k cursors + CSR output)
```

### The first problem: disk exhaustion

Running the 500M → 1B targets, our machine hit **99% disk usage**. The culprit: 500k transactions/chunk × 23 items/txn × 12 bytes/pair = ~138 MB per chunk. At 1B rows that's ~2,000 chunks = **~276 GB of tempfiles**. Oops.

### The second problem: `mine_t` grows super-linearly

| target_rows | add_t | mine_t | total |
|-------------|-------|--------|-------|
| 300M | 36.1s | **516.8s** | 552.9s |
| 500M | 65.2s | **1543.5s** | 1608.7s |

The heap with 1,000+ cursors causes cache thrashing — a fundamental problem with the architecture.

---

## The Insight: HashMap Aggregation

The key observation: the sorted-chunk approach stores **every pair** from every chunk. The real data that matters is `unique_transactions × avg_items_per_transaction`. For 1B rows with ~43M unique transactions × 23 items: that's only **~5GB** vs the sorted approach's **~12GB**.

We replaced the entire chunk + merge system with a single `AHashMap<i64, Vec<i32>>`:

```rust
pub fn add_chunk(&mut self, txn_ids: ..., item_ids: ...) {
    for (&t, &i) in txns.iter().zip(items.iter()) {
        self.txns.entry(t).or_default().push(i);
    }
}
```

`mine()` now just:

1. Collects `(txn_id, &items)` from the HashMap
2. `par_sort` by txn_id
3. Sort+dedup each transaction's item list
4. Build CSR → feed to algorithm

**No k-way merge. No disk spill. No tempfiles.**

### Initial numbers (100M & 200M, FP-Growth)

| target_rows | add_t | mine_t | total |
|-------------|-------|--------|-------|
| 100M | 5.7s | 26.6s | **32.3s** |
| 200M | 10.4s | 84.6s | **95.0s** |

Compared to the old approach's 300M taking 299.5s — a **3× speedup** just from the architecture change.

---

## What We Tried: The Iteration Phase

### Approach: SmallVec + u16 items (❌ Regression)

Using `SmallVec<[u16; 32]>` to store items inline on the stack caused a **2× slowdown**. The 32-element inline size is too large for the CPU stack, causes cache pressure on every HashMap lookup.

**Lesson: measure first, optimize second.**

### Knob: Chunk size (100k vs 500k vs 2M)

| chunk | add_t | mine_t | total |
|-------|-------|--------|-------|
| 100k | 10.0s | 3.8s | **13.8s** |
| 500k | 12.1s | 5.3s | **17.4s** |
| 2M | 12.5s | 5.3s | **17.8s** |

Chunk size barely matters.

### Knob: Algorithm (FP-Growth vs Eclat)

This was the **biggest discovery**. At 100M rows:

| method | add_t | mine_t | total | M rows/s |
|--------|-------|--------|-------|----------|
| fpgrowth | 10.0s | ~55s | ~65s | 1.5 |
| **eclat** | 10.0s | **3.8s** | **13.8s** | **7.24** |

**Eclat is ~14× faster at mining** for dense retail data. The reason: Eclat works with vertical tidlists — for dense datasets with many frequent 2-itemsets, intersection operations are extremely cache-friendly.

---

## Final Results: The Road to 1B

### Dense retail data (avg 23 items/txn)

| target_rows | add_t | mine_t | total | M rows/s | itemsets |
|-------------|-------|--------|-------|----------|----------|
| 100M | 12.5s | 6.2s | **18.7s** | 5.35 | 15,218 |
| 200M | 22.7s | 13.4s | **36.1s** | 5.53 | 15,226 |
| 500M | 61.1s | 38.7s | **99.8s** | 5.01 | 15,234 |
| **1B** | **173.5s** | **208.5s** | **382.1s** | **2.62** | **15,233** |

✅ **1 Billion rows. 382 seconds. 15,233 frequent itemsets. No OOM. No disk spill.**

### Sparse catalogue data (avg 4.4 items/txn)

| target_rows | add_t | mine_t | total |
|-------------|-------|--------|-------|
| 100M | 20.2s | 4.9s | **25.1s** |
| 200M | 26.2s | 9.2s | **35.4s** |
| 500M | ❌ | ❌ | **OOM** |
| **1B** | ❌ | ❌ | **OOM** |

**The 1B challenge for sparse datasets remains open.**

---

## What We Learned

### Architecture beats micro-optimisation

The jump from **k-way merge → HashMap** changed 5-minute runs into 30-second runs.

### Eclat vs FP-Growth depends on density

| data density | winner | why |
|---|---|---|
| Dense (avg >10 items/txn) | **Eclat** | Tidlist intersections are O(n), very cache-friendly |
| Sparse (avg <5 items/txn) | FP-Growth or similar | Fewer candidates, conditional bases are small |

---

## Running It Yourself

```python
from rusket import FPMiner
import numpy as np

miner = FPMiner(n_items=your_n_items)

for txn_ids_chunk, item_ids_chunk in your_data_stream():
    miner.add_chunk(
        txn_ids_chunk.astype(np.int64),
        item_ids_chunk.astype(np.int32),
    )

freq = miner.mine(min_support=0.02, max_len=3, method="eclat")
print(f"Found {len(freq):,} frequent itemsets")
```

The full benchmark script is in [`benchmarks/bench_fpminer_realistic.py`](https://github.com/bmsuisse/rusket/blob/main/benchmarks/bench_fpminer_realistic.py).
