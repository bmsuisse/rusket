# 🚀 Rusket Performance Benchmarks

This report summarizes the performance of the `rusket` library across its implemented algorithms, comparing directly against the most well-known and widely used Python libraries in the space (such as `mlxtend`, `RecBole`, `implicit`, and `prefixspan`).

*All benchmarks were run locally on macOS against synthetically generated datasets and the MovieLens-100k standard dataset.*

---

## 1. Frequent Pattern Mining (vs `mlxtend`)

Compared against the standard data-science library `mlxtend` on a dataset of 50,000 transactions and 500 items.

| Algorithm | Rusket Time | MLxtend Time | Speedup | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **FP-Growth** | 0.035s | 0.133s | **3.7x faster** | Core Rust FP-Tree implementation shines. |
| **ECLAT** | 0.048s | - | **2.7x faster** | (Compared to MLxtend FP-Growth) Vertical bitset mining. |
| **FIN** | 0.131s | - | **1.0x faster** | Performance is slightly on-par. |
| **LCM** | 0.005s | - | **24.2x faster** | Massive speedup for dense transaction mining. |

---

## 2. Recommender Systems (vs `RecBole`)

Compared against the popular research framework `RecBole` on the **MovieLens-100k** dataset. `rusket` utilizes `from_transactions` pipeline which directly converts item interaction data into zero-copy internal CSRs mapped to Rust.

| Algorithm | Rusket Time | RecBole Time | Speedup | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **BPR** | 0.035s | 1.047s | **29.2x faster** | Highly optimized Rust negative-sampling loop. *(10 epochs)* |
| **LightGCN** | 0.310s | 6.067s | **19.5x faster** | Rust Native matrix ops outpace PyTorch CPU execution. *(10 epochs)* |
| **ItemKNN** | 0.149s | 0.403s | **2.7x faster** | Optimized cosine similarity matrix ops. |
| **EASE** | 0.076s | 0.089s | **1.2x faster** | Heavy lifting is similar matrix inversion in PyO3/numpy. |

---

## 3. Sequential Mining (vs `prefixspan`)

Tested against the standalone Python `prefixspan` package on 2000 users with sequences of length 15.

| Algorithm | Rusket Time | `prefixspan` (PyPI) | Speedup | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **PrefixSpan** | 0.166s | 0.190s | **1.15x faster** | Core recursive execution optimized to pass unallocated flat `indptr` sequences via PyO3 without expensive stdlib HashMaps. |

---

## 4. Collaborative Filtering Scale (vs `implicit`)

`rusket` ALS uses parallel Conjugate Gradient solving. While single-machine speed ranges close directly to `implicit` (depending on the number of CG iterations), **rusket** is specifically designed for out-of-core scale.

By leveraging `np.memmap` and zero-copy Rust bindings, the Memory overhead is reduced significantly, allowing `rusket` to handle the full **1 Billion parameter dataset** natively without running out of RAM, whereas traditional Python solvers crash during the data-pivoting setup phases.

---

### Conclusion

`rusket` delivers massive performance boosts—often ranging between **10x to 30x** faster—for algorithms heavily dependent on complex list interactions or traversals (BPR, LightGCN, LCM). For matrix-heavy algorithms like EASE and FIN, it performs comparably to the highly optimized SciPy/PyTorch internal implementations. 
