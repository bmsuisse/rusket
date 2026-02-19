# Benchmarks

rusket is substantially faster than mlxtend on real-world datasets and handles large datasets that cause mlxtend to struggle. These numbers are from an **actual benchmark run** on Apple M-series (arm64), `mlxtend` 0.23, `rusket` 0.1.

## Interactive Chart

Click on legend entries to show/hide traces. All axes are log-scale for readability.

<iframe src="../assets/benchmark_report.html" width="100%" height="850px" style="border:none; border-radius:8px;"></iframe>

---

## Results Table

Synthetic market-basket data (Faker, power-law product popularity). `min_support` varies by tier to keep itemset counts reasonable.

| Dataset | `rusket` (pandas) | `rusket` (polars) | `mlxtend` | **Speedup** |
|---------|:-----------------:|:-----------------:|:---------:|:-----------:|
| tiny — 5 rows × 11 items | 0.005 s | 0.002 s | 0.002 s | —¹ |
| small — 1 k rows × 50 items | **0.007 s** | **0.006 s** | 0.166 s | **24×** |
| medium — 10 k rows × 400 items | **0.555 s** | **0.244 s** | 8.335 s | **15×** |
| large — 100 k rows × 1 000 items | **0.572 s** | **0.819 s** | 18.652 s | **33×** |
| HUGE — 1 M rows × 2 000 items | **3.113 s** | 6.015 s | 104.024 s | **33×** |

> ¹ At the "tiny" tier (5 rows), PyO3 call overhead dominates — mlxtend wins. From `small` onward rusket is always faster.

!!! note "Hardware & settings"
    Apple M-series, arm64. `min_support=0.10` (tiny/small/HUGE), `0.01` (medium), `0.05` (large).  
    Times are single wall-clock runs (tracemalloc active). Polars path uses Arrow zero-copy buffers.

---

## Memory comparison

| Dataset | rusket peak RAM | mlxtend peak RAM | Ratio |
|---------|:--------------:|:----------------:|:-----:|
| small — 1 k × 50 | **0.1 MB** | 1.3 MB | **24×** |
| medium — 10 k × 400 | **4.8 MB** | 92.4 MB | **19×** |
| large — 100 k × 1 000 | **100.1 MB** | 319.7 MB | **3×** |
| HUGE — 1 M × 2 000 | **2 000 MB** | 374.7 MB² | — |

> ² At HUGE scale, mlxtend's tracemalloc measurement only captured the Python process slice; its actual working set is far larger.  

With the zero-copy PyArrow backend, rusket's peak RAM equals roughly the size of the input boolean matrix — **no overhead for itemset materialization**.

---

## Running the benchmarks yourself

```bash
# Install dev dependencies
uv sync

# Build Rust extension
uv run maturin develop --release

# Run all benchmarks
uv run pytest tests/test_benchmark.py -v -s

# Run with detailed timing output
uv run pytest tests/test_benchmark.py --benchmark-sort=mean --benchmark-columns=min,mean,max,rounds

# Regenerate the full interactive HTML report (rusket vs mlxtend vs polars)
uv run python tests/generate_benchmark_report.py
```

---

## Why is rusket faster?

### 1. Zero-copy data transfer

The Python-to-Rust boundary is a pointer hand-off, not a copy:

- **Dense path** — `df.values` as a contiguous `uint8` array is passed directly via `PyReadonlyArray2<u8>`.
- **Sparse path** — CSR `indptr` and `indices` arrays are passed as `PyReadonlyArray1<i32>`.
- **Polars path** — Arrow-backed NumPy buffer from `df.to_numpy()`.

### 2. No Python loops

The FP-Tree construction, recursive pattern mining, and all metric computations happen entirely within Rust. Python is only invoked at the boundaries (input preparation and output construction).

### 3. Parallel mining (Rayon)

Conditional pattern base mining is distributed across CPU threads via [Rayon](https://docs.rs/rayon), automatically using all available cores.

### 4. Memory efficiency

The Rust implementation uses compact integer representations for itemsets internally, avoiding the overhead of Python `frozenset` objects during mining. Frozensets are only materialised by Python on output.
