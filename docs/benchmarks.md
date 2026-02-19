# Benchmarks

rusket is substantially faster than mlxtend on real-world datasets and handles large datasets that cause mlxtend to run out of memory.

## Summary

| Dataset | `rusket` | `mlxtend` | Speedup |
|---------|-----------|-----------|---------|
| Small (1k × 50) | ~2 ms | ~15 ms | **~8×** |
| Medium (10k × 400) | ~0.4 s | ~4 s | **~10×** |
| Large (100k × 1 000) | ~3 s | OOM / very slow | **N/A** |

!!! note "Hardware"
    Benchmarks run on Apple M-series (arm64). `mlxtend` 0.23, `rusket` 0.1. Times are wall-clock medians over multiple runs.  
    `min_support=0.05`, `use_colnames=False`.

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

---

## Memory comparison (Zero-Copy PyArrow)

With the new zero-copy PyArrow backend, `rusket` has effectively **0 overhead** beyond the input DataFrame size during materialization.

| Dataset | rusket Peak RAM | mlxtend Peak RAM | Speedup | Itemsets Found |
|---|---|---|---|---|
| 100k × 2k (min_sup=0.001) | **200 MB** | 1,164 MB | ~29× | 56k |
| 100k × 2k (min_sup=0.0005) | **200 MB** | 1,283 MB | ~42× | 179k |
| 100k × 2k (min_sup=0.0001) | **200 MB** | OOM | ∞ | 2.8M |
| 100k × 2k (min_sup=0.00005) | **200 MB** | OOM | ∞ | 10.2M |

*Note: The input DataFrame occupies ~200MB. A peak RAM of 200MB means pattern mining and creating 10+ million itemsets via PyArrow PyO3 bindings added virtually no memory overhead in Python.*
