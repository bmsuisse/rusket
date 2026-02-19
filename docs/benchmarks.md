# Benchmarks

rusket is substantially faster than mlxtend on real-world datasets and handles large datasets that cause mlxtend to run out of memory.

## Summary

| Dataset | rusket | mlxtend | Speedup |
|---|---|---|---|
| Small (5 rows × 11 items) | **~0.1 ms** | ~1 ms | ~10× |
| Medium (10 000 rows × 400 items) | **< 1 s** | 3–8 s | ~5–8× |
| Large (100 000 rows × 1 000 items) | **< 5 s** | very slow / OOM | N/A |

!!! note "Hardware"
    Benchmarks run on Apple M-series (arm64). Times are wall-clock medians over multiple runs.  
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

## Memory comparison

| Dataset | rusket peak RAM | mlxtend peak RAM |
|---|---|---|
| Medium (10k × 400) | ~150 MB | ~1.2 GB |
| Large (100k × 1000) | ~600 MB | OOM (> 16 GB) |
