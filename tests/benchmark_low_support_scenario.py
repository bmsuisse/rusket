import time
import tracemalloc

import numpy as np
import pandas as pd

from rusket import fpgrowth

try:
    from mlxtend.frequent_patterns import fpgrowth as mlx_fpgrowth

    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False


def _make_transaction_df(n_rows: int, n_cols: int, rng: np.random.Generator) -> pd.DataFrame:
    rank = np.arange(1, n_cols + 1, dtype=np.float64)
    support = 0.5 / rank**0.6
    support = np.clip(support, 0.0001, 0.5)
    matrix = rng.random((n_rows, n_cols)) < support
    columns = [f"Item_{i}" for i in range(n_cols)]
    return pd.DataFrame(matrix.astype(bool), columns=columns)


def _timed(fn, *args, **kwargs):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak


def run_benchmark(n_rows, n_cols, min_sup):
    RNG = np.random.default_rng(42)
    print(f"\n[Scenario] {n_rows:,} rows × {n_cols:,} cols  min_sup={min_sup}", flush=True)

    t0 = time.perf_counter()
    print("  Generating dataset…", end=" ", flush=True)
    df = _make_transaction_df(n_rows, n_cols, RNG)
    print(f"{time.perf_counter() - t0:.2f}s  ({df.values.nbytes / 1e6:.0f} MB)")

    print("  rusket…", end=" ", flush=True)
    our_res, ours_t, ours_mem = _timed(fpgrowth, df, min_support=min_sup)
    n_fi = our_res.shape[0] if our_res is not None else 0
    print(f"{ours_t:.3f}s  peak={ours_mem / 1e6:.1f}MB  itemsets={n_fi:,}")

    if HAS_MLXTEND:
        if min_sup < 0.0005:
            print("  mlxtend… skipped (would take too long/OOM)", flush=True)
        else:
            print("  mlxtend…", end=" ", flush=True)
            try:
                mlx_res, mlx_t, mlx_mem = _timed(mlx_fpgrowth, df, min_support=min_sup)
                speedup = mlx_t / ours_t
                print(f"{mlx_t:.3f}s  peak={mlx_mem / 1e6:.1f}MB  speedup={speedup:.1f}x")
            except Exception as e:
                print(f"FAILED: {e}")


if __name__ == "__main__":
    scenarios = [
        (100_000, 2_000, 0.001),
        (100_000, 2_000, 0.0005),
        (100_000, 2_000, 0.0001),
        (100_000, 2_000, 0.00005),
        (100_000, 2_000, 0.00001),
    ]

    for r, c, s in scenarios:
        run_benchmark(r, c, s)
