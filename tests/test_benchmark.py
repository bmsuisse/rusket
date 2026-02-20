"""Benchmarks: fpgrowth-rs vs fptda-rs vs mlxtend at multiple data sizes + Polars input."""

from __future__ import annotations

import tracemalloc
import time

import numpy as np
import pandas as pd
import pytest

from rusket import fpgrowth, fptda, association_rules

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    from mlxtend.frequent_patterns import fpgrowth as mlx_fpgrowth
    from mlxtend.frequent_patterns import association_rules as mlx_assoc_rules

    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def _make_df(n_rows: int, n_cols: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate a realistic sparse boolean DataFrame."""
    support_values = np.zeros(n_cols)
    n_very_low = int(n_cols * 0.9)
    support_values[:n_very_low] = rng.uniform(0.0001, 0.009, n_very_low)
    n_medium = int(n_cols * 0.06)
    support_values[n_very_low : n_very_low + n_medium] = rng.uniform(
        0.01, 0.1, n_medium
    )
    n_high = n_cols - n_very_low - n_medium
    support_values[n_very_low + n_medium :] = rng.uniform(0.1, 0.65, n_high)
    return pd.DataFrame(
        {f"c{i}": (rng.random(n_rows) < support_values[i]) for i in range(n_cols)}
    )


def _make_sparse_df(
    n_rows: int, n_cols: int, items_per_row: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Real retail-basket style: each transaction contains only a handful of items.

    Each row selects exactly `items_per_row` distinct columns uniformly at
    random — giving a density of items_per_row/n_cols per cell (often < 1%).
    This is the regime where FP-TDA should shine vs FP-Growth.
    """
    data = np.zeros((n_rows, n_cols), dtype=np.uint8)
    for i in range(n_rows):
        cols = rng.choice(n_cols, size=items_per_row, replace=False)
        data[i, cols] = 1
    return pd.DataFrame(data, columns=[f"i{c}" for c in range(n_cols)])


RNG = np.random.default_rng(0)
DF_TINY = _make_df(5, 11, RNG)  # correctness / smoke
DF_SMALL = _make_df(1_000, 50, RNG)
DF_MEDIUM = _make_df(10_000, 400, RNG)
DF_LARGE = _make_df(100_000, 1_000, RNG)

# Sparse "retail basket" datasets — 2-8 items per transaction, large catalogue
RNG_SP = np.random.default_rng(42)
# ~3 items / transaction across 500 items  →  density ≈ 0.6%
DF_SPARSE_SMALL  = _make_sparse_df(10_000,  500,  3, RNG_SP)
# ~5 items / transaction across 1000 items →  density ≈ 0.5%
DF_SPARSE_MEDIUM = _make_sparse_df(30_000,  1_000, 5, RNG_SP)
# ~7 items / transaction across 2000 items →  density ≈ 0.35%
DF_SPARSE_LARGE  = _make_sparse_df(100_000, 2_000, 7, RNG_SP)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _timed(fn, *args, **kwargs):
    """Return (result, elapsed_seconds, peak_bytes)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak


# ---------------------------------------------------------------------------
# pytest-benchmark suites — FP-Growth
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="tiny")
def test_benchmark_tiny(benchmark) -> None:
    result = benchmark(fpgrowth, DF_TINY, min_support=0.5)
    assert result.shape[0] > 0


@pytest.mark.benchmark(group="small")
def test_benchmark_small(benchmark) -> None:
    result = benchmark(fpgrowth, DF_SMALL, min_support=0.1)
    assert result.shape[0] >= 0


@pytest.mark.benchmark(group="medium")
def test_benchmark_medium(benchmark) -> None:
    result = benchmark(fpgrowth, DF_MEDIUM, min_support=0.01)
    assert result.shape[0] >= 0


@pytest.mark.benchmark(group="large")
def test_benchmark_large_skip(benchmark) -> None:
    result = benchmark(fpgrowth, DF_LARGE, min_support=0.05)
    assert result.shape[0] >= 0


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
@pytest.mark.benchmark(group="polars_medium")
def test_benchmark_polars_medium(benchmark) -> None:
    df_pl = pl.from_pandas(DF_MEDIUM)
    result = benchmark(fpgrowth, df_pl, min_support=0.01)
    assert result.shape[0] >= 0


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
@pytest.mark.benchmark(group="polars_large")
def test_benchmark_polars_large(benchmark) -> None:
    df_pl = pl.from_pandas(DF_LARGE)
    result = benchmark(fpgrowth, df_pl, min_support=0.05)
    assert result.shape[0] >= 0


# ---------------------------------------------------------------------------
# pytest-benchmark suites — FP-TDA (same groups, benchmarked in parallel)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="tiny")
def test_benchmark_fptda_tiny(benchmark) -> None:
    result = benchmark(fptda, DF_TINY, min_support=0.5)
    assert result.shape[0] > 0


@pytest.mark.benchmark(group="small")
def test_benchmark_fptda_small(benchmark) -> None:
    result = benchmark(fptda, DF_SMALL, min_support=0.1)
    assert result.shape[0] >= 0


@pytest.mark.benchmark(group="medium")
def test_benchmark_fptda_medium(benchmark) -> None:
    result = benchmark(fptda, DF_MEDIUM, min_support=0.01)
    assert result.shape[0] >= 0


@pytest.mark.benchmark(group="large")
def test_benchmark_fptda_large(benchmark) -> None:
    result = benchmark(fptda, DF_LARGE, min_support=0.05)
    assert result.shape[0] >= 0


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
@pytest.mark.benchmark(group="polars_medium")
def test_benchmark_fptda_polars_medium(benchmark) -> None:
    df_pl = pl.from_pandas(DF_MEDIUM)
    result = benchmark(fptda, df_pl, min_support=0.01)
    assert result.shape[0] >= 0


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
@pytest.mark.benchmark(group="polars_large")
def test_benchmark_fptda_polars_large(benchmark) -> None:
    df_pl = pl.from_pandas(DF_LARGE)
    result = benchmark(fptda, df_pl, min_support=0.05)
    assert result.shape[0] >= 0


# ---------------------------------------------------------------------------
# Sparse "retail basket" benchmarks — the regime where FP-TDA is designed for
# (very few items per transaction, large catalogue)
# ---------------------------------------------------------------------------

# min_support chosen so ≥ a handful of frequent items still exist
# (with 3 items/row over 500 cols, P(any item) ≈ 0.6%)
_SPARSE_SMALL_SUP  = 0.002   # ≈ 20 out of 10k rows
_SPARSE_MEDIUM_SUP = 0.001   # ≈ 50 out of 50k rows
_SPARSE_LARGE_SUP  = 0.0005  # ≈ 100 out of 200k rows


@pytest.mark.benchmark(group="sparse_small")
def test_benchmark_fpgrowth_sparse_small(benchmark) -> None:
    result = benchmark(fpgrowth, DF_SPARSE_SMALL, min_support=_SPARSE_SMALL_SUP)
    assert result.shape[0] >= 0


@pytest.mark.benchmark(group="sparse_small")
def test_benchmark_fptda_sparse_small(benchmark) -> None:
    result = benchmark(fptda, DF_SPARSE_SMALL, min_support=_SPARSE_SMALL_SUP)
    assert result.shape[0] >= 0


@pytest.mark.benchmark(group="sparse_medium")
def test_benchmark_fpgrowth_sparse_medium(benchmark) -> None:
    result = benchmark(fpgrowth, DF_SPARSE_MEDIUM, min_support=_SPARSE_MEDIUM_SUP)
    assert result.shape[0] >= 0


@pytest.mark.benchmark(group="sparse_medium")
def test_benchmark_fptda_sparse_medium(benchmark) -> None:
    result = benchmark(fptda, DF_SPARSE_MEDIUM, min_support=_SPARSE_MEDIUM_SUP)
    assert result.shape[0] >= 0


@pytest.mark.benchmark(group="sparse_large")
def test_benchmark_fpgrowth_sparse_large(benchmark) -> None:
    result = benchmark(fpgrowth, DF_SPARSE_LARGE, min_support=_SPARSE_LARGE_SUP)
    assert result.shape[0] >= 0


@pytest.mark.benchmark(group="sparse_large")
def test_benchmark_fptda_sparse_large(benchmark) -> None:
    result = benchmark(fptda, DF_SPARSE_LARGE, min_support=_SPARSE_LARGE_SUP)
    assert result.shape[0] >= 0


def test_vs_fptda_sparse_comparison() -> None:
    """Head-to-head on all three sparse sizes — prints ratios."""
    for label, df, sup in [
        ("sparse_small  (10k rows, 500 items, 3/row)",  DF_SPARSE_SMALL,  _SPARSE_SMALL_SUP),
        ("sparse_medium (50k rows, 2k items,  5/row)",  DF_SPARSE_MEDIUM, _SPARSE_MEDIUM_SUP),
        ("sparse_large  (200k rows, 5k items, 7/row)",  DF_SPARSE_LARGE,  _SPARSE_LARGE_SUP),
    ]:
        _, fpg_t, fpg_mem = _timed(fpgrowth, df, min_support=sup)
        _, tda_t, tda_mem = _timed(fptda,    df, min_support=sup)
        winner = "fptda" if tda_t < fpg_t else "fpgrowth"
        ratio  = min(fpg_t, tda_t) / max(fpg_t, tda_t)
        print(
            f"\n[{label}]  fpgrowth={fpg_t:.3f}s  fptda={tda_t:.3f}s  "
            f"faster={winner} ({ratio:.2f}×)  "
            f"mem fpg={fpg_mem / 1e6:.1f}MB  tda={tda_mem / 1e6:.1f}MB"
        )


# ---------------------------------------------------------------------------
# Head-to-head: FP-TDA vs FP-Growth (dense data, internal)
# ---------------------------------------------------------------------------


def test_vs_fptda_small() -> None:
    _, fpg_t, fpg_mem = _timed(fpgrowth, DF_SMALL, min_support=0.1)
    _, tda_t, tda_mem = _timed(fptda, DF_SMALL, min_support=0.1)
    print(
        f"\n[small  fp-tda] fpgrowth={fpg_t * 1000:.1f}ms  fptda={tda_t * 1000:.1f}ms  "
        f"ratio={tda_t / fpg_t:.2f}×  mem fpg={fpg_mem / 1e3:.0f}KB  tda={tda_mem / 1e3:.0f}KB"
    )


def test_vs_fptda_medium() -> None:
    _, fpg_t, fpg_mem = _timed(fpgrowth, DF_MEDIUM, min_support=0.01)
    _, tda_t, tda_mem = _timed(fptda, DF_MEDIUM, min_support=0.01)
    print(
        f"\n[medium fp-tda] fpgrowth={fpg_t:.3f}s  fptda={tda_t:.3f}s  "
        f"ratio={tda_t / fpg_t:.2f}×  mem fpg={fpg_mem / 1e6:.1f}MB  tda={tda_mem / 1e6:.1f}MB"
    )


def test_vs_fptda_large() -> None:
    _, fpg_t, fpg_mem = _timed(fpgrowth, DF_LARGE, min_support=0.05)
    _, tda_t, tda_mem = _timed(fptda, DF_LARGE, min_support=0.05)
    print(
        f"\n[large  fp-tda] fpgrowth={fpg_t:.3f}s  fptda={tda_t:.3f}s  "
        f"ratio={tda_t / fpg_t:.2f}×  mem fpg={fpg_mem / 1e6:.1f}MB  tda={tda_mem / 1e6:.1f}MB"
    )


# ---------------------------------------------------------------------------
# Head-to-head comparisons vs mlxtend
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MLXTEND, reason="mlxtend not installed")
def test_vs_mlxtend_small() -> None:
    _, ours, our_mem = _timed(fpgrowth, DF_SMALL, min_support=0.1)
    _, mlx, mlx_mem = _timed(mlx_fpgrowth, DF_SMALL, min_support=0.1)
    print(
        f"\n[small ] ours={ours * 1000:.1f}ms  mlxtend={mlx * 1000:.1f}ms  speedup={mlx / ours:.1f}×  "
        f"mem ours={our_mem / 1e3:.0f}KB  mlx={mlx_mem / 1e3:.0f}KB"
    )
    assert ours < mlx * 5, f"Too slow at small: {ours:.3f}s vs mlxtend {mlx:.3f}s"


@pytest.mark.skipif(not HAS_MLXTEND, reason="mlxtend not installed")
def test_vs_mlxtend_medium() -> None:
    _, ours, our_mem = _timed(fpgrowth, DF_MEDIUM, min_support=0.01)
    _, mlx, mlx_mem = _timed(mlx_fpgrowth, DF_MEDIUM, min_support=0.01)
    print(
        f"\n[medium] ours={ours:.3f}s  mlxtend={mlx:.3f}s  speedup={mlx / ours:.1f}×  "
        f"mem ours={our_mem / 1e6:.1f}MB  mlx={mlx_mem / 1e6:.1f}MB"
    )
    assert ours < mlx * 3, f"Too slow at medium: {ours:.3f}s vs mlxtend {mlx:.3f}s"


@pytest.mark.skipif(not HAS_MLXTEND, reason="mlxtend not installed")
def test_vs_mlxtend_large() -> None:
    _, ours, our_mem = _timed(fpgrowth, DF_LARGE, min_support=0.05)
    _, mlx, mlx_mem = _timed(mlx_fpgrowth, DF_LARGE, min_support=0.05)
    print(
        f"\n[large ] ours={ours:.3f}s  mlxtend={mlx:.3f}s  speedup={mlx / ours:.1f}×  "
        f"mem ours={our_mem / 1e6:.1f}MB  mlx={mlx_mem / 1e6:.1f}MB"
    )
    assert ours < mlx * 3, f"Too slow at large: {ours:.3f}s vs mlxtend {mlx:.3f}s"
    assert our_mem <= mlx_mem * 3.0, (
        f"Memory regression at large: ours {our_mem / 1e6:.1f}MB vs mlxtend {mlx_mem / 1e6:.1f}MB"
    )


@pytest.mark.skipif(not HAS_MLXTEND, reason="mlxtend not installed")
def test_vs_mlxtend_assoc_rules_medium() -> None:
    """End-to-end: fpgrowth + association_rules vs mlxtend pipeline."""

    def ours():
        fi = fpgrowth(DF_MEDIUM, min_support=0.01)
        return association_rules(fi, len(DF_MEDIUM), min_threshold=0.5)

    def mlx():
        fi = mlx_fpgrowth(DF_MEDIUM, min_support=0.01)
        return mlx_assoc_rules(fi, len(DF_MEDIUM), min_threshold=0.5)

    _, ours_t, _ = _timed(ours)
    _, mlx_t, _ = _timed(mlx)
    print(
        f"\n[assoc/medium] ours={ours_t:.3f}s  mlxtend={mlx_t:.3f}s  speedup={mlx_t / ours_t:.1f}×"
    )
    assert ours_t < mlx_t * 3, f"Assoc rules too slow: {ours_t:.3f}s vs {mlx_t:.3f}s"
