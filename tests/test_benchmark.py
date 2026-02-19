"""Benchmarks: fpgrowth-rs vs mlxtend at multiple data sizes + Polars input."""

from __future__ import annotations

import tracemalloc
import time

import numpy as np
import pandas as pd
import pytest

from rusket import fpgrowth, association_rules

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
    support_values[n_very_low : n_very_low + n_medium] = rng.uniform(0.01, 0.1, n_medium)
    n_high = n_cols - n_very_low - n_medium
    support_values[n_very_low + n_medium :] = rng.uniform(0.1, 0.65, n_high)
    return pd.DataFrame(
        {f"c{i}": (rng.random(n_rows) < support_values[i]) for i in range(n_cols)}
    )


RNG = np.random.default_rng(0)
DF_TINY   = _make_df(5,       11,    RNG)   # correctness / smoke
DF_SMALL  = _make_df(1_000,   50,    RNG)
DF_MEDIUM = _make_df(10_000,  400,   RNG)
DF_LARGE  = _make_df(100_000, 1_000, RNG)


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
# pytest-benchmark suites (group per size)
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
# Head-to-head comparisons vs mlxtend
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_MLXTEND, reason="mlxtend not installed")
def test_vs_mlxtend_small() -> None:
    _, ours, our_mem = _timed(fpgrowth, DF_SMALL, min_support=0.1)
    _, mlx,  mlx_mem = _timed(mlx_fpgrowth, DF_SMALL, min_support=0.1)
    print(f"\n[small ] ours={ours*1000:.1f}ms  mlxtend={mlx*1000:.1f}ms  speedup={mlx/ours:.1f}×  "
          f"mem ours={our_mem/1e3:.0f}KB  mlx={mlx_mem/1e3:.0f}KB")
    assert ours < mlx * 5, f"Too slow at small: {ours:.3f}s vs mlxtend {mlx:.3f}s"


@pytest.mark.skipif(not HAS_MLXTEND, reason="mlxtend not installed")
def test_vs_mlxtend_medium() -> None:
    _, ours, our_mem = _timed(fpgrowth, DF_MEDIUM, min_support=0.01)
    _, mlx,  mlx_mem = _timed(mlx_fpgrowth, DF_MEDIUM, min_support=0.01)
    print(f"\n[medium] ours={ours:.3f}s  mlxtend={mlx:.3f}s  speedup={mlx/ours:.1f}×  "
          f"mem ours={our_mem/1e6:.1f}MB  mlx={mlx_mem/1e6:.1f}MB")
    assert ours < mlx * 3, f"Too slow at medium: {ours:.3f}s vs mlxtend {mlx:.3f}s"


@pytest.mark.skipif(not HAS_MLXTEND, reason="mlxtend not installed")
def test_vs_mlxtend_large() -> None:
    _, ours, our_mem = _timed(fpgrowth, DF_LARGE, min_support=0.05)
    _, mlx,  mlx_mem = _timed(mlx_fpgrowth, DF_LARGE, min_support=0.05)
    print(f"\n[large ] ours={ours:.3f}s  mlxtend={mlx:.3f}s  speedup={mlx/ours:.1f}×  "
          f"mem ours={our_mem/1e6:.1f}MB  mlx={mlx_mem/1e6:.1f}MB")
    assert ours < mlx * 3, f"Too slow at large: {ours:.3f}s vs mlxtend {mlx:.3f}s"
    assert our_mem <= mlx_mem * 3.0, (
        f"Memory regression at large: ours {our_mem/1e6:.1f}MB vs mlxtend {mlx_mem/1e6:.1f}MB"
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
    _, mlx_t, _  = _timed(mlx)
    print(f"\n[assoc/medium] ours={ours_t:.3f}s  mlxtend={mlx_t:.3f}s  speedup={mlx_t/ours_t:.1f}×")
    assert ours_t < mlx_t * 3, f"Assoc rules too slow: {ours_t:.3f}s vs {mlx_t:.3f}s"
