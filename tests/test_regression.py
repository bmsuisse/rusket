"""Regression tests: absolute wall-time and memory thresholds.

These tests enforce that rusket never regresses past generous but finite
performance limits.  They are intentionally machine-agnostic (limits are set
to ~3–5× faster-than-CI observed speeds) so they pass on any runner while
still catching catastrophic slowdowns.

Run with:
    uv run pytest tests/test_regression.py -v -s
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from rusket import association_rules, fpgrowth

# ---------------------------------------------------------------------------
# Shared dataset generation (same seed as test_benchmark.py)
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)


def _make_df(n_rows: int, n_cols: int, rng: np.random.Generator) -> pd.DataFrame:
    """Realistic sparse boolean DataFrame matching test_benchmark.py."""
    support_values = np.zeros(n_cols)
    n_very_low = int(n_cols * 0.9)
    support_values[:n_very_low] = rng.uniform(0.0001, 0.009, n_very_low)
    n_medium = int(n_cols * 0.06)
    support_values[n_very_low : n_very_low + n_medium] = rng.uniform(0.01, 0.1, n_medium)
    n_high = n_cols - n_very_low - n_medium
    support_values[n_very_low + n_medium :] = rng.uniform(0.1, 0.65, n_high)
    return pd.DataFrame({f"c{i}": (rng.random(n_rows) < support_values[i]) for i in range(n_cols)})


# Pre-generate datasets at module import (once per session)
_RNG2 = np.random.default_rng(0)
DF_SMALL = _make_df(1_000, 50, _RNG2)
DF_MEDIUM = _make_df(10_000, 400, _RNG2)
DF_LARGE = _make_df(100_000, 1_000, _RNG2)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _timed(fn, *args, **kwargs) -> tuple[object, float, int]:
    """Return (result, elapsed_seconds, peak_bytes)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak


# ---------------------------------------------------------------------------
# Wall-time regression thresholds
#
# Thresholds are deliberately generous (~3–5× faster than CI baseline) so
# they hold on low-end machines while still catching catastrophic regressions.
# Tighten them as you collect CI timing data.
# ---------------------------------------------------------------------------

# (name, dataset, min_support, max_seconds, max_peak_mb)
_SCENARIOS: list[tuple[str, pd.DataFrame, float, float, float]] = [
    ("small", DF_SMALL, 0.10, 2.0, 100.0),
    ("medium", DF_MEDIUM, 0.01, 30.0, 512.0),
    ("large", DF_LARGE, 0.05, 120.0, 2048.0),
]


@pytest.mark.parametrize(
    "name,df,min_support,max_seconds,max_peak_mb",
    _SCENARIOS,
    ids=[s[0] for s in _SCENARIOS],
)
def test_fpgrowth_regression(
    name: str,
    df: pd.DataFrame,
    min_support: float,
    max_seconds: float,
    max_peak_mb: float,
) -> None:
    """fpgrowth must finish within wall-time and memory limits."""
    result, elapsed, peak_bytes = _timed(fpgrowth, df, min_support=min_support)

    peak_mb = peak_bytes / 1e6

    print(
        f"\n[{name}] elapsed={elapsed:.3f}s (limit={max_seconds}s)  "
        f"peak={peak_mb:.1f}MB (limit={max_peak_mb}MB)  "
        f"itemsets={result.shape[0]}"  # type: ignore
    )

    assert elapsed <= max_seconds, f"[{name}] Wall-time regression: {elapsed:.3f}s > {max_seconds}s limit"
    assert peak_mb <= max_peak_mb, f"[{name}] Memory regression: {peak_mb:.1f}MB > {max_peak_mb}MB limit"
    assert result is not None
    assert result.shape[0] >= 0  # type: ignore


# ---------------------------------------------------------------------------
# Association rules regression
# ---------------------------------------------------------------------------


def test_association_rules_regression() -> None:
    """End-to-end fpgrowth + association_rules must finish within 60 s."""
    max_seconds = 60.0

    def _pipeline() -> None:
        fi = fpgrowth(DF_MEDIUM, min_support=0.01)
        association_rules(fi, len(DF_MEDIUM), min_threshold=0.5)

    _, elapsed, _ = _timed(_pipeline)

    print(f"\n[assoc/medium] elapsed={elapsed:.3f}s (limit={max_seconds}s)")

    assert elapsed <= max_seconds, f"Association rules pipeline regression: {elapsed:.3f}s > {max_seconds}s"


# ---------------------------------------------------------------------------
# Polars input regression (optional)
# ---------------------------------------------------------------------------


try:
    import polars as pl

    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False


@pytest.mark.skipif(not _HAS_POLARS, reason="polars not installed")
def test_polars_medium_regression() -> None:
    """Polars input path must not be slower than 30 s on medium dataset."""
    max_seconds = 30.0
    df_pl = pl.from_pandas(DF_MEDIUM)

    result, elapsed, peak_bytes = _timed(fpgrowth, df_pl, min_support=0.01)
    peak_mb = peak_bytes / 1e6

    print(
        f"\n[polars/medium] elapsed={elapsed:.3f}s (limit={max_seconds}s)  "
        f"peak={peak_mb:.1f}MB  itemsets={result.shape[0]}"  # type: ignore
    )

    assert elapsed <= max_seconds, f"Polars input regression: {elapsed:.3f}s > {max_seconds}s"
