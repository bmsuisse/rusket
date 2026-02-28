"""pytest-benchmark suite for the Pipeline API.

Benchmarks the Pipeline at multiple scales:
- Single user recommend
- Batch recommend (Python fallback vs Rust fast path)
- Multi-retriever merge
- Full pipeline (retrieve → rerank → filter)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rusket import ALS, BPR, Pipeline

# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def _generate_interactions(n_users: int, n_items: int, avg: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, int]] = []
    for uid in range(n_users):
        n_int = max(1, min(rng.poisson(avg), n_items))
        for iid in rng.choice(n_items, size=n_int, replace=False):
            rows.append({"user_id": uid, "item_id": int(iid)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures (module-scoped to avoid re-training)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_als() -> ALS:
    """500 users × 1000 items."""
    df = _generate_interactions(500, 1000, 20)
    return ALS(factors=32, iterations=5, seed=42).from_transactions(df, user_col="user_id", item_col="item_id").fit()


@pytest.fixture(scope="module")
def small_bpr() -> BPR:
    """500 users × 1000 items."""
    df = _generate_interactions(500, 1000, 20)
    return BPR(factors=32, iterations=50, seed=42).from_transactions(df, user_col="user_id", item_col="item_id").fit()


@pytest.fixture(scope="module")
def medium_als() -> ALS:
    """2000 users × 5000 items."""
    df = _generate_interactions(2000, 5000, 30, seed=99)
    return ALS(factors=64, iterations=5, seed=99).from_transactions(df, user_col="user_id", item_col="item_id").fit()


@pytest.fixture(scope="module")
def large_als() -> ALS:
    """10000 users × 20000 items."""
    df = _generate_interactions(10000, 20000, 25, seed=77)
    return ALS(factors=64, iterations=3, seed=77).from_transactions(df, user_col="user_id", item_col="item_id").fit()


# ---------------------------------------------------------------------------
# Single-user benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="pipeline_single_user")
def test_bench_single_user_500x1k(benchmark, small_als: ALS) -> None:
    """Single user, 500u × 1k items."""
    pipe = Pipeline(retrieve=small_als)
    result = benchmark(pipe.recommend, user_id=0, n=10)
    assert len(result[0]) == 10


@pytest.mark.benchmark(group="pipeline_single_user")
def test_bench_single_user_2kx5k(benchmark, medium_als: ALS) -> None:
    """Single user, 2ku × 5k items."""
    pipe = Pipeline(retrieve=medium_als)
    result = benchmark(pipe.recommend, user_id=0, n=10)
    assert len(result[0]) == 10


@pytest.mark.benchmark(group="pipeline_single_user")
def test_bench_single_user_10kx20k(benchmark, large_als: ALS) -> None:
    """Single user, 10ku × 20k items."""
    pipe = Pipeline(retrieve=large_als)
    result = benchmark(pipe.recommend, user_id=0, n=10)
    assert len(result[0]) == 10


# ---------------------------------------------------------------------------
# Batch benchmarks — Rust fast path
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="pipeline_batch")
def test_bench_batch_500x1k(benchmark, small_als: ALS) -> None:
    """Batch 500 users × 1k items."""
    pipe = Pipeline(retrieve=small_als)
    result = benchmark(pipe.recommend_batch, n=10, format="records")
    assert len(result) == 500


@pytest.mark.benchmark(group="pipeline_batch")
def test_bench_batch_2kx5k(benchmark, medium_als: ALS) -> None:
    """Batch 2k users × 5k items."""
    pipe = Pipeline(retrieve=medium_als)
    result = benchmark(pipe.recommend_batch, n=10, format="records")
    assert len(result) == 2000


@pytest.mark.benchmark(group="pipeline_batch")
def test_bench_batch_10kx20k(benchmark, large_als: ALS) -> None:
    """Batch 10k users × 20k items."""
    pipe = Pipeline(retrieve=large_als)
    result = benchmark(pipe.recommend_batch, n=10, format="records")
    assert len(result) == 10000


# ---------------------------------------------------------------------------
# Multi-retriever + rerank
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="pipeline_full")
def test_bench_full_pipeline_500x1k(benchmark, small_als: ALS, small_bpr: BPR) -> None:
    """Full pipeline: ALS retrieve → BPR rerank, 500 users."""
    pipe = Pipeline(retrieve=small_als, rerank=small_bpr)
    result = benchmark(pipe.recommend_batch, n=10, format="records")
    assert len(result) == 500


@pytest.mark.benchmark(group="pipeline_full")
def test_bench_full_pipeline_with_filter(benchmark, small_als: ALS, small_bpr: BPR) -> None:
    """Full pipeline + filter, 500 users."""
    blocked = set(range(50))
    pipe = Pipeline(
        retrieve=small_als,
        rerank=small_bpr,
        filter=lambda ids, sc: (
            [i for i in ids if i not in blocked],
            [s for i, s in zip(ids, sc, strict=True) if i not in blocked],
        ),
    )
    result = benchmark(pipe.recommend_batch, n=10, format="records")
    assert len(result) == 500
