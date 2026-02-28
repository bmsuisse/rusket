"""Faker-powered integration tests for the Pipeline at realistic scale.

Uses faker to generate synthetic user-item interactions and tests the
full Pipeline (retrieve → rerank → filter) at scale with timing assertions.
"""

import time

import numpy as np
import pandas as pd
import pytest
from faker import Faker

from rusket import ALS, BPR, Pipeline

fake = Faker()
Faker.seed(42)


def _generate_interactions(
    n_users: int = 500,
    n_items: int = 1000,
    avg_interactions_per_user: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate fake user-item interaction data."""
    rng = np.random.default_rng(seed)

    rows: list[dict[str, int]] = []
    for user_id in range(n_users):
        n_interactions = rng.poisson(avg_interactions_per_user)
        n_interactions = max(1, min(n_interactions, n_items))
        items = rng.choice(n_items, size=n_interactions, replace=False)
        for item_id in items:
            rows.append({"user_id": user_id, "item_id": int(item_id)})

    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def interactions_500x1000() -> pd.DataFrame:
    """500 users × 1000 items, ~20 interactions each → ~10k rows."""
    return _generate_interactions(500, 1000, 20)


@pytest.fixture(scope="module")
def als_model(interactions_500x1000: pd.DataFrame) -> ALS:
    return ALS(factors=32, iterations=5, seed=42).from_transactions(
        interactions_500x1000, user_col="user_id", item_col="item_id"
    ).fit()


@pytest.fixture(scope="module")
def bpr_model(interactions_500x1000: pd.DataFrame) -> BPR:
    return BPR(factors=32, iterations=50, seed=42).from_transactions(
        interactions_500x1000, user_col="user_id", item_col="item_id"
    ).fit()


class TestFakerPipelineIntegration:
    """End-to-end pipeline tests with realistic fake data."""

    def test_single_retriever_at_scale(self, als_model: ALS) -> None:
        """Single-model pipeline at 500 users."""
        pipe = Pipeline(retrieve=als_model)
        items, scores = pipe.recommend(user_id=0, n=10)
        assert len(items) == 10
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_multi_retriever_at_scale(self, als_model: ALS, bpr_model: BPR) -> None:
        """Multi-model pipeline with merge."""
        pipe = Pipeline(retrieve=[als_model, bpr_model], merge_strategy="max")
        items, scores = pipe.recommend(user_id=42, n=10)
        assert len(items) == 10

    def test_full_pipeline_at_scale(self, als_model: ALS, bpr_model: BPR) -> None:
        """Full retrieve → rerank → filter at scale."""
        blocked = set(range(0, 100))  # block first 100 items
        pipe = Pipeline(
            retrieve=als_model,
            rerank=bpr_model,
            filter=lambda ids, sc: (
                [i for i in ids if i not in blocked],
                [s for i, s in zip(ids, sc, strict=True) if i not in blocked],
            ),
        )
        items, scores = pipe.recommend(user_id=0, n=10)
        assert len(items) <= 10
        for item in items:
            assert item not in blocked

    def test_batch_recommend_speed(self, als_model: ALS) -> None:
        """Batch recommend for all 500 users should complete in < 5s."""
        pipe = Pipeline(retrieve=als_model)

        start = time.perf_counter()
        result = pipe.recommend_batch(n=10, format="records")
        elapsed = time.perf_counter() - start

        assert len(result) == 500
        assert all(len(r["item_ids"]) == 10 for r in result)
        assert elapsed < 5.0, f"Batch recommend took {elapsed:.2f}s (expected < 5s)"

    def test_batch_with_rerank_speed(self, als_model: ALS, bpr_model: BPR) -> None:
        """Batch with reranker for 500 users should complete in < 5s."""
        pipe = Pipeline(retrieve=als_model, rerank=bpr_model)

        start = time.perf_counter()
        result = pipe.recommend_batch(n=10, format="records")
        elapsed = time.perf_counter() - start

        assert len(result) == 500
        assert elapsed < 5.0, f"Batch + rerank took {elapsed:.2f}s (expected < 5s)"

    def test_batch_multi_retriever_with_filter(
        self, als_model: ALS, bpr_model: BPR
    ) -> None:
        """Multi-retriever + filter batch at 500 users."""
        blocked = {0, 1, 2, 3, 4}
        pipe = Pipeline(
            retrieve=[als_model, bpr_model],
            merge_strategy="sum",
            filter=lambda ids, sc: (
                [i for i in ids if i not in blocked],
                [s for i, s in zip(ids, sc, strict=True) if i not in blocked],
            ),
        )
        result = pipe.recommend_batch(n=5, format="records")
        assert len(result) == 500
        for r in result:
            for item in r["item_ids"]:
                assert item not in blocked

    def test_batch_pandas_output(self, als_model: ALS) -> None:
        """Batch output as pandas DataFrame."""
        pipe = Pipeline(retrieve=als_model)
        df = pipe.recommend_batch(user_ids=[0, 1, 2], n=5, format="pandas")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["user_id", "item_ids", "scores"]


class TestLargeScale:
    """Performance tests at larger scale (2k users × 5k items)."""

    @pytest.fixture(scope="class")
    def large_als(self) -> ALS:
        df = _generate_interactions(2000, 5000, 30, seed=99)
        return ALS(factors=64, iterations=5, seed=99).from_transactions(
            df, user_col="user_id", item_col="item_id"
        ).fit()

    def test_large_batch(self, large_als: ALS) -> None:
        """2000 users × 5000 items batch should complete in < 10s."""
        pipe = Pipeline(retrieve=large_als)

        start = time.perf_counter()
        result = pipe.recommend_batch(n=10, format="records")
        elapsed = time.perf_counter() - start

        assert len(result) == 2000
        assert all(len(r["item_ids"]) == 10 for r in result)
        assert elapsed < 10.0, f"Large batch took {elapsed:.2f}s (expected < 10s)"
        print(f"\n  ⚡ 2k users × 5k items batch: {elapsed:.3f}s")

    def test_large_single_user(self, large_als: ALS) -> None:
        """Single user recommendation at 5k items should be near-instant."""
        pipe = Pipeline(retrieve=large_als)

        start = time.perf_counter()
        items, scores = pipe.recommend(user_id=0, n=20)
        elapsed = time.perf_counter() - start

        assert len(items) == 20
        assert elapsed < 0.5, f"Single user took {elapsed:.2f}s (expected < 0.5s)"
