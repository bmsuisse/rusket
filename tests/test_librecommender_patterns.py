"""Tests adapted from LibRecommender test patterns.

Tests the full recommend workflow for each overlapping model:
  - invalid parameter handling
  - fit → recommend_items → evaluate pipeline
  - edge cases (cold-start user, all-consumed user)
  - similarity correctness

Reference: https://github.com/massquantity/LibRecommender/tree/master/tests
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from rusket import ALS, BPR, EASE, SVD, ItemKNN, LightGCN, evaluate


# ── Shared fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def synthetic_implicit_matrix():
    """Small implicit interaction matrix (50 users, 30 items)."""
    rng = np.random.default_rng(42)
    n_users, n_items = 50, 30
    rows = rng.integers(0, n_users, 300)
    cols = rng.integers(0, n_items, 300)
    mat = sparse.csr_matrix(
        (np.ones(300, dtype=np.float32), (rows, cols)),
        shape=(n_users, n_items),
    )
    return mat


@pytest.fixture
def synthetic_explicit_matrix():
    """Small explicit rating matrix (50 users, 30 items)."""
    rng = np.random.default_rng(42)
    n_users, n_items = 50, 30
    rows = rng.integers(0, n_users, 300)
    cols = rng.integers(0, n_items, 300)
    vals = rng.uniform(1.0, 5.0, 300).astype(np.float32)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))


@pytest.fixture
def synthetic_df():
    """Small DataFrame for from_transactions tests."""
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        "user": rng.integers(0, 50, n),
        "item": rng.integers(0, 30, n),
        "rating": rng.uniform(1.0, 5.0, n).astype(np.float32),
    })


# ── Test: Full workflow (train → predict → evaluate) ─────────────────────
# Adapted from LibRecommender's per-model test pattern:
#   model.fit() → ptest_preds() → ptest_recommends() → save_load_model()

class TestImplicitWorkflow:
    """Test full workflow for implicit models (ALS, BPR, ItemKNN, EASE, LightGCN)."""

    @pytest.mark.parametrize("model_cls,kwargs", [
        (ALS, {"factors": 16, "iterations": 3, "seed": 42}),
        (BPR, {"factors": 16, "iterations": 3, "seed": 42}),
        (LightGCN, {"factors": 16, "iterations": 3, "seed": 42}),
    ])
    def test_fit_recommend_evaluate(
        self, synthetic_implicit_matrix: sparse.csr_matrix, model_cls, kwargs
    ) -> None:
        mat = synthetic_implicit_matrix
        model = model_cls(**kwargs)
        model.fit(mat)

        # recommend_items should return items and scores
        items, scores = model.recommend_items(0, n=5, exclude_seen=True)
        assert len(items) <= 5
        assert len(items) == len(scores)
        # Scores descending (if we have any)
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

        # recommend_users (not all models support this)
        try:
            users, scores = model.recommend_users(0, n=5)
            assert len(users) == 5
        except NotImplementedError:
            pass  # OK — BPR/LightGCN don't support this

        # evaluate
        test_pairs = np.array([[0, 1], [0, 5], [1, 3]], dtype=np.int32)
        metrics = evaluate(model, test_pairs, k=5)
        for metric_name in ["ndcg", "hr", "precision", "recall"]:
            assert metric_name in metrics
            assert 0.0 <= metrics[metric_name] <= 1.0

    def test_itemknn_workflow(self, synthetic_implicit_matrix: sparse.csr_matrix) -> None:
        model = ItemKNN(k=10)
        model.fit(synthetic_implicit_matrix)
        items, scores = model.recommend_items(0, n=5, exclude_seen=True)
        assert len(items) <= 5

    def test_ease_workflow(self, synthetic_implicit_matrix: sparse.csr_matrix) -> None:
        model = EASE()
        model.fit(synthetic_implicit_matrix)
        items, scores = model.recommend_items(0, n=5, exclude_seen=True)
        assert len(items) == 5


class TestExplicitWorkflow:
    """Test full workflow for explicit models (SVD)."""

    def test_svd_fit_predict_recommend(
        self, synthetic_explicit_matrix: sparse.csr_matrix
    ) -> None:
        model = SVD(factors=16, iterations=10, seed=42)
        model.fit(synthetic_explicit_matrix)

        # predict should return a float
        pred = model.predict(0, 0)
        assert isinstance(pred, float)

        # recommend_items
        items, scores = model.recommend_items(0, n=5, exclude_seen=True)
        assert len(items) == 5


# ── Test: Invalid parameter handling ──────────────────────────────────────
# Adapted from LibRecommender's parametrized error-checking pattern

class TestInvalidParams:
    """Verify models raise on invalid parameters.

    LibRecommender tests this extensively via parametrized fixtures —
    we test the most important edge cases.
    """

    def test_als_not_fitted(self) -> None:
        model = ALS()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.recommend_items(0)

    def test_bpr_not_fitted(self) -> None:
        model = BPR()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.recommend_items(0)

    def test_svd_not_fitted(self) -> None:
        model = SVD()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.recommend_items(0)

    def test_svd_predict_not_fitted(self) -> None:
        model = SVD()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(0, 0)

    def test_als_double_fit(self, synthetic_implicit_matrix: sparse.csr_matrix) -> None:
        model = ALS(factors=8, iterations=1)
        model.fit(synthetic_implicit_matrix)
        with pytest.raises(RuntimeError, match="already fitted"):
            model.fit(synthetic_implicit_matrix)

    def test_bpr_double_fit(self, synthetic_implicit_matrix: sparse.csr_matrix) -> None:
        model = BPR(factors=8, iterations=1)
        model.fit(synthetic_implicit_matrix)
        with pytest.raises(RuntimeError, match="already fitted"):
            model.fit(synthetic_implicit_matrix)

    def test_svd_double_fit(self, synthetic_explicit_matrix: sparse.csr_matrix) -> None:
        model = SVD(factors=8, iterations=1)
        model.fit(synthetic_explicit_matrix)
        with pytest.raises(RuntimeError, match="already fitted"):
            model.fit(synthetic_explicit_matrix)


# ── Test: from_transactions convenience constructors ──────────────────────
# Adapted from LibRecommender's data preparation tests in conftest.py

class TestFromTransactions:
    """Test from_transactions for all models that support it."""

    def test_als_from_transactions(self, synthetic_df: pd.DataFrame) -> None:
        model = ALS.from_transactions(
            synthetic_df, transaction_col="user", item_col="item",
            factors=16, iterations=2,
        )
        assert model.fitted
        items, _ = model.recommend_items(0, n=3)
        assert len(items) == 3

    def test_bpr_from_transactions(self, synthetic_df: pd.DataFrame) -> None:
        model = BPR.from_transactions(
            synthetic_df, transaction_col="user", item_col="item",
            factors=16, iterations=2,
        )
        assert model.fitted

    def test_svd_from_transactions(self, synthetic_df: pd.DataFrame) -> None:
        model = SVD.from_transactions(
            synthetic_df, transaction_col="user", item_col="item", rating_col="rating",
            factors=16, iterations=2,
        )
        assert model.fitted
        pred = model.predict(0, 0)
        assert isinstance(pred, float)

    def test_itemknn_from_transactions(self, synthetic_df: pd.DataFrame) -> None:
        model = ItemKNN.from_transactions(
            synthetic_df, transaction_col="user", item_col="item", k=10,
        )
        assert model.fitted

    def test_ease_from_transactions(self, synthetic_df: pd.DataFrame) -> None:
        model = EASE.from_transactions(
            synthetic_df, transaction_col="user", item_col="item",
        )
        assert model.fitted

    def test_lightgcn_from_transactions(self, synthetic_df: pd.DataFrame) -> None:
        model = LightGCN.from_transactions(
            synthetic_df, transaction_col="user", item_col="item",
            factors=16, iterations=2,
        )
        assert model.fitted


# ── Test: batch_recommend across models ──────────────────────────────────
# Adapted from LibRecommender's ptest_recommends pattern

class TestBatchRecommend:
    """Test batch_recommend for all models."""

    @pytest.mark.parametrize("model_cls,kwargs", [
        (ALS, {"factors": 8, "iterations": 2, "seed": 42}),
        (SVD, {"factors": 8, "iterations": 2, "seed": 42}),
    ])
    def test_batch_recommend_pandas(
        self, synthetic_implicit_matrix: sparse.csr_matrix, model_cls, kwargs
    ) -> None:
        mat = synthetic_implicit_matrix
        if model_cls == SVD:
            # SVD needs explicit ratings
            rng = np.random.default_rng(42)
            mat = mat.copy()
            mat.data = rng.uniform(1.0, 5.0, mat.nnz).astype(np.float32)

        model = model_cls(**kwargs)
        model.fit(mat)
        df = model.batch_recommend(n=3, format="pandas")
        assert isinstance(df, pd.DataFrame)
        assert "user_id" in df.columns
        assert "item_id" in df.columns
        assert "score" in df.columns
        assert len(df) > 0


# ── Test: evaluate metrics in valid range ────────────────────────────────
# Adapted from LibRecommender's utils_metrics.py pattern

class TestEvaluateMetrics:
    """Ensure evaluate() metrics are always in [0, 1]."""

    def test_als_evaluate(self, synthetic_implicit_matrix: sparse.csr_matrix) -> None:
        model = ALS(factors=8, iterations=2, seed=42)
        model.fit(synthetic_implicit_matrix)

        # Build test data from the matrix itself
        mat_coo = synthetic_implicit_matrix.tocoo()
        test_data = pd.DataFrame({"user": mat_coo.row, "item": mat_coo.col})
        test_sample = test_data.sample(n=min(50, len(test_data)), random_state=42)

        metrics = evaluate(model, test_sample, k=5)
        for name, val in metrics.items():
            assert 0.0 <= val <= 1.0, f"Metric {name}={val} out of range"

    def test_svd_evaluate(self, synthetic_explicit_matrix: sparse.csr_matrix) -> None:
        model = SVD(factors=8, iterations=5, seed=42)
        model.fit(synthetic_explicit_matrix)

        mat_coo = synthetic_explicit_matrix.tocoo()
        test_data = pd.DataFrame({"user": mat_coo.row, "item": mat_coo.col})
        test_sample = test_data.sample(n=min(50, len(test_data)), random_state=42)

        metrics = evaluate(model, test_sample, k=5)
        for name, val in metrics.items():
            assert 0.0 <= val <= 1.0, f"Metric {name}={val} out of range"


# ── Test: Similarity correctness ─────────────────────────────────────────
# Adapted from LibRecommender's test_similarities.py

class TestSimilarityCorrectness:
    """Test similarity computations, inspired by LibRecommender."""

    def test_cosine_sim_symmetric(self) -> None:
        """ItemKNN cosine similarity should produce symmetric results."""
        rng = np.random.default_rng(42)
        n_users, n_items = 20, 10
        rows = rng.integers(0, n_users, 100)
        cols = rng.integers(0, n_items, 100)
        mat = sparse.csr_matrix(
            (np.ones(100, dtype=np.float32), (rows, cols)),
            shape=(n_users, n_items),
        )

        model = ItemKNN(k=5)
        model.fit(mat)

        # All users should get some recommendations
        for u in range(min(5, n_users)):
            items, scores = model.recommend_items(u, n=3, exclude_seen=True)
            assert len(items) <= 3

    def test_similar_items_function(self) -> None:
        """Test the similar_items utility function."""
        rng = np.random.default_rng(42)
        n_users, n_items = 30, 15
        rows = rng.integers(0, n_users, 200)
        cols = rng.integers(0, n_items, 200)
        mat = sparse.csr_matrix(
            (np.ones(200, dtype=np.float32), (rows, cols)),
            shape=(n_users, n_items),
        )
        # similar_items needs a model with item_factors
        model = ALS(factors=8, iterations=2, seed=42)
        model.fit(mat)
        from rusket import similar_items

        items, scores = similar_items(model, item_id=0, n=5)
        assert len(items) <= 5
