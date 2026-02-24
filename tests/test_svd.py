"""Tests for SVD (Funk SVD / Biased SGD Matrix Factorization)."""

import numpy as np
import pytest
from scipy import sparse

from rusket import SVD

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def small_explicit_matrix():
    """Small explicit rating matrix for testing."""
    rng = np.random.default_rng(42)
    n_users, n_items = 50, 30
    nnz = 300
    rows = rng.integers(0, n_users, nnz)
    cols = rng.integers(0, n_items, nnz)
    vals = rng.uniform(1.0, 5.0, nnz).astype(np.float32)
    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    # Remove duplicates by converting to csr
    mat = mat.tocsr()
    return mat


# ── Basic fit / predict / recommend tests ─────────────────────────────────


class TestSVDBasic:
    def test_fit_sparse(self, small_explicit_matrix: sparse.csr_matrix) -> None:
        model = SVD(factors=16, iterations=5, seed=42)
        model.fit(small_explicit_matrix)
        assert model._fitted
        assert model.user_factors.shape == (50, 16)
        assert model.item_factors.shape == (30, 16)
        assert model.user_biases.shape == (50,)
        assert model.item_biases.shape == (30,)

    def test_predict(self, small_explicit_matrix: sparse.csr_matrix) -> None:
        model = SVD(factors=16, iterations=10, seed=42)
        model.fit(small_explicit_matrix)
        pred = model.predict(0, 0)
        assert isinstance(pred, float)
        assert not np.isnan(pred)

    def test_recommend_items(self, small_explicit_matrix: sparse.csr_matrix) -> None:
        model = SVD(factors=16, iterations=10, seed=42)
        model.fit(small_explicit_matrix)
        items, scores = model.recommend_items(0, n=5, exclude_seen=True)
        assert len(items) == 5
        assert len(scores) == 5
        # Scores should be sorted descending
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_recommend_items_no_exclude(self, small_explicit_matrix: sparse.csr_matrix) -> None:
        model = SVD(factors=16, iterations=5, seed=42)
        model.fit(small_explicit_matrix)
        items, scores = model.recommend_items(0, n=5, exclude_seen=False)
        assert len(items) == 5

    def test_recommend_users(self, small_explicit_matrix: sparse.csr_matrix) -> None:
        model = SVD(factors=16, iterations=5, seed=42)
        model.fit(small_explicit_matrix)
        users, scores = model.recommend_users(0, n=5)
        assert len(users) == 5
        assert len(scores) == 5

    def test_batch_recommend(self, small_explicit_matrix: sparse.csr_matrix) -> None:
        model = SVD(factors=16, iterations=5, seed=42)
        model.fit(small_explicit_matrix)
        df = model.batch_recommend(n=3, format="pandas")
        assert "user_id" in df.columns
        assert "item_id" in df.columns
        assert "score" in df.columns
        # 50 users * 3 items each (some may have fewer if all consumed)
        assert len(df) > 0

    def test_global_mean(self, small_explicit_matrix: sparse.csr_matrix) -> None:
        model = SVD(factors=16, iterations=5, seed=42)
        model.fit(small_explicit_matrix)
        gm = model.global_mean
        # Mean of uniform(1,5) should be ~3.0
        assert 1.0 < gm < 5.0

    def test_repr(self) -> None:
        model = SVD(factors=32, learning_rate=0.01, regularization=0.05, iterations=10)
        s = repr(model)
        assert "SVD" in s
        assert "32" in s


# ── Error handling tests ──────────────────────────────────────────────────


class TestSVDErrors:
    def test_not_fitted(self) -> None:
        model = SVD()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.recommend_items(0)

    def test_double_fit(self, small_explicit_matrix: sparse.csr_matrix) -> None:
        model = SVD(factors=8, iterations=2)
        model.fit(small_explicit_matrix)
        with pytest.raises(RuntimeError, match="already fitted"):
            model.fit(small_explicit_matrix)

    def test_predict_not_fitted(self) -> None:
        model = SVD()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(0, 0)


# ── from_transactions convenience ─────────────────────────────────────────


class TestSVDFromTransactions:
    def test_from_transactions(self) -> None:
        import pandas as pd

        rng = np.random.default_rng(42)
        n = 500
        df = pd.DataFrame(
            {
                "user": rng.integers(0, 50, n),
                "item": rng.integers(0, 30, n),
                "rating": rng.uniform(1.0, 5.0, n).astype(np.float32),
            }
        )
        model = SVD.from_transactions(
            df, user_col="user", item_col="item", rating_col="rating", factors=16, iterations=5
        ).fit()
        assert model._fitted
        items, scores = model.recommend_items(0, n=3)
        assert len(items) == 3


# ── RMSE convergence test ─────────────────────────────────────────────────


class TestSVDConvergence:
    def test_rmse_decreases(self, small_explicit_matrix: sparse.csr_matrix) -> None:
        """SVD should produce predictions closer to actual ratings after training."""
        model = SVD(factors=32, iterations=50, learning_rate=0.005, regularization=0.02, seed=42)
        model.fit(small_explicit_matrix)

        # Measure RMSE on training data
        mat = small_explicit_matrix.tocoo()
        errors = []
        for u, i, r in zip(mat.row, mat.col, mat.data, strict=True):
            pred = model.predict(int(u), int(i))
            errors.append((r - pred) ** 2)
        rmse = np.sqrt(np.mean(errors))
        # After 50 iterations, RMSE should be reasonable (< 2.0 for random data)
        assert rmse < 2.0, f"RMSE too high: {rmse}"
