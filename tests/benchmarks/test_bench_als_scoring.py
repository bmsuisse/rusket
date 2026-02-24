"""pytest-benchmark: ALS / BPR fit + scoring performance.

Establishes a wall-time baseline covering:
  - ALS fit (CG solver, various factor sizes)
  - ALS recommend_items  (single-user scoring, linear scan)
  - ALS batch_recommend  (all users — reveals GEMM bottleneck)
  - BPR fit              (Hogwild parallel SGD)
  - BPR recommend_items

Run the baseline (before optimisations):
    uv run pytest tests/benchmarks/test_bench_als_scoring.py -v \\
        --benchmark-only --benchmark-sort=name \\
        --benchmark-json=benchmarks/baseline_als.json

After optimisations, compare:
    uv run pytest tests/benchmarks/test_bench_als_scoring.py -v \\
        --benchmark-only \\
        --benchmark-compare=benchmarks/baseline_als.json
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import rusket
from rusket import ALS, BPR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ml100k_mat() -> sparse.csr_matrix:
    """Synthetic MovieLens-100k-scale sparse matrix (943 users × 1682 items)."""
    rng = np.random.default_rng(42)
    n_users, n_items, nnz = 943, 1682, 100_000
    rows = rng.integers(0, n_users, nnz)
    cols = rng.integers(0, n_items, nnz)
    vals = np.ones(nnz, dtype=np.float32)
    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    mat.sum_duplicates()
    return mat


@pytest.fixture(scope="module")
def ml100k_large() -> sparse.csr_matrix:
    """Larger matrix for batch-scoring benchmarks (5000 users × 10000 items)."""
    rng = np.random.default_rng(7)
    n_users, n_items = 5_000, 10_000
    nnz = 500_000
    rows = rng.integers(0, n_users, nnz)
    cols = rng.integers(0, n_items, nnz)
    vals = np.ones(nnz, dtype=np.float32)
    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    mat.sum_duplicates()
    return mat


@pytest.fixture(scope="module")
def als_fitted(ml100k_mat: sparse.csr_matrix) -> ALS:
    """Pre-fitted ALS model for scoring benchmarks."""
    model = ALS(factors=64, iterations=15, regularization=0.01, alpha=40.0, seed=42)
    model.fit(ml100k_mat)
    return model


@pytest.fixture(scope="module")
def als_fitted_large(ml100k_large: sparse.csr_matrix) -> ALS:
    """Pre-fitted ALS model on larger matrix for batch-scoring benchmarks."""
    model = ALS(factors=64, iterations=5, regularization=0.01, alpha=40.0, seed=42)
    model.fit(ml100k_large)
    return model


@pytest.fixture(scope="module")
def bpr_fitted(ml100k_mat: sparse.csr_matrix) -> BPR:
    """Pre-fitted BPR model for scoring benchmarks."""
    model = BPR(factors=64, iterations=10, learning_rate=0.01, regularization=0.01, seed=42)
    model.fit(ml100k_mat)
    return model


# ---------------------------------------------------------------------------
# ALS fit benchmarks
# ---------------------------------------------------------------------------


def test_als_fit_k32(benchmark, ml100k_mat: sparse.csr_matrix) -> None:
    """ALS fit: 32 factors, 15 iterations (ML-100k scale)."""
    def _fit() -> ALS:
        m = ALS(factors=32, iterations=15, regularization=0.01, alpha=40.0, seed=42)
        m.fit(ml100k_mat)
        return m
    result = benchmark(_fit)
    assert result.fitted


def test_als_fit_k64(benchmark, ml100k_mat: sparse.csr_matrix) -> None:
    """ALS fit: 64 factors, 15 iterations (ML-100k scale)."""
    def _fit() -> ALS:
        m = ALS(factors=64, iterations=15, regularization=0.01, alpha=40.0, seed=42)
        m.fit(ml100k_mat)
        return m
    result = benchmark(_fit)
    assert result.fitted


def test_als_fit_k128(benchmark, ml100k_mat: sparse.csr_matrix) -> None:
    """ALS fit: 128 factors, 15 iterations."""
    def _fit() -> ALS:
        m = ALS(factors=128, iterations=15, regularization=0.01, alpha=40.0, seed=42)
        m.fit(ml100k_mat)
        return m
    result = benchmark(_fit)
    assert result.fitted


# ---------------------------------------------------------------------------
# ALS scoring benchmarks — single user
# ---------------------------------------------------------------------------


def test_als_recommend_single_user(benchmark, als_fitted: ALS) -> None:
    """ALS recommend_items: single user, top-10 from 1682 items.

    This exercises the sequential dot-product scan (candidate for GEMV).
    """
    result = benchmark(als_fitted.recommend_items, user_id=0, n=10, exclude_seen=True)
    ids, scores = result
    assert len(ids) >= 1


def test_als_recommend_top50(benchmark, als_fitted: ALS) -> None:
    """ALS recommend_items: single user, top-50 from 1682 items."""
    result = benchmark(als_fitted.recommend_items, user_id=42, n=50, exclude_seen=False)
    ids, scores = result
    assert len(ids) >= 1


# ---------------------------------------------------------------------------
# ALS batch scoring — main bottleneck for large catalogues
# ---------------------------------------------------------------------------


def test_als_batch_recommend_small(benchmark, als_fitted: ALS) -> None:
    """ALS batch_recommend: all 943 users × 1682 items — top-10 each.

    This is the hot path that should benefit most from GEMM.
    """
    def _batch() -> object:
        return als_fitted.batch_recommend(n=10, exclude_seen=False, format="pandas")
    result = benchmark(_batch)
    assert result is not None


def test_als_batch_recommend_large(benchmark, als_fitted_large: ALS) -> None:
    """ALS batch_recommend: 5000 users × 10000 items — top-10 each.

    Large-scale scoring — primary candidate for GEMM optimisation.
    """
    def _batch() -> object:
        return als_fitted_large.batch_recommend(n=10, exclude_seen=False, format="pandas")
    result = benchmark(_batch)
    assert result is not None


# ---------------------------------------------------------------------------
# BPR fit benchmarks
# ---------------------------------------------------------------------------


def test_bpr_fit_k64(benchmark, ml100k_mat: sparse.csr_matrix) -> None:
    """BPR fit: 64 factors, 10 iterations (ML-100k scale)."""
    def _fit() -> BPR:
        m = BPR(factors=64, iterations=10, learning_rate=0.01, regularization=0.01, seed=42)
        m.fit(ml100k_mat)
        return m
    result = benchmark(_fit)
    assert result.fitted


def test_bpr_fit_k128(benchmark, ml100k_mat: sparse.csr_matrix) -> None:
    """BPR fit: 128 factors, 10 iterations."""
    def _fit() -> BPR:
        m = BPR(factors=128, iterations=10, learning_rate=0.01, regularization=0.01, seed=42)
        m.fit(ml100k_mat)
        return m
    result = benchmark(_fit)
    assert result.fitted


# ---------------------------------------------------------------------------
# BPR scoring benchmarks
# ---------------------------------------------------------------------------


def test_bpr_recommend_single_user(benchmark, bpr_fitted: BPR) -> None:
    """BPR recommend_items: single user, top-10."""
    result = benchmark(bpr_fitted.recommend_items, user_id=0, n=10, exclude_seen=True)
    ids, scores = result
    assert len(ids) >= 1
