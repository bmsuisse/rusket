"""Benchmark tests for SVD (Funk SVD / Biased SGD) fit performance.

Tracks wall-time for SVD.fit() on a ~100k-rating sparse matrix.

Run:
    uv run pytest tests/benchmarks/test_bench_svd.py -v --benchmark-only
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from rusket import SVD


@pytest.fixture(scope="module")
def svd_benchmark_matrix() -> sparse.csr_matrix:
    """Generate a MovieLens-scale sparse rating matrix for benchmarking.

    ~943 users, ~1682 items, ~100k explicit ratings (density ~6.3%).
    """
    rng = np.random.default_rng(42)
    n_users, n_items = 943, 1682
    nnz = 100_000
    rows = rng.integers(0, n_users, nnz)
    cols = rng.integers(0, n_items, nnz)
    vals = rng.uniform(1.0, 5.0, nnz).astype(np.float32)
    mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    return mat


def test_svd_fit_f64_k64(benchmark, svd_benchmark_matrix: sparse.csr_matrix) -> None:
    """Benchmark SVD fit with 64 factors, 20 iterations."""

    def _fit():
        model = SVD(factors=64, iterations=20, learning_rate=0.005, regularization=0.02, seed=42)
        model.fit(svd_benchmark_matrix)
        return model

    result = benchmark(_fit)
    assert result._fitted


def test_svd_fit_f32_k32(benchmark, svd_benchmark_matrix: sparse.csr_matrix) -> None:
    """Benchmark SVD fit with 32 factors, 10 iterations (smaller config)."""

    def _fit():
        model = SVD(factors=32, iterations=10, learning_rate=0.005, regularization=0.02, seed=42)
        model.fit(svd_benchmark_matrix)
        return model

    result = benchmark(_fit)
    assert result._fitted


def test_svd_fit_f128_k128(benchmark, svd_benchmark_matrix: sparse.csr_matrix) -> None:
    """Benchmark SVD fit with 128 factors, 20 iterations (heavy config)."""

    def _fit():
        model = SVD(factors=128, iterations=20, learning_rate=0.005, regularization=0.02, seed=42)
        model.fit(svd_benchmark_matrix)
        return model

    result = benchmark(_fit)
    assert result._fitted
