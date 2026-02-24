"""Comprehensive SIMD-opportunity benchmarks for rusket.

Covers all algorithms with hot-path SIMD candidates:
  - SVD (Funk SVD): dot + update_factors (8-wide already; sanity check)
  - SVD++ (Koren 2008): sum_y + factor-update loops (scalar → SIMD target)
    NOTE: SVDpp benchmarks require svdpp_fit to be registered in lib.rs.
          They are present after the SIMD implementation step.
  - BPR: factor update triad (scalar → SIMD target)
  - ALS: CG inner mat-vec (scalar → SIMD target)
  - Eclat: BitSet intersect + popcount (u64 → u128 SIMD target)

Run baseline BEFORE any SIMD changes:
    RUSTFLAGS="-C target-cpu=native" uv run maturin develop --release
    uv run pytest tests/benchmarks/test_bench_all.py --benchmark-only \\
        --benchmark-save=before_simd

Compare after SIMD changes:
    uv run pytest tests/benchmarks/test_bench_all.py --benchmark-only \\
        --benchmark-save=after_simd --benchmark-compare=before_simd
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from rusket import ALS, SVD
from rusket.bpr import BPR
from rusket.eclat import Eclat


# ─── Shared fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rating_matrix_100k() -> sp.csr_matrix:
    """~943 users × 1682 items, ~100k explicit ratings (MovieLens-scale)."""
    rng = np.random.default_rng(42)
    n_users, n_items = 943, 1682
    nnz = 100_000
    rows = rng.integers(0, n_users, nnz)
    cols = rng.integers(0, n_items, nnz)
    vals = rng.uniform(1.0, 5.0, nnz).astype(np.float32)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))


@pytest.fixture(scope="module")
def implicit_matrix_100k() -> sp.csr_matrix:
    """~1000 users × 2000 items, 100k implicit binary interactions."""
    rng = np.random.default_rng(99)
    n_users, n_items = 1000, 2000
    nnz = 100_000
    rows = rng.integers(0, n_users, nnz)
    cols = rng.integers(0, n_items, nnz)
    vals = np.ones(nnz, dtype=np.float32)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))


@pytest.fixture(scope="module")
def dense_eclat_matrix_df():
    """10k transactions × 200 items, density ~10% — Eclat BitSet hot path."""
    import pandas as pd

    rng = np.random.default_rng(7)
    arr = (rng.random((10_000, 200)) > 0.9).astype(np.uint8)
    return pd.DataFrame(arr)


# ─── SVD (Funk SVD) ─────────────────────────────────────────────────────────


def test_bench_svd_k64(benchmark, rating_matrix_100k: sp.csr_matrix) -> None:
    """SVD fit: k=64, 20 iters — baseline for dot/update_factors."""

    def _fit() -> SVD:
        m = SVD(factors=64, iterations=20, learning_rate=0.005, regularization=0.02, seed=42)
        m.fit(rating_matrix_100k)
        return m

    result = benchmark(_fit)
    assert result._fitted


def test_bench_svd_k128(benchmark, rating_matrix_100k: sp.csr_matrix) -> None:
    """SVD fit: k=128, 20 iters — wider vectors stress-test 8-wide dot."""

    def _fit() -> SVD:
        m = SVD(factors=128, iterations=20, learning_rate=0.005, regularization=0.02, seed=42)
        m.fit(rating_matrix_100k)
        return m

    result = benchmark(_fit)
    assert result._fitted


# ─── BPR ───────────────────────────────────────────────────────────────────


def test_bench_bpr_k64(benchmark, implicit_matrix_100k: sp.csr_matrix) -> None:
    """BPR fit: k=64, 20 iters — 3-array factor update triad SIMD target."""

    def _fit() -> BPR:
        m = BPR(factors=64, iterations=20, learning_rate=0.01, regularization=0.01, seed=42)
        m.fit(implicit_matrix_100k)
        return m

    result = benchmark(_fit)
    assert result.fitted


def test_bench_bpr_k128(benchmark, implicit_matrix_100k: sp.csr_matrix) -> None:
    """BPR fit: k=128, 20 iters — wider vectors amplify update-triad cost."""

    def _fit() -> BPR:
        m = BPR(factors=128, iterations=20, learning_rate=0.01, regularization=0.01, seed=42)
        m.fit(implicit_matrix_100k)
        return m

    result = benchmark(_fit)
    assert result.fitted


# ─── ALS ────────────────────────────────────────────────────────────────────


def test_bench_als_k64_cg(benchmark, implicit_matrix_100k: sp.csr_matrix) -> None:
    """ALS fit CG: k=64, 15 iters — apply_a gramian mat-vec inner loop."""

    def _fit() -> ALS:
        m = ALS(factors=64, iterations=15, regularization=0.01, alpha=40.0, cg_iters=10, seed=42)
        m.fit(implicit_matrix_100k)
        return m

    result = benchmark(_fit)
    assert result.fitted


def test_bench_als_k64_cholesky(benchmark, implicit_matrix_100k: sp.csr_matrix) -> None:
    """ALS fit Cholesky: k=64, 15 iters — reference (faer; no custom loop)."""

    def _fit() -> ALS:
        m = ALS(
            factors=64,
            iterations=15,
            regularization=0.01,
            alpha=40.0,
            use_cholesky=True,
            seed=42,
        )
        m.fit(implicit_matrix_100k)
        return m

    result = benchmark(_fit)
    assert result.fitted


# ─── Eclat ──────────────────────────────────────────────────────────────────


def test_bench_eclat_dense(benchmark, dense_eclat_matrix_df) -> None:
    """Eclat on 10k×200 dense matrix — BitSet intersect + popcount hot path."""

    def _mine() -> None:
        Eclat(dense_eclat_matrix_df, min_support=0.05).mine()

    benchmark(_mine)


def test_bench_eclat_dense_maxlen2(benchmark, dense_eclat_matrix_df) -> None:
    """Eclat max_len=2 — intersection counting only, exposes u64/u128 diff."""

    def _mine() -> None:
        Eclat(dense_eclat_matrix_df, min_support=0.05, max_len=2).mine()

    benchmark(_mine)
