"""Comprehensive SIMD-opportunity benchmarks for rusket.

Covers all algorithms across a **wide variety of dataset sizes, densities, and
factor counts**, so the SIMD hot paths are exercised under realistic conditions.

Dataset matrix
--------------
  tiny:   ~1k users,  ~500 items,   10k  ratings  (unit-test scale)
  small:  ~943 users, ~1682 items,  100k ratings  (MovieLens-100k scale)
  medium: ~6k users,  ~4k items,    1M   ratings  (MovieLens-1M scale)
  large:  ~70k users, ~10k items,   5M   implicit  (production scale)

Run before SIMD changes (baseline):
    RUSTFLAGS="-C target-cpu=native" uv run maturin develop --release
    uv run pytest tests/benchmarks/test_bench_all.py --benchmark-only \\
        --benchmark-save=before_simd

Compare after SIMD changes:
    uv run pytest tests/benchmarks/test_bench_all.py --benchmark-only \\
        --benchmark-save=after_simd --benchmark-compare=before_simd
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import rusket._rusket as _rusket
from rusket import ALS, SVD
from rusket.bpr import BPR
from rusket.eclat import Eclat

# ════════════════════════════════════════════════════════════════════════════
# Fixtures — rating matrices
# ════════════════════════════════════════════════════════════════════════════


def _make_explicit(n_users: int, n_items: int, nnz: int, seed: int = 42) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, n_users, nnz)
    cols = rng.integers(0, n_items, nnz)
    vals = rng.uniform(1.0, 5.0, nnz).astype(np.float32)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))


def _make_implicit(n_users: int, n_items: int, nnz: int, seed: int = 99) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, n_users, nnz)
    cols = rng.integers(0, n_items, nnz)
    return sp.csr_matrix((np.ones(nnz, dtype=np.float32), (rows, cols)), shape=(n_users, n_items))


def _make_eclat_df(n_rows: int, n_items: int, density: float, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame((rng.random((n_rows, n_items)) < density).astype(np.uint8))


# ── explicit (ratings) ───────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def mat_tiny() -> sp.csr_matrix:
    """~1k users × 500 items, 10k ratings."""
    return _make_explicit(1_000, 500, 10_000)


@pytest.fixture(scope="module")
def mat_small() -> sp.csr_matrix:
    """943 users × 1 682 items, 100k ratings — MovieLens-100k scale."""
    return _make_explicit(943, 1_682, 100_000)


@pytest.fixture(scope="module")
def mat_medium() -> sp.csr_matrix:
    """6 040 users × 3 952 items, 1M ratings — MovieLens-1M scale."""
    return _make_explicit(6_040, 3_952, 1_000_000)


# ── implicit (interactions) ──────────────────────────────────────────────────


@pytest.fixture(scope="module")
def imp_small() -> sp.csr_matrix:
    """1k users × 2k items, 100k interactions."""
    return _make_implicit(1_000, 2_000, 100_000)


@pytest.fixture(scope="module")
def imp_medium() -> sp.csr_matrix:
    """10k users × 5k items, 500k interactions."""
    return _make_implicit(10_000, 5_000, 500_000)


@pytest.fixture(scope="module")
def imp_large() -> sp.csr_matrix:
    """70k users × 10k items, 5M interactions — production scale."""
    return _make_implicit(70_000, 10_000, 5_000_000)


# ── Eclat DataFrames at different scales ────────────────────────────────────


@pytest.fixture(scope="module")
def eclat_small_sparse() -> pd.DataFrame:
    """1k transactions × 100 items, density 5% (sparse)."""
    return _make_eclat_df(1_000, 100, 0.05)


@pytest.fixture(scope="module")
def eclat_medium_sparse() -> pd.DataFrame:
    """10k transactions × 200 items, density 10%."""
    return _make_eclat_df(10_000, 200, 0.10)


@pytest.fixture(scope="module")
def eclat_medium_dense() -> pd.DataFrame:
    """10k transactions × 200 items, density 30% — stresses popcount."""
    return _make_eclat_df(10_000, 200, 0.30)


@pytest.fixture(scope="module")
def eclat_large_sparse() -> pd.DataFrame:
    """100k transactions × 300 items, density 5% — shows u128 speedup."""
    return _make_eclat_df(100_000, 300, 0.05)


@pytest.fixture(scope="module")
def eclat_large_dense() -> pd.DataFrame:
    """100k transactions × 200 items, density 20%."""
    return _make_eclat_df(100_000, 200, 0.20)


# ════════════════════════════════════════════════════════════════════════════
# SVD (Funk SVD)  —  dot + update_factors (already 8-wide; sanity check)
# ════════════════════════════════════════════════════════════════════════════


def test_bench_svd_tiny_k32(benchmark, mat_tiny: sp.csr_matrix) -> None:
    """SVD tiny dataset, k=32 — overhead-dominated."""

    def _fit() -> SVD:
        m = SVD(factors=32, iterations=10, learning_rate=0.005, regularization=0.02, seed=42)
        m.fit(mat_tiny)
        return m

    result = benchmark(_fit)
    assert result._fitted


def test_bench_svd_small_k64(benchmark, mat_small: sp.csr_matrix) -> None:
    """SVD 100k ratings, k=64 — primary benchmark."""

    def _fit() -> SVD:
        m = SVD(factors=64, iterations=20, learning_rate=0.005, regularization=0.02, seed=42)
        m.fit(mat_small)
        return m

    result = benchmark(_fit)
    assert result._fitted


def test_bench_svd_small_k128(benchmark, mat_small: sp.csr_matrix) -> None:
    """SVD 100k ratings, k=128 — wider vectors."""

    def _fit() -> SVD:
        m = SVD(factors=128, iterations=20, learning_rate=0.005, regularization=0.02, seed=42)
        m.fit(mat_small)
        return m

    result = benchmark(_fit)
    assert result._fitted


def test_bench_svd_medium_k64(benchmark, mat_medium: sp.csr_matrix) -> None:
    """SVD 1M ratings, k=64 — MovieLens-1M scale."""

    def _fit() -> SVD:
        m = SVD(factors=64, iterations=10, learning_rate=0.005, regularization=0.02, seed=42)
        m.fit(mat_medium)
        return m

    result = benchmark(_fit)
    assert result._fitted


# ════════════════════════════════════════════════════════════════════════════
# SVD++ — 5 vectorized inner loops
# ════════════════════════════════════════════════════════════════════════════


def _svdpp_inputs(mat: sp.csr_matrix):
    mat = mat.astype(np.float32)
    mat.sort_indices()
    return (
        mat.indptr.astype(np.int64),
        mat.indices.astype(np.int32),
        mat.data,
        *mat.shape,
    )


def test_bench_svdpp_tiny_k32(benchmark, mat_tiny: sp.csr_matrix) -> None:
    """SVD++ tiny dataset, k=32, 10 iters."""
    ip, ix, d, nu, ni = _svdpp_inputs(mat_tiny)
    benchmark(lambda: _rusket.svdpp_fit(ip, ix, d, nu, ni, 32, 0.005, 0.02, 10, 42, False))


def test_bench_svdpp_small_k32(benchmark, mat_small: sp.csr_matrix) -> None:
    """SVD++ 100k ratings, k=32, 10 iters."""
    ip, ix, d, nu, ni = _svdpp_inputs(mat_small)
    benchmark(lambda: _rusket.svdpp_fit(ip, ix, d, nu, ni, 32, 0.005, 0.02, 10, 42, False))


def test_bench_svdpp_small_k64(benchmark, mat_small: sp.csr_matrix) -> None:
    """SVD++ 100k ratings, k=64, 10 iters — primary SIMD target."""
    ip, ix, d, nu, ni = _svdpp_inputs(mat_small)
    benchmark(lambda: _rusket.svdpp_fit(ip, ix, d, nu, ni, 64, 0.005, 0.02, 10, 42, False))


def test_bench_svdpp_medium_k32(benchmark, mat_medium: sp.csr_matrix) -> None:
    """SVD++ 1M ratings, k=32, 5 iters — shows y_j loop cost at scale."""
    ip, ix, d, nu, ni = _svdpp_inputs(mat_medium)
    benchmark(lambda: _rusket.svdpp_fit(ip, ix, d, nu, ni, 32, 0.005, 0.02, 5, 42, False))


# ════════════════════════════════════════════════════════════════════════════
# BPR — 8-wide triad update
# ════════════════════════════════════════════════════════════════════════════


def test_bench_bpr_small_k32(benchmark, imp_small: sp.csr_matrix) -> None:
    """BPR 100k interactions, k=32, 20 iters."""

    def _fit() -> BPR:
        m = BPR(factors=32, iterations=20, learning_rate=0.01, regularization=0.01, seed=42)
        m.fit(imp_small)
        return m

    result = benchmark(_fit)
    assert result.fitted


def test_bench_bpr_small_k64(benchmark, imp_small: sp.csr_matrix) -> None:
    """BPR 100k interactions, k=64, 20 iters — primary SIMD target."""

    def _fit() -> BPR:
        m = BPR(factors=64, iterations=20, learning_rate=0.01, regularization=0.01, seed=42)
        m.fit(imp_small)
        return m

    result = benchmark(_fit)
    assert result.fitted


def test_bench_bpr_small_k128(benchmark, imp_small: sp.csr_matrix) -> None:
    """BPR 100k interactions, k=128, 20 iters."""

    def _fit() -> BPR:
        m = BPR(factors=128, iterations=20, learning_rate=0.01, regularization=0.01, seed=42)
        m.fit(imp_small)
        return m

    result = benchmark(_fit)
    assert result.fitted


def test_bench_bpr_medium_k64(benchmark, imp_medium: sp.csr_matrix) -> None:
    """BPR 500k interactions, k=64, 10 iters."""

    def _fit() -> BPR:
        m = BPR(factors=64, iterations=10, learning_rate=0.01, regularization=0.01, seed=42)
        m.fit(imp_medium)
        return m

    result = benchmark(_fit)
    assert result.fitted


# ════════════════════════════════════════════════════════════════════════════
# ALS — CG apply_a mat-vec (4-wide unroll)
# ════════════════════════════════════════════════════════════════════════════


def test_bench_als_small_k32_cg(benchmark, imp_small: sp.csr_matrix) -> None:
    """ALS CG, 100k interactions, k=32, 15 iters, cg_iters=10."""

    def _fit() -> ALS:
        m = ALS(factors=32, iterations=15, regularization=0.01, alpha=40.0, cg_iters=10, seed=42)
        m.fit(imp_small)
        return m

    result = benchmark(_fit)
    assert result.fitted


def test_bench_als_small_k64_cg(benchmark, imp_small: sp.csr_matrix) -> None:
    """ALS CG, 100k interactions, k=64, 15 iters — primary SIMD target."""

    def _fit() -> ALS:
        m = ALS(factors=64, iterations=15, regularization=0.01, alpha=40.0, cg_iters=10, seed=42)
        m.fit(imp_small)
        return m

    result = benchmark(_fit)
    assert result.fitted


def test_bench_als_small_k128_cg(benchmark, imp_small: sp.csr_matrix) -> None:
    """ALS CG, 100k interactions, k=128, 15 iters — wider gram mat-vec."""

    def _fit() -> ALS:
        m = ALS(factors=128, iterations=15, regularization=0.01, alpha=40.0, cg_iters=10, seed=42)
        m.fit(imp_small)
        return m

    result = benchmark(_fit)
    assert result.fitted


def test_bench_als_medium_k64_cg(benchmark, imp_medium: sp.csr_matrix) -> None:
    """ALS CG, 500k interactions, k=64, 10 iters."""

    def _fit() -> ALS:
        m = ALS(factors=64, iterations=10, regularization=0.01, alpha=40.0, cg_iters=10, seed=42)
        m.fit(imp_medium)
        return m

    result = benchmark(_fit)
    assert result.fitted


def test_bench_als_small_k64_cholesky(benchmark, imp_small: sp.csr_matrix) -> None:
    """ALS Cholesky, 100k interactions, k=64 — reference (faer handles this)."""

    def _fit() -> ALS:
        m = ALS(
            factors=64,
            iterations=15,
            regularization=0.01,
            alpha=40.0,
            use_cholesky=True,
            seed=42,
        )
        m.fit(imp_small)
        return m

    result = benchmark(_fit)
    assert result.fitted


# ════════════════════════════════════════════════════════════════════════════
# Eclat BitSet — u128 popcount speedup shows at large row counts
# ════════════════════════════════════════════════════════════════════════════


def test_bench_eclat_small_sparse(benchmark, eclat_small_sparse: pd.DataFrame) -> None:
    """Eclat 1k×100 items, density=5% — small baseline."""
    benchmark(lambda: Eclat(eclat_small_sparse, min_support=0.05).mine())


def test_bench_eclat_medium_sparse(benchmark, eclat_medium_sparse: pd.DataFrame) -> None:
    """Eclat 10k×200 items, density=10%."""
    benchmark(lambda: Eclat(eclat_medium_sparse, min_support=0.05).mine())


def test_bench_eclat_medium_dense(benchmark, eclat_medium_dense: pd.DataFrame) -> None:
    """Eclat 10k×200 items, density=30% — more pairs pass threshold."""
    benchmark(lambda: Eclat(eclat_medium_dense, min_support=0.05).mine())


def test_bench_eclat_large_sparse(benchmark, eclat_large_sparse: pd.DataFrame) -> None:
    """Eclat 100k×300 items, density=5% — primarily shows u128 advantage."""
    benchmark(lambda: Eclat(eclat_large_sparse, min_support=0.01).mine())


def test_bench_eclat_large_dense(benchmark, eclat_large_dense: pd.DataFrame) -> None:
    """Eclat 100k×200 items, density=20% — large dense BitSet, popcount heavy."""
    benchmark(lambda: Eclat(eclat_large_dense, min_support=0.05).mine())


def test_bench_eclat_large_maxlen2(benchmark, eclat_large_sparse: pd.DataFrame) -> None:
    """Eclat max_len=2 at 100k rows — intersection only, pure u128 speedup."""
    benchmark(lambda: Eclat(eclat_large_sparse, min_support=0.01, max_len=2).mine())
