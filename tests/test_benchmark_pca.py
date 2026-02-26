"""Benchmarks: rusket PCA vs scikit-learn PCA at multiple data sizes."""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pytest

import rusket

try:
    from sklearn.decomposition import PCA as SkPCA

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

# float32 is used natively by standard_normal() and rusket.PCA
X_TINY = RNG.standard_normal((1_000, 20)).astype(np.float32)
X_SMALL = RNG.standard_normal((10_000, 50)).astype(np.float32)
X_MEDIUM = RNG.standard_normal((100_000, 100)).astype(np.float32)
X_LARGE = RNG.standard_normal((500_000, 200)).astype(np.float32)


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


@pytest.mark.benchmark(group="pca_tiny")
def test_benchmark_pca_rusket_tiny(benchmark) -> None:
    result = benchmark(rusket.pca, X_TINY, n_components=5)
    assert result.shape == (1_000, 5)


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
@pytest.mark.benchmark(group="pca_tiny")
def test_benchmark_pca_sklearn_tiny(benchmark) -> None:
    result = benchmark(SkPCA(n_components=5).fit_transform, X_TINY)
    assert result.shape == (1_000, 5)


@pytest.mark.benchmark(group="pca_small")
def test_benchmark_pca_rusket_small(benchmark) -> None:
    result = benchmark(rusket.pca, X_SMALL, n_components=10)
    assert result.shape == (10_000, 10)


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
@pytest.mark.benchmark(group="pca_small")
def test_benchmark_pca_sklearn_small(benchmark) -> None:
    result = benchmark(SkPCA(n_components=10).fit_transform, X_SMALL)
    assert result.shape == (10_000, 10)


@pytest.mark.benchmark(group="pca_medium")
def test_benchmark_pca_rusket_medium(benchmark) -> None:
    result = benchmark(rusket.pca, X_MEDIUM, n_components=20)
    assert result.shape == (100_000, 20)


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
@pytest.mark.benchmark(group="pca_medium")
def test_benchmark_pca_sklearn_medium(benchmark) -> None:
    result = benchmark(SkPCA(n_components=20).fit_transform, X_MEDIUM)
    assert result.shape == (100_000, 20)


@pytest.mark.benchmark(group="pca_large")
def test_benchmark_pca_rusket_large_skip(benchmark) -> None:
    result = benchmark(rusket.pca, X_LARGE, n_components=50)
    assert result.shape == (500_000, 50)


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
@pytest.mark.benchmark(group="pca_large")
def test_benchmark_pca_sklearn_large_skip(benchmark) -> None:
    result = benchmark(SkPCA(n_components=50).fit_transform, X_LARGE)
    assert result.shape == (500_000, 50)


# ---------------------------------------------------------------------------
# Head-to-head comparisons vs sklearn (with memory tracking)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
def test_vs_sklearn_small() -> None:
    _, ours, our_mem = _timed(rusket.pca, X_SMALL, n_components=10)
    _, sk, sk_mem = _timed(SkPCA(n_components=10).fit_transform, X_SMALL)
    print(
        f"\\n[small ] ours={ours * 1000:.1f}ms  sklearn={sk * 1000:.1f}ms  speedup={sk / ours:.1f}x  "
        f"mem ours={our_mem / 1e3:.0f}KB  sklearn={sk_mem / 1e3:.0f}KB"
    )


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
def test_vs_sklearn_medium() -> None:
    _, ours, our_mem = _timed(rusket.pca, X_MEDIUM, n_components=20)
    _, sk, sk_mem = _timed(SkPCA(n_components=20).fit_transform, X_MEDIUM)
    print(
        f"\\n[medium] ours={ours:.3f}s  sklearn={sk:.3f}s  speedup={sk / ours:.1f}x  "
        f"mem ours={our_mem / 1e6:.1f}MB  sklearn={sk_mem / 1e6:.1f}MB"
    )


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not installed")
def test_vs_sklearn_large() -> None:
    _, ours, our_mem = _timed(rusket.pca, X_LARGE, n_components=50)
    _, sk, sk_mem = _timed(SkPCA(n_components=50).fit_transform, X_LARGE)
    print(
        f"\\n[large ] ours={ours:.3f}s  sklearn={sk:.3f}s  speedup={sk / ours:.1f}x  "
        f"mem ours={our_mem / 1e6:.1f}MB  sklearn={sk_mem / 1e6:.1f}MB"
    )
