"""Tests for rusket.PaCMAP — validates Rust-backed PaCMAP."""

from __future__ import annotations

import numpy as np
import pytest

import rusket

RNG = np.random.default_rng(42)
X_SMALL = RNG.standard_normal((200, 20)).astype(np.float32)


# ── Shape tests ────────────────────────────────────────────────────────


def test_fit_shapes() -> None:
    model = rusket.PaCMAP(n_components=2)
    model.fit(X_SMALL)
    assert model.embedding_.shape == (200, 2)


def test_transform_shape() -> None:
    model = rusket.PaCMAP(n_components=2)
    model.fit(X_SMALL)
    Xt = model.transform(X_SMALL)
    assert Xt.shape == (200, 2)


def test_fit_transform_shape() -> None:
    model = rusket.PaCMAP(n_components=2)
    Xt = model.fit_transform(X_SMALL)
    assert Xt.shape == (200, 2)


def test_3d_shape() -> None:
    model = rusket.PaCMAP(n_components=3)
    Xt = model.fit_transform(X_SMALL)
    assert Xt.shape == (200, 3)


# ── Convenience functions ─────────────────────────────────────────────


def test_pacmap2_shape() -> None:
    result = rusket.pacmap2(X_SMALL)
    assert result.shape == (200, 2)


def test_pacmap3_shape() -> None:
    result = rusket.pacmap3(X_SMALL)
    assert result.shape == (200, 3)


def test_pacmap_function() -> None:
    result = rusket.pacmap(X_SMALL, n_components=2)
    assert result.shape == (200, 2)


# ── Determinism ────────────────────────────────────────────────────────


def test_deterministic() -> None:
    e1 = rusket.PaCMAP(n_components=2, seed=42).fit_transform(X_SMALL)
    e2 = rusket.PaCMAP(n_components=2, seed=42).fit_transform(X_SMALL)
    np.testing.assert_array_equal(e1, e2)


# ── Cluster separation ────────────────────────────────────────────────


def test_cluster_separation() -> None:
    """Three well-separated Gaussian blobs should remain separated after PaCMAP."""
    rng = np.random.default_rng(123)
    n_per = 80
    c1 = rng.standard_normal((n_per, 20)).astype(np.float32) + 10.0
    c2 = rng.standard_normal((n_per, 20)).astype(np.float32) - 10.0
    c3 = rng.standard_normal((n_per, 20)).astype(np.float32)
    c3[:, 0] += 20.0
    X = np.vstack([c1, c2, c3])
    labels = np.array([0] * n_per + [1] * n_per + [2] * n_per)

    embedding = rusket.PaCMAP(n_components=2, seed=42, num_iters=300).fit_transform(X)

    # Compute cluster centroids in 2D
    centroids = np.array([embedding[labels == i].mean(axis=0) for i in range(3)])

    # All inter-cluster distances should be > 0 (non-degenerate)
    for i in range(3):
        for j in range(i + 1, 3):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            assert dist > 0.1, f"Clusters {i} and {j} are too close: {dist}"


# ── Neighborhood preservation test ────────────────────────────────────


def test_knn_recall() -> None:
    """PaCMAP should preserve local neighborhoods for structured data."""
    from sklearn.neighbors import NearestNeighbors

    # Use structured data: 5 separated Gaussian blobs in 30D
    rng = np.random.default_rng(99)
    n_per = 60
    blobs = []
    for i in range(5):
        center = np.zeros(30, dtype=np.float32)
        center[i * 6 : (i + 1) * 6] = 8.0  # separate clusters in different subspaces
        blob = rng.standard_normal((n_per, 30)).astype(np.float32) + center
        blobs.append(blob)
    X = np.vstack(blobs)
    K = 10

    # Exact K-NN in high-dim
    nn_hd = NearestNeighbors(n_neighbors=K + 1, algorithm="brute").fit(X)
    _, hd_indices = nn_hd.kneighbors(X)
    hd_indices = hd_indices[:, 1:]  # exclude self

    # PaCMAP embedding
    embedding = rusket.PaCMAP(n_components=2, n_neighbors=K, seed=42).fit_transform(X)

    # Exact K-NN in low-dim
    nn_ld = NearestNeighbors(n_neighbors=K + 1, algorithm="brute").fit(embedding)
    _, ld_indices = nn_ld.kneighbors(embedding)
    ld_indices = ld_indices[:, 1:]

    # Compute recall
    recalls = []
    for i in range(len(X)):
        hd_set = set(hd_indices[i])
        ld_set = set(ld_indices[i])
        recalls.append(len(hd_set & ld_set) / K)

    avg_recall = np.mean(recalls)
    assert avg_recall > 0.15, f"KNN recall too low: {avg_recall:.3f}"


# ── Edge cases ─────────────────────────────────────────────────────────


def test_not_fitted_raises() -> None:
    model = rusket.PaCMAP(n_components=2)
    with pytest.raises(RuntimeError, match="not been fitted"):
        _ = model.embedding_


def test_small_dataset() -> None:
    """Should work on tiny datasets (n=10)."""
    X_tiny = RNG.standard_normal((10, 5)).astype(np.float32)
    embedding = rusket.PaCMAP(n_components=2, n_neighbors=3, seed=42).fit_transform(X_tiny)
    assert embedding.shape == (10, 2)
    assert np.all(np.isfinite(embedding))


def test_repr() -> None:
    model = rusket.PaCMAP(n_components=2, n_neighbors=10, num_iters=450)
    assert "PaCMAP" in repr(model)
    assert "n_components=2" in repr(model)


# ── No NaN/Inf output ─────────────────────────────────────────────────


def test_no_nan_inf() -> None:
    embedding = rusket.PaCMAP(n_components=2, seed=42).fit_transform(X_SMALL)
    assert np.all(np.isfinite(embedding)), "Embedding contains NaN or Inf"


# ── Plotting integration ──────────────────────────────────────────────


def test_plot_pacmap() -> None:
    """ProjectedSpace should support .plot()."""
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    result = rusket.pacmap2(X_SMALL)
    fig = result.plot(title="PaCMAP 2D")
    assert isinstance(fig, go.Figure)
