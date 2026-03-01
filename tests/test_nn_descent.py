"""Tests for NN-Descent k-NN graph builder."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from rusket._rusket import nn_descent_build

RNG = np.random.default_rng(42)


def test_basic_shapes() -> None:
    X = RNG.standard_normal((100, 10)).astype(np.float32)
    indices, distances = nn_descent_build(X, k=5)
    assert indices.shape == (100, 5)
    assert distances.shape == (100, 5)


def test_self_is_nearest() -> None:
    """When querying the training data, the nearest neighbor should be found correctly."""
    X = RNG.standard_normal((200, 10)).astype(np.float32)
    indices, distances = nn_descent_build(X, k=10, seed=42)

    # Each point's actual nearest neighbor should appear in its k-NN list
    nn = NearestNeighbors(n_neighbors=2, algorithm="brute").fit(X)
    _, exact_idx = nn.kneighbors(X)
    true_nn = exact_idx[:, 1]  # exclude self

    found = 0
    for i in range(len(X)):
        if true_nn[i] in indices[i]:
            found += 1

    recall_1nn = found / len(X)
    assert recall_1nn > 0.85, f"1-NN recall too low: {recall_1nn:.3f}"


def test_recall_vs_brute_force() -> None:
    """NN-Descent should achieve >90% recall at k=10 on random data."""
    X = RNG.standard_normal((2000, 20)).astype(np.float32)
    K = 10

    approx_indices, _ = nn_descent_build(X, k=K, seed=42)

    exact_nn = NearestNeighbors(n_neighbors=K + 1, algorithm="brute").fit(X)
    _, exact_indices = exact_nn.kneighbors(X)
    exact_indices = exact_indices[:, 1:]  # drop self

    total_recall = 0.0
    for i in range(len(X)):
        approx_set = {int(x) for x in approx_indices[i] if x != 2**32 - 1}
        exact_set = {int(x) for x in exact_indices[i]}
        total_recall += len(approx_set & exact_set) / K

    avg_recall = total_recall / len(X)
    assert avg_recall > 0.85, f"Average recall too low: {avg_recall:.3f}"


def test_distances_sorted() -> None:
    """Distances should be returned in ascending order per point."""
    X = RNG.standard_normal((500, 15)).astype(np.float32)
    _, distances = nn_descent_build(X, k=10, seed=42)

    for i in range(len(X)):
        dists = distances[i]
        valid = dists[dists < np.inf]
        assert np.all(valid[:-1] <= valid[1:]), f"Distances not sorted for point {i}"


def test_distances_nonnegative() -> None:
    X = RNG.standard_normal((100, 5)).astype(np.float32)
    _, distances = nn_descent_build(X, k=5, seed=42)
    assert np.all(distances >= 0), "Negative distances found"


def test_small_dataset() -> None:
    """Should handle very small datasets."""
    X = RNG.standard_normal((5, 3)).astype(np.float32)
    indices, distances = nn_descent_build(X, k=2, seed=42)
    assert indices.shape == (5, 2)
    assert distances.shape == (5, 2)


def test_error_on_k_too_large() -> None:
    X = RNG.standard_normal((10, 5)).astype(np.float32)
    with pytest.raises(ValueError):
        nn_descent_build(X, k=10)


def test_error_on_1d() -> None:
    X = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises((ValueError, TypeError)):
        nn_descent_build(X, k=1)  # type: ignore[arg-type]


def test_deterministic() -> None:
    X = RNG.standard_normal((200, 10)).astype(np.float32)
    idx1, dist1 = nn_descent_build(X, k=5, seed=42)
    idx2, dist2 = nn_descent_build(X, k=5, seed=42)
    np.testing.assert_array_equal(idx1, idx2)
    np.testing.assert_array_equal(dist1, dist2)
