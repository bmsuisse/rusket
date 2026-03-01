import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from rusket import ApproximateNearestNeighbors


def test_ann_basic_functionality():
    # Generate some simple random data
    rng = np.random.RandomState(42)
    X = rng.rand(1000, 10).astype(np.float32)

    ann = ApproximateNearestNeighbors(n_trees=10, leaf_size=10, seed=42)
    ann.fit(X)

    # Query the exact same points
    indices, distances = ann.kneighbors(X[:10], n_neighbors=5)

    assert indices.shape == (10, 5)
    assert distances.shape == (10, 5)

    # The first neighbor of each point should be the point itself (distance == 0)
    for i in range(10):
        # We allow for floating point inaccuracies
        assert indices[i, 0] == i
        assert distances[i, 0] < 1e-5


def test_ann_recall_vs_sklearn():
    # Compare with exact nearest neighbors on a larger dataset to ensure recall is high
    rng = np.random.RandomState(42)
    X = rng.rand(5000, 20).astype(np.float32)

    # Fit approximate
    ann = ApproximateNearestNeighbors(n_trees=20, leaf_size=20, seed=42)
    ann.fit(X)

    queries = rng.rand(50, 20).astype(np.float32)
    approx_indices, approx_distances = ann.kneighbors(queries, n_neighbors=10)

    # Fit exact
    exact_nn = NearestNeighbors(n_neighbors=10, algorithm="brute")
    exact_nn.fit(X)
    exact_distances, exact_indices = exact_nn.kneighbors(queries)

    # Calculate recall: intersection of Top 10
    total_recall = 0
    for i in range(len(queries)):
        approx_set = set(approx_indices[i])
        exact_set = set(exact_indices[i])
        recall = len(approx_set.intersection(exact_set)) / 10.0
        total_recall += recall

    avg_recall = total_recall / len(queries)
    # With 20 trees we should easily get >80% recall on random data
    assert avg_recall > 0.80, f"Recall was too low: {avg_recall}"


def test_ann_errors_on_wrong_types():
    ann = ApproximateNearestNeighbors()
    with pytest.raises(ValueError):
        ann.fit(np.array([1, 2, 3]))  # 1D array
