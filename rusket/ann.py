import numpy as np

from ._rusket import AnnIndex as _AnnIndex


class ApproximateNearestNeighbors:
    """Approximate Nearest Neighbors using Random Projection Forests.

    This class wraps the fast Rust implementation of Random Projection Forests
    for approximate nearest neighbor search. It builds multiple trees and queries
    them in parallel to achieve high recall very quickly.

    Args:
        n_trees (int, default=10): Number of random projection trees to build.
            More trees increase recall but also increase memory and build time.
        leaf_size (int, default=30): Maximum number of points in a leaf node.
        seed (int, default=42): Random seed for reproducibility.
    """

    def __init__(self, n_trees: int = 10, leaf_size: int = 30, seed: int = 42):
        self.n_trees = n_trees
        self.leaf_size = leaf_size
        self.seed = seed
        self._index = None

    def fit(self, X: np.ndarray) -> "ApproximateNearestNeighbors":
        """Build the index on the data X.

        Args:
            X (np.ndarray): Array of shape (n_samples, n_features).
                Must be of type float32. Data is not copied if it is
                C-contiguous float32, otherwise a copy is made.

        Returns:
            self
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        self._index = _AnnIndex(
            data=X,
            n_trees=self.n_trees,
            leaf_size=self.leaf_size,
            seed=self.seed,
        )
        return self

    def kneighbors(self, X: np.ndarray, n_neighbors: int, search_k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Find the K-nearest neighbors of points in X.

        Args:
            X (np.ndarray): Query points of shape (n_queries, n_features).
            n_neighbors (int): Number of neighbors to return.
            search_k (int | None, default=None): Maximum number of nodes to search
                across all trees. Higher values yield better recall at the cost of
                search time. If None, defaults to `n_neighbors * n_trees * 2`.

        Returns:
            neighbors (np.ndarray): Indices of the nearest neighbors,
                shape (n_queries, n_neighbors).
            distances (np.ndarray): Euclidean distances to the nearest neighbors,
                shape (n_queries, n_neighbors).
        """
        if self._index is None:
            raise RuntimeError("Index has not been built. Call fit() first.")

        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        indices, distances = self._index.kneighbors(X, n_neighbors, search_k)
        return indices, distances
