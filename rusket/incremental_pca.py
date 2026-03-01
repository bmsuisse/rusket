import numpy as np

from ._rusket import RustIncrementalPCA


class IncrementalPCA:
    """Incremental Principal Component Analysis (IPCA).

    Linear dimensionality reduction using Singular Value Decomposition of the
    data, keeping only the most significant singular vectors to project the
    data to a lower dimensional space. The input data is centered but not scaled
    for each feature before applying the SVD.

    Depends on a custom fast Rust implementation for online SVD aggregation,
    enabling out-of-core scaling for extremely large datasets.

    Args:
        n_components (int): Number of components to keep.
        batch_size (int, default=None): The number of samples to use for each
            batch. Only used when calling `fit`. If None, defaults to `5 * n_features`.
    """

    def __init__(self, n_components: int, batch_size: int | None = None):
        self.n_components = n_components
        self.batch_size = batch_size
        self._ipca = None
        self.n_samples_seen_ = 0

    def partial_fit(self, X: np.ndarray, y=None) -> "IncrementalPCA":
        """Incremental fit with X. All of X is processed as a single batch.

        Args:
            X (np.ndarray): Array of shape (n_samples, n_features).
            y: Ignored.

        Returns:
            self
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        n_features = X.shape[1]

        if self._ipca is None:
            self._ipca = RustIncrementalPCA(self.n_components, n_features)

        self._ipca.partial_fit(X)
        self.n_samples_seen_ = self._ipca.n_samples_seen
        return self

    def fit(self, X: np.ndarray, y=None) -> "IncrementalPCA":
        """Fit the model with X, using minibatches of size `batch_size`.

        Args:
            X (np.ndarray): Array of shape (n_samples, n_features).
            y: Ignored.

        Returns:
            self
        """
        n_samples, n_features = X.shape
        batch_size = self.batch_size or max(5 * n_features, 1024)

        for i in range(0, n_samples, batch_size):
            X_batch = X[i:i + batch_size]
            self.partial_fit(X_batch)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction to X.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features).

        Returns:
            X_new (np.ndarray): Projected data of shape (n_samples, n_components).
        """
        if self._ipca is None:
            raise RuntimeError("Model has not been fitted.")

        X = np.ascontiguousarray(X, dtype=np.float32)
        mean = self.mean_
        components = self.components_

        # (X - mean) @ components.T
        return np.dot(X - mean, components.T)

    @property
    def components_(self) -> np.ndarray:
        """Principal axes in feature space, representing the directions of maximum variance."""
        return self._ipca.get_components()

    @property
    def singular_values_(self) -> np.ndarray:
        """The singular values corresponding to each of the selected components."""
        return self._ipca.get_singular_values()

    @property
    def mean_(self) -> np.ndarray:
        """Per-feature empirical mean, estimated from the training set."""
        return self._ipca.get_mean()
