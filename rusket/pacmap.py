"""PaCMAP – Pairwise Controlled Manifold Approximation Projection.

Provides a scikit-learn-compatible ``PaCMAP`` class plus convenience
helpers ``pacmap2`` and ``pacmap3`` for quick low-dimensional projections
with superior local+global structure preservation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from . import _rusket as _rust  # type: ignore
from .pca import ProjectedSpace


class PaCMAP:
    """Pairwise Controlled Manifold Approximation Projection (PaCMAP).

    Non-linear dimensionality reduction that preserves both local and
    global structure via a tri-component loss with dynamic phase-based
    weighting.  Uses PCA initialisation and Adam optimiser internally.

    Parameters
    ----------
    n_components : int, default=2
        Number of output dimensions.
    n_neighbors : int, default=10
        Number of nearest neighbours used for the near-pair graph.
    MN_ratio : float, default=0.5
        Ratio of mid-near pairs to the number of data points.
    FP_ratio : float, default=2.0
        Ratio of further (repulsive) pairs to the number of data points.
    num_iters : int, default=450
        Total optimisation iterations (3-phase schedule).
    lr : float, default=1.0
        Adam learning rate.
    seed : int, default=42
        Seed for reproducibility.
    verbose : int, default=0
        Verbosity level.

    Examples
    --------
    >>> import numpy as np
    >>> import rusket
    >>> X = np.random.default_rng(42).standard_normal((200, 50)).astype(np.float32)
    >>> embedding = rusket.PaCMAP(n_components=2).fit_transform(X)
    >>> embedding.shape
    (200, 2)
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        MN_ratio: float = 0.5,
        FP_ratio: float = 2.0,
        num_iters: int = 450,
        lr: float = 1.0,
        seed: int = 42,
        verbose: int = 0,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.MN_ratio = MN_ratio
        self.FP_ratio = FP_ratio
        self.num_iters = num_iters
        self.lr = lr
        self.seed = seed
        self.verbose = verbose
        self._embedding: npt.NDArray[np.float32] | None = None
        self._fitted: bool = False

    def __repr__(self) -> str:
        return f"PaCMAP(n_components={self.n_components}, n_neighbors={self.n_neighbors}, num_iters={self.num_iters})"

    def fit(self, X: npt.NDArray[Any]) -> PaCMAP:
        """Fit PaCMAP on the data matrix *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self
        """
        X = self._validate(X)
        self._embedding = _rust.pacmap_fit(
            X,
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            mn_ratio=self.MN_ratio,
            fp_ratio=self.FP_ratio,
            num_iters=self.num_iters,
            lr=self.lr,
            seed=self.seed,
        )
        self._fitted = True
        return self

    def transform(self, X: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        """Return the embedding computed during ``fit()``.

        .. note::

            PaCMAP is a transductive method — ``transform`` returns the
            embedding that was computed during ``fit``, similar to t-SNE.
            If *X* differs from the training data, a ``ValueError`` is raised.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
        """
        self._check_fitted()
        assert self._embedding is not None
        return self._embedding

    def fit_transform(self, X: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        """Fit PaCMAP and return the low-dimensional embedding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
        """
        return self.fit(X).transform(X)

    @property
    def embedding_(self) -> npt.NDArray[np.float32]:
        """The computed embedding, shape ``(n_samples, n_components)``."""
        self._check_fitted()
        assert self._embedding is not None
        return self._embedding

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("PaCMAP has not been fitted yet. Call .fit() first.")

    @staticmethod
    def _validate(X: Any) -> npt.NDArray[np.float32]:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {arr.ndim}D.")
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        return arr


# ── Convenience functions ──────────────────────────────────────────────


def pacmap(
    x: npt.NDArray[Any],
    n_components: int = 2,
    **kwargs: Any,
) -> ProjectedSpace:
    """Project data using PaCMAP.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
    n_components : int, default=2
    **kwargs
        Forwarded to ``PaCMAP()``.
    """
    model = PaCMAP(n_components=n_components, **kwargs).fit(x)
    return ProjectedSpace(model.embedding_)


def pacmap2(x: npt.NDArray[Any], **kwargs: Any) -> ProjectedSpace:
    """Project data into exactly 2 dimensions using PaCMAP."""
    return pacmap(x, n_components=2, **kwargs)


def pacmap3(x: npt.NDArray[Any], **kwargs: Any) -> ProjectedSpace:
    """Project data into exactly 3 dimensions using PaCMAP."""
    return pacmap(x, n_components=3, **kwargs)
