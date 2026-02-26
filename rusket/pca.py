"""PCA (Principal Component Analysis) via Rust-backed SVD.

Provides a scikit-learn-compatible ``PCA`` class plus one-shot convenience
helpers ``pca``, ``pca2``, and ``pca3`` for quick dimensionality reduction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from . import _rusket as _rust  # type: ignore


class PCA:
    """Principal Component Analysis (PCA).

    Linear dimensionality reduction using Singular Value Decomposition
    of the centred data, computed entirely in Rust via the ``faer`` crate.

    Parameters
    ----------
    n_components : int
        Number of principal components to keep.

    Attributes (available after ``fit()``)
    ----------
    components_ : np.ndarray
        Principal axes in feature space, shape ``(n_components, n_features)``.
    explained_variance_ : np.ndarray
        Variance explained per component (uses ``n - 1`` degrees of freedom).
    explained_variance_ratio_ : np.ndarray
        Fraction of total variance explained per component.
    singular_values_ : np.ndarray
        Singular values corresponding to each component.
    mean_ : np.ndarray
        Per-feature empirical mean estimated from the training data.
    n_components_ : int
        Number of components that were actually fitted (may be less than
        requested if ``n_components > min(n_samples, n_features)``).

    Examples
    --------
    >>> import numpy as np
    >>> import rusket
    >>> X = np.random.default_rng(42).standard_normal((100, 10)).astype(np.float32)
    >>> pca = rusket.PCA(n_components=3)
    >>> pca.fit(X)
    PCA(n_components=3)
    >>> pca.transform(X).shape
    (100, 3)
    >>> pca.explained_variance_ratio_.sum()  # close to fraction of total
    0.4...
    """

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self._components: npt.NDArray[np.float32] | None = None
        self._explained_variance: npt.NDArray[np.float32] | None = None
        self._explained_variance_ratio: npt.NDArray[np.float32] | None = None
        self._singular_values: npt.NDArray[np.float32] | None = None
        self._mean: npt.NDArray[np.float32] | None = None
        self._n_components: int | None = None
        self._fitted: bool = False

    def __repr__(self) -> str:
        return f"PCA(n_components={self.n_components})"

    # ── Fit ────────────────────────────────────────────────────────────
    def fit(self, X: npt.NDArray[Any]) -> PCA:
        """Fit PCA on the data matrix ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self
        """
        X = self._validate(X)
        comp, ev, evr, sv, mean = _rust.pca_fit(X, self.n_components)
        self._components = comp
        self._explained_variance = ev
        self._explained_variance_ratio = evr
        self._singular_values = sv
        self._mean = mean
        self._n_components = int(comp.shape[0])
        self._fitted = True
        return self

    # ── Transform ──────────────────────────────────────────────────────
    def transform(self, X: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        """Apply dimensionality reduction to ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
        """
        self._check_fitted()
        X = self._validate(X)
        result: npt.NDArray[np.float32] = _rust.pca_transform(X, self._mean, self._components)
        return result

    # ── Fit + transform ────────────────────────────────────────────────
    def fit_transform(self, X: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        """Fit the model with ``X`` and apply dimensionality reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
        """
        return self.fit(X).transform(X)

    # ── Properties ─────────────────────────────────────────────────────
    @property
    def components_(self) -> npt.NDArray[np.float32]:
        """Principal axes, shape ``(n_components, n_features)``."""
        self._check_fitted()
        assert self._components is not None
        return self._components

    @property
    def explained_variance_(self) -> npt.NDArray[np.float32]:
        """Explained variance per component."""
        self._check_fitted()
        assert self._explained_variance is not None
        return self._explained_variance

    @property
    def explained_variance_ratio_(self) -> npt.NDArray[np.float32]:
        """Fraction of total variance per component."""
        self._check_fitted()
        assert self._explained_variance_ratio is not None
        return self._explained_variance_ratio

    @property
    def singular_values_(self) -> npt.NDArray[np.float32]:
        """Singular values corresponding to each component."""
        self._check_fitted()
        assert self._singular_values is not None
        return self._singular_values

    @property
    def mean_(self) -> npt.NDArray[np.float32]:
        """Per-feature empirical mean."""
        self._check_fitted()
        assert self._mean is not None
        return self._mean

    @property
    def n_components_(self) -> int:
        """Actual number of fitted components."""
        self._check_fitted()
        assert self._n_components is not None
        return self._n_components

    # ── Internal helpers ───────────────────────────────────────────────
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("PCA has not been fitted yet. Call .fit() first.")

    @staticmethod
    def _validate(X: Any) -> npt.NDArray[np.float32]:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {arr.ndim}D.")
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        return arr


# ── Convenience functions ──────────────────────────────────────────────


def pca(data: npt.NDArray[Any], n_components: int = 2) -> npt.NDArray[np.float32]:
    """One-shot PCA: fit and transform in a single call.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
    n_components : int
        Number of principal components.

    Returns
    -------
    ndarray of shape (n_samples, n_components)
    """
    return PCA(n_components=n_components).fit_transform(data)


def pca2(data: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """One-shot PCA to 2 dimensions.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)

    Returns
    -------
    ndarray of shape (n_samples, 2)
    """
    return pca(data, n_components=2)


def pca3(data: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """One-shot PCA to 3 dimensions.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)

    Returns
    -------
    ndarray of shape (n_samples, 3)
    """
    return pca(data, n_components=3)
