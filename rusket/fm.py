"""Factorization Machines (FM) for Context-aware Recommendation."""

from __future__ import annotations

import typing
from typing import Any

from . import _rusket as _rust  # type: ignore
from .model import BaseModel


class FM(BaseModel):
    """Factorization Machines (FM) context-aware model for predictive tasks (e.g. CTR).

    This model supports binary classification tasks using Log Loss (Binary Cross Entropy).
    Inputs should be formatted as a scipy sparse CSR matrix where features are binary (0/1).
    Each row is a sample consisting of User, Item, and Context features.

    Parameters
    ----------
    factors : int
        Number of latent factors for the cross terms (default: 8).
    learning_rate : float
        SGD learning rate (default: 0.05).
    regularization : float
        L2 regularization weight (default: 0.01).
    iterations : int
        Number of training epochs (default: 100).
    seed : int
        Random seed for SGD sampling (default: 42).
    verbose : bool
        Whether to print training progress (default: False).
    """

    def __init__(
        self,
        factors: int = 8,
        learning_rate: float = 0.05,
        regularization: float = 0.01,
        iterations: int = 100,
        seed: int = 42,
        verbose: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.factors = factors
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.iterations = iterations
        self.seed = seed
        self.verbose = verbose

        self.w0_: float | None = None
        self.w_: Any = None
        self.v_: Any = None

        self._n_features: int = 0
        self.fitted: bool = False

    def __repr__(self) -> str:
        return (
            f"FM(factors={self.factors}, learning_rate={self.learning_rate}, "
            f"regularization={self.regularization}, iterations={self.iterations})"
        )

    @classmethod
    def from_transactions(cls, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("FM requires an explicit feature matrix. Use .fit(X, y) instead.")

    def fit(self, X: Any, y: Any) -> FM:
        """Fit the FM model to Context-aware Data.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy array
            Sparse binary feature matrix of shape (n_samples, n_features).
            Each row represents a single interaction with all its context features.
        y : numpy.ndarray
            Binary target labels (0.0 or 1.0) of shape (n_samples,).
        """
        import numpy as np
        from scipy import sparse as sp

        if self.fitted:
            raise RuntimeError("Model is already fitted.")

        if sp.issparse(X):
            csr = sp.csr_matrix(X, dtype=np.float32)
        elif isinstance(X, np.ndarray):
            csr = sp.csr_matrix(X.astype(np.float32))
        else:
            raise TypeError(f"Expected scipy sparse matrix or numpy array for X, got {type(X)}")

        csr.eliminate_zeros()

        y_arr = np.asarray(y, dtype=np.float32)
        n_samples, n_features = typing.cast(tuple[int, int], csr.shape)

        if len(y_arr) != n_samples:
            raise ValueError(f"X has {n_samples} samples but y has {len(y_arr)} labels.")

        indptr = np.asarray(csr.indptr, dtype=np.int64)
        indices = np.asarray(csr.indices, dtype=np.int32)

        self.w0_, self.w_, self.v_ = _rust.fm_fit(  # type: ignore[attr-defined]
            indptr,
            indices,
            y_arr,
            n_samples,
            n_features,
            self.factors,
            self.learning_rate,
            self.regularization,
            self.iterations,
            self.seed,
            bool(self.verbose),
        )

        self._n_features = n_features
        self.fitted = True
        return self

    def predict_proba(self, X: Any) -> Any:
        """Predict the probability (CTR) of interactions.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy array
            Sparse binary feature matrix of shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Predicted probabilities of shape (n_samples,).
        """
        import numpy as np
        from scipy import sparse as sp

        self._check_fitted()
        assert self.w0_ is not None
        assert self.w_ is not None
        assert self.v_ is not None

        if sp.issparse(X):
            csr = sp.csr_matrix(X, dtype=np.float32)
        else:
            csr = sp.csr_matrix(X.astype(np.float32))

        csr.eliminate_zeros()
        n_samples, n_features = typing.cast(tuple[int, int], csr.shape)

        if n_features != self._n_features:
            raise ValueError(f"X has {n_features} features, but model expects {self._n_features}.")

        indptr = np.asarray(csr.indptr, dtype=np.int64)
        indices = np.asarray(csr.indices, dtype=np.int32)

        return _rust.fm_predict(  # type: ignore[attr-defined]
            indptr,
            indices,
            self.w0_,
            self.w_,
            self.v_,
            self.factors,
            n_samples,
        )

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
