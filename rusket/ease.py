"""EASE (Embarrassingly Shallow Autoencoders) collaborative filtering recommender."""

from __future__ import annotations

import typing
from typing import Any

from . import _rusket as _rust  # type: ignore
from .model import ImplicitRecommender


class EASE(ImplicitRecommender):
    """Embarrassingly Shallow Autoencoders for Sparse Data (EASE).

    An implicit collaborative filtering algorithm that computes a closed-form
    item-item similarity matrix by solving a ridge regression problem. EASE
    often achieves state-of-the-art recommendation quality and very fast
    inference, particularly on datasets with strong item-item correlations.

    Parameters
    ----------
    regularization : float
        L2 regularization weight (lambda). Higher values encourage smaller weights
        and reduce overfitting. Default is 500.0.
    """

    def __init__(
        self,
        regularization: float = 500.0,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(data=None, **kwargs)
        self.regularization = float(regularization)
        self.verbose = verbose

        self.item_weights: Any = None
        self._n_users: int = 0
        self._n_items: int = 0
        self._fit_indptr: Any = None
        self._fit_indices: Any = None
        self._fit_data: Any = None

        self._user_labels: list[Any] | None = None
        self._item_labels: list[Any] | None = None
        self.fitted: bool = False

    def __repr__(self) -> str:
        return f"EASE(regularization={self.regularization})"

    def fit(self, interactions: Any) -> EASE:
        """Fit the model to the user-item interaction matrix."""
        import numpy as np
        from scipy import sparse as sp

        if self.fitted:
            raise RuntimeError("Model is already fitted. Create a new instance to refit.")

        if sp.issparse(interactions):
            csr = sp.csr_matrix(interactions, dtype=np.float32)
        elif isinstance(interactions, np.ndarray):
            csr = sp.csr_matrix(interactions.astype(np.float32))
        else:
            raise TypeError(f"Expected scipy sparse matrix or numpy array, got {type(interactions)}")

        if not isinstance(csr, sp.csr_matrix):
            csr = csr.tocsr()

        n_users, n_items = typing.cast(tuple[int, int], csr.shape)

        if self.verbose:
            print(f"EASE fitting Gram matrix for {n_items} items...")

        G = csr.T.dot(csr)
        G_dense = G.toarray()

        diag_indices = np.diag_indices(n_items)
        G_dense[diag_indices] += self.regularization

        if self.verbose:
            print("EASE inverting Gram matrix...")

        P = np.linalg.inv(G_dense)
        B = P / (-np.diag(P))
        B[diag_indices] = 0.0

        self.item_weights = B.astype(np.float32)

        self._n_users = n_users
        self._n_items = n_items
        self._fit_indptr = np.asarray(csr.indptr, dtype=np.int64)
        self._fit_indices = np.asarray(csr.indices, dtype=np.int32)
        self._fit_data = np.asarray(csr.data, dtype=np.float32)
        self.fitted = True

        if self.verbose:
            print("EASE fit complete.")

        return self

    def recommend_items(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> tuple[Any, Any]:
        """Top-N items for a user. Set exclude_seen=False to include already-seen items."""
        import numpy as np

        self._check_fitted()
        if user_id < 0 or user_id >= self._n_users:
            raise ValueError(f"user_id {user_id} is out of bounds for model with {self._n_users} users.")

        if exclude_seen and self._fit_indptr is not None and self._fit_indices is not None:
            exc_indptr = self._fit_indptr
            exc_indices = self._fit_indices
        else:
            exc_indptr = np.zeros(self._n_users + 1, dtype=np.int64)
            exc_indices = np.array([], dtype=np.int32)

        ids, scores = _rust.ease_recommend_items(
            self.item_weights,
            self._fit_indptr,
            self._fit_indices,
            self._fit_data,
            user_id,
            n,
            exc_indptr,
            exc_indices,
        )
        return np.asarray(ids), np.asarray(scores)

    def recommend_users(self, item_id: int, n: int = 10) -> tuple[Any, Any]:
        """Top-N users for an item. Not implemented for EASE currently."""
        raise NotImplementedError("EASE does not efficiently support recommending users for an item.")

    def _check_fitted(self) -> None:
        if self.item_weights is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
