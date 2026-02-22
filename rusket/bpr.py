"""Bayesian Personalized Ranking (BPR) implicit recommender."""

from __future__ import annotations
import typing
from typing import Any

from . import _rusket as _rust  # type: ignore

from .model import ImplicitRecommender


class BPR(ImplicitRecommender):
    """Bayesian Personalized Ranking (BPR) model for implicit feedback.

    BPR optimizes for ranking rather than reconstruction error (like ALS).
    It works by drawing positive items the user interacted with, and negative items
    they haven't, and adjusting latent factors to ensure the positive item scores higher.

    Parameters
    ----------
    factors : int
        Number of latent factors (default: 64).
    learning_rate : float
        SGD learning rate (default: 0.05).
    regularization : float
        L2 regularization weight (default: 0.01).
    iterations : int
        Number of passes over the entire interaction dataset (default: 150).
    seed : int
        Random seed for Hogwild! SGD sampling (default: 42).
    """

    def __init__(
        self,
        factors: int = 64,
        learning_rate: float = 0.05,
        regularization: float = 0.01,
        iterations: int = 150,
        seed: int = 42,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(data=None, **kwargs)
        self.factors = factors
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.iterations = iterations
        self.seed = seed
        self.verbose = verbose
        self._user_factors: Any = None
        self._item_factors: Any = None
        self._n_users: int = 0
        self._n_items: int = 0
        self._fit_indptr: Any = None
        self._fit_indices: Any = None
        self._user_labels: list[Any] | None = None
        self._item_labels: list[Any] | None = None
        self.fitted: bool = False

    def __repr__(self) -> str:
        return (
            f"BPR(factors={self.factors}, learning_rate={self.learning_rate}, "
            f"regularization={self.regularization}, iterations={self.iterations})"
        )

    def fit(self, interactions: Any) -> "BPR":
        """Fit the BPR model to the user-item interaction matrix."""
        import numpy as np
        from scipy import sparse as sp

        if self.fitted:
            raise RuntimeError("Model is already fitted.")

        if sp.issparse(interactions):
            csr = sp.csr_matrix(interactions, dtype=np.float32)
        elif isinstance(interactions, np.ndarray):
            csr = sp.csr_matrix(interactions.astype(np.float32))
        else:
            raise TypeError(
                f"Expected scipy sparse matrix or numpy array, got {type(interactions)}"
            )

        if not isinstance(csr, sp.csr_matrix):
            csr = csr.tocsr()

        csr.eliminate_zeros()

        n_users, n_items = typing.cast(tuple[int, int], csr.shape)
        indptr = np.asarray(csr.indptr, dtype=np.int64)
        indices = np.asarray(csr.indices, dtype=np.int32)

        self._user_factors, self._item_factors = _rust.bpr_fit_implicit(
            indptr,
            indices,
            n_users,
            n_items,
            self.factors,
            self.learning_rate,
            self.regularization,
            self.iterations,
            self.seed,
            self.verbose,
        )
        self._n_users = n_users
        self._n_items = n_items
        self._fit_indptr = indptr
        self._fit_indices = indices
        self.fitted = True
        return self



    def recommend_items(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> tuple[Any, Any]:
        """Top-N items for a user."""
        import numpy as np

        self._check_fitted()
        if user_id < 0 or user_id >= self._n_users:
            raise ValueError(f"user_id {user_id} is out of bounds for model with {self._n_users} users.")
        if (
            exclude_seen
            and self._fit_indptr is not None
            and self._fit_indices is not None
        ):
            exc_indptr = self._fit_indptr
            exc_indices = self._fit_indices
        else:
            exc_indptr = np.zeros(self._n_users + 1, dtype=np.int64)
            exc_indices = np.array([], dtype=np.int32)

        ids, scores = _rust.als_recommend_items(
            self._user_factors,
            self._item_factors,
            user_id,
            n,
            exc_indptr,
            exc_indices,
        )
        return np.asarray(ids), np.asarray(scores)

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    @property
    def user_factors(self) -> Any:
        self._check_fitted()
        return self._user_factors

    @property
    def item_factors(self) -> Any:
        self._check_fitted()
        return self._item_factors
