"""Popularity-based baseline recommender."""

from __future__ import annotations

import typing
from typing import Any

import numpy as np

from .model import ImplicitRecommender


class PopularityRecommender(ImplicitRecommender):
    """Recommend items by global popularity (interaction count).

    A non-personalised baseline that ranks every item by the total number
    of interactions it received.  Useful as a sanity-check baseline when
    evaluating more sophisticated models.

    Parameters
    ----------
    verbose : int, default=0
        Verbosity level.
    """

    def __init__(self, verbose: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.verbose = verbose
        self._item_scores: np.ndarray | None = None
        self._n_users: int = 0
        self._n_items: int = 0
        self._fit_indptr: np.ndarray | None = None
        self._fit_indices: np.ndarray | None = None
        self.fitted: bool = False

    def __repr__(self) -> str:
        return "PopularityRecommender()"

    # ── fit ────────────────────────────────────────────────────────────

    def fit(self, interactions: Any = None) -> PopularityRecommender:
        """Fit the model by counting interactions per item.

        Parameters
        ----------
        interactions : sparse matrix or numpy array, optional
            User-item interaction matrix.  If *None*, uses the matrix
            prepared by ``from_transactions()``.

        Returns
        -------
        PopularityRecommender
            The fitted model.
        """
        if interactions is None:
            interactions = getattr(self, "_prepared_interactions", None)
            if interactions is None:
                raise ValueError("No interactions provided. Pass a matrix or use from_transactions() first.")

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

        # Global popularity = sum of interactions per item (column-wise)
        self._item_scores = np.asarray(csr.sum(axis=0), dtype=np.float64).ravel()

        self._n_users = n_users
        self._n_items = n_items
        self._fit_indptr = np.asarray(csr.indptr, dtype=np.int64)
        self._fit_indices = np.asarray(csr.indices, dtype=np.int32)
        self.fitted = True

        if self.verbose:
            print(f"PopularityRecommender fitted: {n_items} items, top score={self._item_scores.max():.0f}")

        return self

    # ── recommend ──────────────────────────────────────────────────────

    def recommend_items(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the *n* most popular items for a user.

        Parameters
        ----------
        user_id : int
            Internal user index.
        n : int, default=10
            Number of items to return.
        exclude_seen : bool, default=True
            Whether to exclude items the user has already interacted with.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(item_ids, scores)`` sorted by descending popularity.
        """
        self._check_fitted()
        assert self._item_scores is not None  # for type narrowing

        if user_id < 0 or user_id >= self._n_users:
            raise ValueError(f"user_id {user_id} is out of bounds for model with {self._n_users} users.")

        scores = self._item_scores.copy()

        if exclude_seen and self._fit_indptr is not None and self._fit_indices is not None:
            start = self._fit_indptr[user_id]
            end = self._fit_indptr[user_id + 1]
            seen = self._fit_indices[start:end]
            scores[seen] = -np.inf

        top_n = np.argsort(scores)[::-1][:n]
        return top_n.astype(np.intp), scores[top_n].astype(np.float32)

    # ── helpers ────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    @property
    def item_popularity(self) -> np.ndarray:
        """Raw item popularity scores (interaction counts)."""
        self._check_fitted()
        assert self._item_scores is not None
        return self._item_scores
