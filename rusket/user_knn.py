"""UserKNN — User-Based K-Nearest Neighbors collaborative filtering recommender."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np

from . import _rusket as _rust  # type: ignore
from .item_knn import _bm25_weight, _cosine_weight, _tfidf_weight
from .model import ImplicitRecommender


class UserKNN(ImplicitRecommender):
    """User-Based K-Nearest Neighbors Recommender.

    Computes a user-user similarity matrix and recommends items that similar
    users have interacted with. Similarity methods include BM25, TF-IDF,
    Cosine, or unweighted Count.

    Parameters
    ----------
    method : {'bm25', 'tfidf', 'cosine', 'count'}, default='cosine'
        Weighting scheme applied to the interaction matrix before computing
        user-user similarity.
    k : int, default=20
        Number of nearest neighbors to retain per user.
    bm25_k1 : float, default=1.2
        BM25 term-frequency saturation parameter (only used when method='bm25').
    bm25_b : float, default=0.75
        BM25 length-normalization parameter (only used when method='bm25').
    verbose : int, default=0
        Verbosity level.
    """

    def __init__(
        self,
        method: Literal["bm25", "tfidf", "cosine", "count"] = "cosine",
        k: int = 20,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        verbose: int = 0,
        use_cuda: bool | None = None,
        **kwargs: Any,
    ):
        _use_cuda = kwargs.pop("use_gpu", use_cuda)  # backward compat
        super().__init__()
        self.method = method
        self.k = k
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.verbose = verbose
        from ._config import _resolve_cuda

        self.use_cuda = _resolve_cuda(_use_cuda)

        self.w_indptr: np.ndarray | None = None
        self.w_indices: np.ndarray | None = None
        self.w_data: np.ndarray | None = None
        self.fitted: bool = False

    def __repr__(self) -> str:
        return f"UserKNN(method='{self.method}', k={self.k})"

    def fit(self, interactions: Any = None) -> UserKNN:
        """Fit the UserKNN model.

        Parameters
        ----------
        interactions : scipy.sparse.csr_matrix, optional
            A sparse matrix of shape (n_users, n_items).
            If None, uses the matrix prepared by ``from_transactions()``.

        Returns
        -------
        UserKNN
            The fitted model.
        """
        if interactions is None:
            interactions = getattr(self, "_prepared_interactions", None)
            if interactions is None:
                raise ValueError("No interactions provided. Pass a matrix or use from_transactions() first.")
        import numpy as np
        import scipy.sparse as sp

        if not sp.isspmatrix_csr(interactions):
            interactions = interactions.tocsr()

        interactions.eliminate_zeros()

        # Apply weighting
        if self.method == "bm25":
            X_weighted = _bm25_weight(interactions, K1=self.bm25_k1, B=self.bm25_b)
        elif self.method == "tfidf":
            X_weighted = _tfidf_weight(interactions)
        elif self.method == "cosine":
            X_weighted = _cosine_weight(interactions)
        elif self.method == "count":
            X_weighted = interactions
        else:
            raise ValueError(f"Unknown method {self.method}")

        # Compute user-user similarity W = X * X^T  (note: ItemKNN uses X^T * X)
        if self.method == "cosine":
            W = X_weighted.dot(X_weighted.T)
        else:
            W = X_weighted.dot(interactions.T)

        W = W.tocsr()
        W.eliminate_zeros()

        # Prune to Top-K neighbors per user in Rust
        ip, ix, dt = _rust.userknn_top_k(  # type: ignore[attr-defined]
            W.indptr.astype(np.int64), W.indices.astype(np.int32), W.data.astype(np.float32), self.k
        )

        self.w_indptr = ip
        self.w_indices = ix
        self.w_data = dt
        self._n_users = interactions.shape[0]
        self._n_items = interactions.shape[1]

        # Store fit interactions for recommendations and exclude_seen
        self._fit_indptr = interactions.indptr
        self._fit_indices = interactions.indices
        self._fit_data = interactions.data
        self.fitted = True

        return self

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")

    def recommend_items(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> tuple[Any, Any]:
        """Top-N items for a user based on similar users.

        Parameters
        ----------
        user_id : int
            The user ID to generate recommendations for.
        n : int, default=10
            Number of items to return.
        exclude_seen : bool, default=True
            Whether to exclude items the user has already interacted with.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(item_ids, scores)`` sorted by descending score.
        """
        self._check_fitted()

        import numpy as np

        if user_id < 0 or user_id >= self._n_users:
            raise ValueError(f"user_id {user_id} is out of bounds for model with {self._n_users} users.")

        if (
            exclude_seen
            and getattr(self, "_fit_indptr", None) is not None
            and getattr(self, "_fit_indices", None) is not None
        ):
            exc_indptr = self._fit_indptr
            exc_indices = self._fit_indices
        else:
            exc_indptr = np.zeros(self._n_users + 1, dtype=np.int64)
            exc_indices = np.array([], dtype=np.int32)

        if getattr(self, "_fit_data", None) is None:
            user_data = np.ones_like(self._fit_indices, dtype=np.float32)
        else:
            user_data = self._fit_data

        ids, scores = _rust.userknn_recommend_items(  # type: ignore[attr-defined]
            self.w_indptr.astype(np.int64),  # type: ignore[union-attr]
            self.w_indices.astype(np.int32),  # type: ignore[union-attr]
            self.w_data.astype(np.float32),  # type: ignore[union-attr]
            getattr(self, "_fit_indptr", np.zeros(self._n_users + 1, dtype=np.int64)).astype(np.int64),
            getattr(self, "_fit_indices", np.array([], dtype=np.int32)).astype(np.int32),
            user_data.astype(np.float32),
            user_id,
            n,
            exc_indptr.astype(np.int64),
            exc_indices.astype(np.int32),
            self._n_items,
        )
        return np.asarray(ids), np.asarray(scores)
