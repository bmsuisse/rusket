"""UserKNN — User-Based K-Nearest Neighbors collaborative filtering recommender."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np

from .. import _rusket as _rust  # type: ignore
from ..model import ImplicitRecommender
from .item_knn import _bm25_weight, _cosine_weight, _tfidf_weight


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
        from .._internal._config import _resolve_cuda

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

        # Compute user-user similarity W = X * X^T (note: ItemKNN uses X^T * X), fused
        # with the top-K prune in Rust so the full n_users x n_users Gram matrix is
        # never materialised in Python/scipy.
        if self.method == "cosine":
            b = X_weighted
        else:
            b = interactions
        if not sp.isspmatrix_csr(b):
            b = b.tocsr()
        b.eliminate_zeros()

        n_users, n_items = interactions.shape
        ip, ix, dt = _rust.userknn_gram_top_k(  # type: ignore[attr-defined]
            X_weighted.indptr.astype(np.int64),
            X_weighted.indices.astype(np.int32),
            X_weighted.data.astype(np.float32),
            b.indptr.astype(np.int64),
            b.indices.astype(np.int32),
            b.data.astype(np.float32),
            n_users,
            n_items,
            self.k,
        )

        # userknn_gram_top_k already returns exact dtypes (int64/int32/float32); no cast needed.
        self.w_indptr = ip
        self.w_indices = ix
        self.w_data = dt
        self._n_users = interactions.shape[0]
        self._n_items = interactions.shape[1]

        # Store fit interactions for recommendations and exclude_seen.
        # Cast ONCE here (not per-call in recommend_items) to the exact dtypes Rust
        # wants, so PyReadonlyArray1 is zero-copy on every recommend_items() call.
        self._fit_indptr = interactions.indptr.astype(np.int64)
        self._fit_indices = interactions.indices.astype(np.int32)
        self._fit_data = interactions.data.astype(np.float32)
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

        fit_indptr = getattr(self, "_fit_indptr", None)
        fit_indices = getattr(self, "_fit_indices", None)
        if fit_indptr is None:
            fit_indptr = np.zeros(self._n_users + 1, dtype=np.int64)
        if fit_indices is None:
            fit_indices = np.array([], dtype=np.int32)

        if exclude_seen and fit_indptr is not None and fit_indices is not None:
            exc_indptr = fit_indptr
            exc_indices = fit_indices
        else:
            exc_indptr = np.zeros(self._n_users + 1, dtype=np.int64)
            exc_indices = np.array([], dtype=np.int32)

        fit_data = getattr(self, "_fit_data", None)
        if fit_data is None:
            user_data = np.ones_like(fit_indices, dtype=np.float32)
        else:
            user_data = fit_data

        # All arrays below were already cast to their exact Rust dtypes once in
        # fit() (or constructed with those dtypes above), so this is zero-copy.
        ids, scores = _rust.userknn_recommend_items(  # type: ignore[attr-defined]
            self.w_indptr,  # type: ignore[union-attr]
            self.w_indices,  # type: ignore[union-attr]
            self.w_data,  # type: ignore[union-attr]
            fit_indptr,
            fit_indices,
            user_data,
            user_id,
            n,
            exc_indptr,
            exc_indices,
            self._n_items,
        )
        return np.asarray(ids), np.asarray(scores)
