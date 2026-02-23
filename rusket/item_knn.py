from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp

from . import _rusket as _rust  # type: ignore
from .model import ImplicitRecommender


def _bm25_weight(X: sp.csr_matrix, K1: float = 1.2, B: float = 0.75) -> sp.csr_matrix:
    """Weighs each item-user interaction by BM25."""
    import numpy as np
    import scipy.sparse as sp

    X_coo = X.tocoo()

    # Calculate item frequencies
    N = float(X_coo.shape[0])  # type: ignore[index]
    item_counts = np.bincount(X_coo.col, minlength=X_coo.shape[1])  # type: ignore[index]
    idf = np.log((N - item_counts + 0.5) / (item_counts + 0.5) + 1.0)

    # Calculate user frequencies
    user_lens = np.bincount(X_coo.row, minlength=X_coo.shape[0])  # type: ignore[index]
    avg_len = user_lens.mean()
    if avg_len == 0:
        avg_len = 1.0

    # Weight
    weight = (X_coo.data * (K1 + 1.0)) / (X_coo.data + K1 * (1.0 - B + B * user_lens[X_coo.row] / avg_len))
    weight = weight * idf[X_coo.col]

    return sp.csr_matrix((weight, (X_coo.row, X_coo.col)), shape=X_coo.shape)


def _tfidf_weight(X: sp.csr_matrix) -> sp.csr_matrix:
    """Weighs each item-user interaction by TF-IDF."""
    import numpy as np
    import scipy.sparse as sp

    X_coo = X.tocoo()

    N = float(X_coo.shape[0])  # type: ignore[index]
    item_counts = np.bincount(X_coo.col, minlength=X_coo.shape[1])  # type: ignore[index]
    # Standard IDF
    idf = np.log(N / (item_counts + 1.0)) + 1.0

    weight = X_coo.data * idf[X_coo.col]

    return sp.csr_matrix((weight, (X_coo.row, X_coo.col)), shape=X_coo.shape)


def _cosine_weight(X: sp.csr_matrix) -> sp.csr_matrix:
    """Normalize rows for cosine similarity."""
    import numpy as np
    import scipy.sparse as sp

    row_norms = np.array(X.multiply(X).sum(axis=1)).flatten()
    row_norms = np.sqrt(row_norms)
    row_norms[row_norms == 0] = 1.0

    X_coo = X.tocoo()
    data = X_coo.data / row_norms[X_coo.row]
    return sp.csr_matrix((data, (X_coo.row, X_coo.col)), shape=X.shape)


class ItemKNN(ImplicitRecommender):
    """
    Ultra-fast Sparse Item-Item K-Nearest Neighbors Recommender.

    Computes an item-item similarity matrix and only retains the top-K neighbors
    per item. Similarity methods include BM25, TF-IDF, Cosine, or unweighted Count.
    """

    def __init__(
        self,
        method: Literal["bm25", "tfidf", "cosine", "count"] = "bm25",
        k: int = 20,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        verbose: int = 0,
        **kwargs: Any,
    ):
        super().__init__()
        self.method = method
        self.k = k
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.verbose = verbose

        self.w_indptr: np.ndarray | None = None
        self.w_indices: np.ndarray | None = None
        self.w_data: np.ndarray | None = None
        self.fitted: bool = False

    def __repr__(self) -> str:
        return f"ItemKNN(method='{self.method}', k={self.k})"

    def fit(self, interactions: Any) -> ItemKNN:
        """Fit the ItemKNN model.

        Parameters
        ----------
        interactions : scipy.sparse.csr_matrix
            A sparse matrix of shape (n_users, n_items).

        Returns
        -------
        ItemKNN
            The fitted model.
        """
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

        # Compute item-item similarity W = X^T * X
        # For Cosine, we should row-normalize before dot product, which X_weighted handles if method="cosine".
        # But wait, cosine is X_normalized^T * X_normalized.
        if self.method == "cosine":
            W = X_weighted.T.dot(X_weighted)
        else:
            # BM25/TF-IDF is usually X_weighted.T * X
            W = X_weighted.T.dot(interactions)

        # Ensure it's CSR
        W = W.tocsr()
        W.eliminate_zeros()

        # Optimize by pruning to Top-K neighbors per item in Rust
        ip, ix, dt = _rust.itemknn_top_k(  # type: ignore[attr-defined]
            W.indptr.astype(np.int64), W.indices.astype(np.int32), W.data.astype(np.float32), self.k
        )

        self.w_indptr = ip
        self.w_indices = ix
        self.w_data = dt
        self._n_users = interactions.shape[0]
        self._n_items = interactions.shape[1]

        # Store fit interactions to omit seen items in recommend_items
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
        """Top-N items for a user.

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
            user_data = np.ones_like(exc_indices, dtype=np.float32)
        else:
            user_data = self._fit_data

        ids, scores = _rust.itemknn_recommend_items(  # type: ignore[attr-defined]
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


