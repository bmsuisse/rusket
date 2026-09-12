from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp

from .. import _rusket as _rust  # type: ignore
from ..model import ImplicitRecommender


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
        return f"ItemKNN(method='{self.method}', k={self.k})"

    def fit(self, interactions: Any = None) -> ItemKNN:
        """Fit the ItemKNN model.

        Parameters
        ----------
        interactions : scipy.sparse.csr_matrix, optional
            A sparse matrix of shape (n_users, n_items).
            If None, uses the matrix prepared by ``from_transactions()``.

        Returns
        -------
        ItemKNN
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

        # Compute item-item similarity W = X^T * X (or X^T * interactions), fused with
        # the top-K prune in Rust so the full n_items x n_items Gram matrix (which can
        # be tens of GB dense-equivalent) is never materialised in Python/scipy.
        # For Cosine, both operands are the row-normalized X_weighted; for the other
        # methods, the right-hand operand is the raw (unweighted) interactions, same
        # asymmetry as the old scipy path.
        if self.method == "cosine":
            b = X_weighted
        else:
            b = interactions
        if not sp.isspmatrix_csr(b):
            b = b.tocsr()
        b.eliminate_zeros()

        n_users, n_items = interactions.shape
        ip, ix, dt = _rust.itemknn_gram_top_k(  # type: ignore[attr-defined]
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

        # itemknn_gram_top_k already returns exact dtypes (int64/int32/float32); no cast needed.
        self.w_indptr = ip
        self.w_indices = ix
        self.w_data = dt
        self._n_users = interactions.shape[0]
        self._n_items = interactions.shape[1]

        # Store fit interactions to omit seen items in recommend_items.
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

        if self.use_cuda:
            # KNN similarity matrices are sparse × sparse — CUDA acceleration
            # for sparse×sparse is not beneficial. Fall through to Rust path.
            pass

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
        ids, scores = _rust.itemknn_recommend_items(  # type: ignore[attr-defined]
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
