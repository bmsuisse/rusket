"""ALS (Alternating Least Squares) collaborative filtering recommender."""
from __future__ import annotations
import typing
from typing import Any
from . import _rusket as _rust  # type: ignore


class ALS:
    """Implicit ALS collaborative filtering model.

    Parameters
    ----------
    factors : int
        Number of latent factors.
    regularization : float
        L2 regularisation weight.
    alpha : float
        Confidence scaling: ``confidence = 1 + alpha * r``.
    iterations : int
        Number of ALS iterations.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        alpha: float = 40.0,
        iterations: int = 15,
        seed: int = 42,
    ) -> None:
        self.factors = factors
        self.regularization = float(regularization)
        self.alpha = float(alpha)
        self.iterations = iterations
        self.seed = seed
        self._user_factors: Any = None
        self._item_factors: Any = None
        self._n_users: int = 0
        self._n_items: int = 0
        self._fit_indptr: Any = None
        self._fit_indices: Any = None
        self._user_labels: list[Any] | None = None
        self._item_labels: list[Any] | None = None

    def __repr__(self) -> str:
        return (
            f"ALS(factors={self.factors}, regularization={self.regularization}, "
            f"alpha={self.alpha}, iterations={self.iterations})"
        )

    def fit(self, user_item_matrix: Any) -> "ALS":
        """Fit on a user-item interaction matrix (scipy sparse or numpy array)."""
        import numpy as np
        from scipy import sparse as sp

        if sp.issparse(user_item_matrix):
            csr = sp.csr_matrix(user_item_matrix, dtype=np.float32)
        elif isinstance(user_item_matrix, np.ndarray):
            csr = sp.csr_matrix(user_item_matrix.astype(np.float32))
        else:
            raise TypeError(
                f"Expected scipy sparse matrix or numpy array, got {type(user_item_matrix)}"
            )

        csr.eliminate_zeros()
        n_users, n_items = typing.cast(tuple[int, int], csr.shape)
        indptr = np.asarray(csr.indptr, dtype=np.int64)
        indices = np.asarray(csr.indices, dtype=np.int32)
        data = np.asarray(csr.data, dtype=np.float32)

        uf, itf = _rust.als_fit_implicit(
            indptr, indices, data, n_users, n_items,
            self.factors, float(self.regularization),
            float(self.alpha), self.iterations, self.seed,
        )
        self._user_factors = np.asarray(uf)
        self._item_factors = np.asarray(itf)
        self._n_users = n_users
        self._n_items = n_items
        self._fit_indptr = indptr
        self._fit_indices = indices
        return self

    def fit_transactions(
        self,
        data: Any,
        user_col: str | None = None,
        item_col: str | None = None,
        rating_col: str | None = None,
    ) -> "ALS":
        """Fit from a long-format Pandas/Polars/Spark DataFrame."""
        import numpy as np
        import pandas as _pd
        from scipy import sparse as sp

        t = type(data).__name__
        if t == "DataFrame" and getattr(data, "__module__", "").startswith("pyspark"):
            data = typing.cast(Any, data).toPandas()
        elif t == "DataFrame" and getattr(data, "__module__", "").startswith("polars"):
            data = typing.cast(Any, data).to_pandas()

        if not isinstance(data, _pd.DataFrame):
            raise TypeError(f"Expected Pandas/Polars/Spark DataFrame, got {type(data)}")

        cols = list(data.columns)
        u_col = user_col or str(cols[0])
        i_col = item_col or str(cols[1])

        user_codes, user_uniques = _pd.factorize(data[u_col], sort=False)
        item_codes, item_uniques = _pd.factorize(data[i_col], sort=True)
        n_users = len(user_uniques)
        n_items = len(item_uniques)

        values = (
            np.asarray(data[rating_col], dtype=np.float32)
            if rating_col is not None
            else np.ones(len(user_codes), dtype=np.float32)
        )
        csr = sp.csr_matrix(
            (values, (user_codes.astype(np.int64), item_codes.astype(np.int64))),
            shape=(n_users, n_items),
        )
        self._user_labels = list(user_uniques)
        self._item_labels = [str(c) for c in item_uniques]
        return self.fit(csr)

    def recommend_items(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> tuple[Any, Any]:
        """Top-N items for a user. Set exclude_seen=False to include already-seen items."""
        import numpy as np
        self._check_fitted()
        if exclude_seen and self._fit_indptr is not None and self._fit_indices is not None:
            exc_indptr = self._fit_indptr
            exc_indices = self._fit_indices
        else:
            exc_indptr = np.zeros(self._n_users + 1, dtype=np.int64)
            exc_indices = np.array([], dtype=np.int32)
        ids, scores = _rust.als_recommend_items(
            self._user_factors, self._item_factors,
            user_id, n, exc_indptr, exc_indices,
        )
        return np.asarray(ids), np.asarray(scores)

    def recommend_users(self, item_id: int, n: int = 10) -> tuple[Any, Any]:
        """Top-N users for an item."""
        import numpy as np
        self._check_fitted()
        ids, scores = _rust.als_recommend_users(
            self._user_factors, self._item_factors, item_id, n,
        )
        return np.asarray(ids), np.asarray(scores)

    @property
    def user_factors(self) -> Any:
        """User factor matrix (n_users, factors)."""
        self._check_fitted()
        return self._user_factors

    @property
    def item_factors(self) -> Any:
        """Item factor matrix (n_items, factors)."""
        self._check_fitted()
        return self._item_factors

    def _check_fitted(self) -> None:
        if self._user_factors is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
