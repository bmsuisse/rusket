"""ALS (Alternating Least Squares) collaborative filtering recommender."""

from __future__ import annotations

import typing
from typing import Any

from . import _rusket as _rust  # type: ignore
from .model import ImplicitRecommender


class ALS(ImplicitRecommender):
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
        Number of ALS outer iterations.
    seed : int
        Random seed.
    cg_iters : int
        Conjugate Gradient iterations per user/item solve (ignored when
        ``use_cholesky=True``).  Reduce to 3 for very large datasets.
    use_cholesky : bool
        Use a direct Cholesky solve instead of iterative CG. Exact solution;
        faster when users have many interactions relative to ``factors``.
    anderson_m : int
        History window for **Anderson Acceleration** of the outer ALS loop
        (default 0 = disabled).  Recommended value: **5**.

        ALS is a fixed-point iteration ``(U,V) → F(U,V)``.  Anderson mixing
        extrapolates over the last ``m`` residuals to reach the fixed point
        faster, typically reducing the number of outer iterations by 30–50 %
        at identical recommendation quality::

            # Baseline: 15 iterations
            model = ALS(iterations=15, cg_iters=3)

            # Anderson-accelerated: 10 iterations, ~2.5× faster, same quality
            model = ALS(iterations=10, cg_iters=3, anderson_m=5)

        Memory overhead: ``m`` copies of the full ``(U ∥ V)`` matrix
        (~57 MB per copy at 25M ratings, k=64).
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        alpha: float = 40.0,
        iterations: int = 15,
        seed: int = 42,
        verbose: int = 0,
        cg_iters: int = 10,
        use_cholesky: bool = False,
        anderson_m: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(data=None, **kwargs)
        self.factors = factors
        self.regularization = float(regularization)
        self.alpha = float(alpha)
        self.iterations = iterations
        self.seed = seed
        self.verbose = verbose
        self.cg_iters = cg_iters
        self.use_cholesky = use_cholesky
        self.anderson_m = anderson_m
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
            f"ALS(factors={self.factors}, regularization={self.regularization}, "
            f"alpha={self.alpha}, iterations={self.iterations})"
        )

    def fit(self, interactions: Any) -> ALS:
        """Fit the model to the user-item interaction matrix.

        Raises
        ------
        RuntimeError
            If the model is already fitted. Create a new instance to refit.
        TypeError
            If the input matrix is not a recognizable sparse matrix or numpy array.
        """
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

        # SciPy's C++ algorithms overflow int32 for nnz > 1B, so skip canonical
        # format enforcement on very large matrices.
        if csr.nnz < 1_000_000_000:
            if not csr.has_canonical_format:
                csr.sum_duplicates()
            csr.eliminate_zeros()

        n_users, n_items = typing.cast(tuple[int, int], csr.shape)
        indptr = np.asarray(csr.indptr, dtype=np.int64)
        indices = np.asarray(csr.indices, dtype=np.int32)
        data = np.asarray(csr.data, dtype=np.float32)

        self._user_factors, self._item_factors = _rust.als_fit_implicit(
            indptr,
            indices,
            data,
            n_users,
            n_items,
            self.factors,
            self.regularization,
            self.alpha,
            self.iterations,
            self.seed,
            bool(self.verbose),
            self.cg_iters,
            self.use_cholesky,
            self.anderson_m,
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
        ids, scores = _rust.als_recommend_items(
            self._user_factors,
            self._item_factors,
            user_id,
            n,
            exc_indptr,
            exc_indices,
        )
        return np.asarray(ids), np.asarray(scores)

    def batch_recommend(
        self,
        n: int = 10,
        exclude_seen: bool = True,
        format: str = "polars",
    ) -> Any:
        """Top-N items for all users efficiently computed in parallel.

        Parameters
        ----------
        n : int, default=10
            The number of items to recommend per user.
        exclude_seen : bool, default=True
            Whether to exclude items the user has already interacted with.
        format : str, default="polars"
            The DataFrame format to return. One of "pandas", "polars", or "spark".

        Returns
        -------
        DataFrame
            A DataFrame with columns `user_id`, `item_id`, and `score`.
        """
        import numpy as np

        self._check_fitted()
        if exclude_seen and self._fit_indptr is not None and self._fit_indices is not None:
            exc_indptr = self._fit_indptr
            exc_indices = self._fit_indices
        else:
            exc_indptr = np.zeros(self._n_users + 1, dtype=np.int64)
            exc_indices = np.array([], dtype=np.int32)

        u_ids, i_ids, scores = _rust.als_recommend_all(
            self._user_factors,
            self._item_factors,
            n,
            exc_indptr,
            exc_indices,
        )

        u_ids_arr = np.asarray(u_ids)
        i_ids_arr = np.asarray(i_ids)
        scores_arr = np.asarray(scores)

        import polars as pl

        df = pl.DataFrame({"user_id": u_ids_arr, "item_id": i_ids_arr, "score": scores_arr})
        if self._user_labels is not None and len(self._user_labels) == self._n_users:
            # Use categorical / explicit mapping
            mapping = dict(enumerate(self._user_labels))
            df = df.with_columns(pl.col("user_id").replace_strict(mapping, default=pl.col("user_id")).alias("user_id"))

        if self._item_labels is not None and len(self._item_labels) == self._n_items:
            mapping = dict(enumerate(self._item_labels))
            df = df.with_columns(pl.col("item_id").replace_strict(mapping, default=pl.col("item_id")).alias("item_id"))

        if format == "polars":
            return df
        elif format == "pandas":
            return df.to_pandas()
        elif format == "spark":
            from pyspark.sql import SparkSession

            spark = SparkSession.getActiveSession()
            if spark is None:
                raise RuntimeError("No active SparkSession found. Initialize Spark first.")
            return spark.createDataFrame(df.to_pandas())
        else:
            raise ValueError(f"Unknown format: {format}")

    def recommend_users(self, item_id: int, n: int = 10) -> tuple[Any, Any]:
        """Top-N users for an item."""
        import numpy as np

        self._check_fitted()
        if item_id < 0 or item_id >= self._n_items:
            raise ValueError(f"item_id {item_id} is out of bounds for model with {self._n_items} items.")
        ids, scores = _rust.als_recommend_users(
            self._user_factors,
            self._item_factors,
            item_id,
            n,
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
