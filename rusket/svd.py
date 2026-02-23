"""SVD (Funk SVD / Biased SGD Matrix Factorization) recommender."""

from __future__ import annotations

import typing
from typing import Any

from . import _rusket as _rust  # type: ignore
from .model import ImplicitRecommender


class SVD(ImplicitRecommender):
    """Funk SVD collaborative filtering model.

    Biased matrix factorization trained with SGD:
        r̂_ui = μ + b_u + b_i + p_u · q_i

    Parameters
    ----------
    factors : int
        Number of latent factors.
    learning_rate : float
        SGD learning rate.
    regularization : float
        L2 regularisation weight.
    iterations : int
        Number of SGD epochs.
    seed : int
        Random seed for reproducibility.
    verbose : int
        Verbosity level (0 = silent, 1+ = progress).
    """

    def __init__(
        self,
        factors: int = 64,
        learning_rate: float = 0.005,
        regularization: float = 0.02,
        iterations: int = 20,
        seed: int = 42,
        verbose: int = 0,
        **kwargs: Any,
    ) -> None:
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.seed = seed
        self.verbose = verbose
        self._fitted = False
        self._user_factors = None
        self._item_factors = None
        self._user_biases = None
        self._item_biases = None
        self._global_mean: float = 0.0
        self._interactions = None

    def __repr__(self) -> str:
        return (
            f"SVD(factors={self.factors}, lr={self.learning_rate}, "
            f"reg={self.regularization}, iterations={self.iterations})"
        )

    def fit(self, interactions: Any) -> SVD:
        """Fit the model to the user-item interaction matrix.

        Parameters
        ----------
        interactions : scipy.sparse matrix, np.ndarray, pd.DataFrame, or polars DataFrame
            User-item interaction matrix with explicit ratings.

        Returns
        -------
        self
        """
        if self._fitted:
            raise RuntimeError("SVD model is already fitted. Create a new instance to refit.")

        import numpy as np

        try:
            import scipy.sparse as sp
        except ImportError:
            sp = None  # type: ignore

        csr = None

        if sp is not None and sp.issparse(interactions):
            csr = sp.csr_matrix(interactions)
        elif isinstance(interactions, np.ndarray):
            if sp is None:
                raise ImportError("scipy is required to convert dense arrays.")
            csr = sp.csr_matrix(interactions)
        else:
            # Try pandas or polars DataFrame
            try:
                import pandas as pd
            except ImportError:
                pd = None  # type: ignore
            try:
                import polars as pl
            except ImportError:
                pl = None  # type: ignore

            if pd is not None and isinstance(interactions, pd.DataFrame):
                if sp is None:
                    raise ImportError("scipy is required to convert DataFrames.")
                csr = sp.csr_matrix(
                    (
                        interactions.iloc[:, 2].values.astype(np.float32),
                        (
                            interactions.iloc[:, 0].values.astype(np.int32),
                            interactions.iloc[:, 1].values.astype(np.int32),
                        ),
                    )
                )
            elif pl is not None and isinstance(interactions, pl.DataFrame):
                if sp is None:
                    raise ImportError("scipy is required to convert DataFrames.")
                cols = interactions.columns
                csr = sp.csr_matrix(
                    (
                        interactions[cols[2]].to_numpy().astype(np.float32),
                        (
                            interactions[cols[0]].to_numpy().astype(np.int32),
                            interactions[cols[1]].to_numpy().astype(np.int32),
                        ),
                    )
                )

            if csr is None:
                raise TypeError(f"Cannot convert {type(interactions).__name__} to CSR matrix.")

        csr = csr.astype(np.float32)
        n_users, n_items = csr.shape

        indptr = np.asarray(csr.indptr, dtype=np.int64)
        indices = np.asarray(csr.indices, dtype=np.int32)
        data = np.asarray(csr.data, dtype=np.float32)

        uf, itf, ub, ib, gm = _rust.svd_fit(  # type: ignore
            indptr,
            indices,
            data,
            n_users,
            n_items,
            self.factors,
            self.learning_rate,
            self.regularization,
            self.iterations,
            self.seed,
            self.verbose > 0,
        )

        self._user_factors = uf
        self._item_factors = itf
        self._user_biases = ub
        self._item_biases = ib
        self._global_mean = gm
        self._interactions = csr
        self._fitted = True
        return self

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
            User index.
        n : int
            Number of items to recommend.
        exclude_seen : bool
            Whether to filter already-seen items.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (item_ids, scores)
        """
        self._check_fitted()
        import numpy as np

        if exclude_seen and self._interactions is not None:
            indptr = np.asarray(self._interactions.indptr, dtype=np.int64)
            indices = np.asarray(self._interactions.indices, dtype=np.int32)
        else:
            n_users = self._user_factors.shape[0]  # type: ignore
            indptr = np.zeros(n_users + 1, dtype=np.int64)
            indices = np.array([], dtype=np.int32)

        return _rust.svd_recommend_items(  # type: ignore
            self._user_factors.ravel().astype(np.float32),  # type: ignore
            self._item_factors.ravel().astype(np.float32),  # type: ignore
            self._user_biases.astype(np.float32),  # type: ignore
            self._item_biases.astype(np.float32),  # type: ignore
            self._global_mean,
            user_id,
            n,
            indptr,
            indices,
        )

    def predict(self, user_id: int, item_id: int) -> float:
        """Predict the rating for a user-item pair.

        Parameters
        ----------
        user_id : int
            User index.
        item_id : int
            Item index.

        Returns
        -------
        float
            Predicted rating.
        """
        self._check_fitted()
        import numpy as np

        pu = self._user_factors[user_id]  # type: ignore
        qi = self._item_factors[item_id]  # type: ignore
        bu = self._user_biases[user_id]  # type: ignore
        bi = self._item_biases[item_id]  # type: ignore
        return float(self._global_mean + bu + bi + np.dot(pu, qi))

    def batch_recommend(
        self,
        n: int = 10,
        exclude_seen: bool = True,
        format: str = "polars",
    ) -> Any:
        """Top-N items for all users efficiently computed in parallel.

        Parameters
        ----------
        n : int
            Number of items per user.
        exclude_seen : bool
            Whether to filter already-seen items.
        format : str
            Output format: "polars" or "pandas".

        Returns
        -------
        DataFrame
            A DataFrame with columns ``user_id``, ``item_id``, and ``score``.
        """
        self._check_fitted()
        import numpy as np

        if exclude_seen and self._interactions is not None:
            indptr = np.asarray(self._interactions.indptr, dtype=np.int64)
            indices = np.asarray(self._interactions.indices, dtype=np.int32)
        else:
            n_users = self._user_factors.shape[0]  # type: ignore
            indptr = np.zeros(n_users + 1, dtype=np.int64)
            indices = np.array([], dtype=np.int32)

        u_ids, i_ids, scores = _rust.svd_recommend_all(  # type: ignore
            self._user_factors.ravel().astype(np.float32),  # type: ignore
            self._item_factors.ravel().astype(np.float32),  # type: ignore
            self._user_biases.astype(np.float32),  # type: ignore
            self._item_biases.astype(np.float32),  # type: ignore
            self._global_mean,
            n,
            indptr,
            indices,
        )

        data = {
            "user_id": np.asarray(u_ids),
            "item_id": np.asarray(i_ids),
            "score": np.asarray(scores),
        }

        if format == "polars":
            try:
                import polars as pl

                return pl.DataFrame(data)
            except ImportError:
                pass

        import pandas as pd

        return pd.DataFrame(data)

    def recommend_users(self, item_id: int, n: int = 10) -> tuple[Any, Any]:
        """Top-N users for an item."""
        self._check_fitted()
        import numpy as np

        return _rust.svd_recommend_users(  # type: ignore
            self._user_factors.ravel().astype(np.float32),  # type: ignore
            self._item_factors.ravel().astype(np.float32),  # type: ignore
            self._user_biases.astype(np.float32),  # type: ignore
            self._item_biases.astype(np.float32),  # type: ignore
            self._global_mean,
            item_id,
            n,
        )

    @typing.no_type_check
    @property
    def user_factors(self):
        """User factor matrix (n_users, factors)."""
        self._check_fitted()
        return self._user_factors

    @typing.no_type_check
    @property
    def item_factors(self):
        """Item factor matrix (n_items, factors)."""
        self._check_fitted()
        return self._item_factors

    @typing.no_type_check
    @property
    def user_biases(self):
        """User bias vector (n_users,)."""
        self._check_fitted()
        return self._user_biases

    @typing.no_type_check
    @property
    def item_biases(self):
        """Item bias vector (n_items,)."""
        self._check_fitted()
        return self._item_biases

    @property
    def global_mean(self) -> float:
        """Global mean rating from training data."""
        self._check_fitted()
        return self._global_mean

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call .fit() first.")
