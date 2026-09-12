"""ALS (Alternating Least Squares) collaborative filtering recommender."""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np

from .. import _rusket as _rust  # type: ignore
from ..model import ImplicitRecommender


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
    cg_iters : int, optional
        Conjugate Gradient iterations per user/item solve (ignored when
        ``use_cholesky=True``). Defaults to ``5``.

        Each outer iteration warm-starts CG from the previous iteration's
        factors, so only a few inner steps are needed to track the moving
        solution. Measured on 3.8M real interactions (38k users x 47k items,
        factors=256, 15 outer iterations), NDCG@10 by ``cg_iters``: 3 -> .0696,
        5 -> .0721, 8 -> .0717, 15 -> .0716, i.e. converged by 5 and flat after
        it; the same holds at factors 32/64/128. Raising it only costs time.

    use_cholesky : bool
        Use a direct Cholesky solve instead of iterative CG. Exact solution;
        faster when users have many interactions relative to ``factors``.
    use_eals : bool
        Use element-wise ALS (eALS). Usually faster than Cholesky/CG and less memory intensive.
    eals_iters : int
        Number of inner iterations for eALS (default 1).
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
    popularity_weighting : str
        Weighting scheme for missing data in **eALS**. Items that are frequently
        interacted-with provide stronger negative signals when *not* chosen.
        Options: ``"none"`` (uniform, default), ``"sqrt"``, ``"log"``, ``"linear"``.
        Only used when ``use_eals=True``.
    use_biases : bool
        If True, learn global bias (μ), user biases (b_u), and item biases (b_i)
        so that prediction becomes ``μ + b_u + b_i + w_u · h_i``.
    alpha_view : float
        Confidence scaling for **view** interactions in VALS mode.
        Pass ``view_matrix`` to ``fit()`` to enable. Default 10.0.
    view_target : float
        Target value for view interactions (between 0.0 and 1.0).
        Purchases always target 1.0. Default 0.5.
    use_cuda : bool
        If True, use CUDA acceleration (CuPy or PyTorch) for batch
        recommendation. Falls back to CPU if no CUDA backend found.
        Default False.

    Examples
    --------
    Fold in a new user without retraining the entire model matrix:

    >>> import rusket
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> # Fit model on some data
    >>> model = rusket.ALS(factors=8).fit(csr_matrix(np.random.randint(0, 2, size=(10, 20))))
    >>> # New user interacts with items 3, 5, and 12
    >>> latent_factors = model.recalculate_user([3, 5, 12])
    >>> # `latent_factors` is a 1D array of length `factors=8`
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        alpha: float = 40.0,
        iterations: int = 15,
        seed: int = 42,
        verbose: int = 0,
        cg_iters: int | None = None,
        use_cholesky: bool = False,
        use_eals: bool = False,
        eals_iters: int = 1,
        anderson_m: int = 0,
        popularity_weighting: str = "none",
        use_biases: bool = False,
        alpha_view: float = 10.0,
        view_target: float = 0.5,
        use_cuda: bool | None = None,
        **kwargs: Any,
    ) -> None:
        _use_cuda = kwargs.pop("use_gpu", use_cuda)  # backward compat
        super().__init__(data=None, **kwargs)
        self.factors = factors
        self.regularization = float(regularization)
        self.alpha = float(alpha)
        self.iterations = iterations
        self.seed = seed
        self.verbose = verbose
        self.cg_iters = 5 if cg_iters is None else cg_iters
        self.use_cholesky = use_cholesky
        self.use_eals = use_eals
        self.eals_iters = eals_iters
        self.anderson_m = anderson_m
        self.popularity_weighting = popularity_weighting
        self.use_biases = use_biases
        self.alpha_view = float(alpha_view)
        self.view_target = float(view_target)
        from .._internal._config import _resolve_cuda

        self.use_cuda = _resolve_cuda(_use_cuda)
        self._user_factors: Any = None
        self._item_factors: Any = None
        self._n_users: int = 0
        self._n_items: int = 0
        self._fit_indptr: Any = None
        self._fit_indices: Any = None
        self._item_pop_weights: Any = None
        self._global_bias: float = 0.0
        self._user_biases: Any = None
        self._item_biases: Any = None
        self._user_labels: list[Any] | None = None
        self._item_labels: list[Any] | None = None
        self.fitted: bool = False

    def __repr__(self) -> str:
        return (
            f"ALS(factors={self.factors}, regularization={self.regularization}, "
            f"alpha={self.alpha}, iterations={self.iterations})"
        )

    def fit(self, interactions: Any = None, *, view_matrix: Any = None) -> ALS:
        """Fit the model to the user-item interaction matrix.

        Parameters
        ----------
        interactions : sparse matrix or numpy array, optional
            If None, uses the matrix prepared by ``from_transactions()``.
        view_matrix : sparse matrix or numpy array, optional
            Optional view/browse interaction matrix (same shape as ``interactions``).
            When provided, enables **VALS** mode: views are treated as weaker
            positive signals with confidence ``alpha_view`` targeting ``view_target``.

        Raises
        ------
        RuntimeError
            If the model is already fitted. Create a new instance to refit.
        TypeError
            If the input matrix is not a recognizable sparse matrix or numpy array.
        """
        if interactions is None:
            interactions = getattr(self, "_prepared_interactions", None)
            if interactions is None:
                raise ValueError("No interactions provided. Pass a matrix or use from_transactions() first.")
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

        # VALS: merge view interactions into the main matrix
        if view_matrix is not None:
            if sp.issparse(view_matrix):
                view_csr = sp.csr_matrix(view_matrix, dtype=np.float32)
            elif isinstance(view_matrix, np.ndarray):
                view_csr = sp.csr_matrix(view_matrix.astype(np.float32))
            else:
                raise TypeError(f"Expected sparse matrix or ndarray for view_matrix, got {type(view_matrix)}")
            if view_csr.shape != csr.shape:
                raise ValueError(f"view_matrix shape {view_csr.shape} != interactions shape {csr.shape}")
            # Scale purchase entries: data encodes confidence. We use 1.0 for purchases.
            # Scale view entries: use (alpha_view / alpha) * view_target as the data value,
            # so that confidence = 1 + alpha * data = 1 + alpha_view * view_target
            # This gives views intermediate confidence between unobserved (1) and purchase (1+alpha)
            view_only = view_csr - view_csr.multiply(csr.astype(bool))  # views not already purchased
            view_only.eliminate_zeros()
            if view_only.nnz > 0:
                scale = (self.alpha_view * self.view_target) / max(self.alpha, 1e-9)
                view_only.data[:] = scale
                csr = csr + view_only
                csr.sum_duplicates()

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

        # Compute item popularity weights for eALS
        item_pop_weights = None
        if self.use_eals and self.popularity_weighting != "none":
            col_sums = np.bincount(indices, minlength=n_items).astype(np.float64)
            col_sums /= max(col_sums.sum(), 1.0)  # normalize to probabilities
            if self.popularity_weighting == "sqrt":
                item_pop_weights = np.sqrt(col_sums).astype(np.float32)
            elif self.popularity_weighting == "log":
                item_pop_weights = np.log1p(col_sums * n_items).astype(np.float32)
                item_pop_weights /= max(item_pop_weights.max(), 1e-9)
            elif self.popularity_weighting == "linear":
                item_pop_weights = col_sums.astype(np.float32)
            else:
                raise ValueError(
                    f"Unknown popularity_weighting: '{self.popularity_weighting}'. "
                    "Must be one of: 'none', 'sqrt', 'log', 'linear'."
                )
            # Ensure minimum weight so rare items aren't zeroed out
            item_pop_weights = np.maximum(item_pop_weights, 1e-6)

        self._user_factors, self._item_factors, self._global_bias, self._user_biases, self._item_biases = (
            _rust.als_fit_implicit(
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
                self.use_eals,
                self.eals_iters,
                item_pop_weights,
                self.use_biases,
            )
        )
        self._n_users = n_users
        self._n_items = n_items
        self._fit_indptr = indptr
        self._fit_indices = indices
        self._item_pop_weights = item_pop_weights
        self.fitted = True
        return self

    def recalculate_user(self, user_items: Any) -> np.ndarray:
        """Calculate the latent factors for a new or existing user given their interacted items.

        Parameters
        ----------
        user_items : list of int or 1D array-like
            The item indices the user has interacted with. If the model was fitted
            using a DataFrame with item names, these should be the mapped item indices
            from 0 to n_items - 1.

            Note: Confidence values for interactions are currently treated as 1.

        Returns
        -------
        ndarray
            A 1D numpy array of shape (factors,) containing the user's latent factors.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        ValueError
            If any item index is out of bounds.
        """
        import numpy as np

        self._check_fitted()

        # Validate indices
        indices = np.asarray(user_items, dtype=np.int32)
        if len(indices) > 0:
            if indices.min() < 0 or indices.max() >= self._n_items:
                raise ValueError(f"Item indices must be between 0 and {self._n_items - 1}")

        # We assume all interactions have a value of 1.0 for now
        data = np.ones_like(indices, dtype=np.float32)

        return _rust.als_recalculate_user(
            self._item_factors,
            indices,
            data,
            self.regularization,
            self.alpha,
            self.cg_iters,
            self.use_cholesky,
            self.use_eals,
            self.eals_iters,
            self._item_pop_weights,
        )

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

        if self.use_cuda:
            from ..integrations.cuda import get_cuda_backend_safe, gpu_score_user

            gpu = get_cuda_backend_safe()
            if gpu is not None:
                backend, lib = gpu
                scores = gpu_score_user(
                    self._user_factors[user_id],
                    self._item_factors,
                    backend,
                    lib,
                )
                # Add biases
                scores = scores + self._global_bias + self._user_biases[user_id] + self._item_biases
                if exclude_seen and self._fit_indptr is not None and self._fit_indices is not None:
                    start = self._fit_indptr[user_id]
                    end = self._fit_indptr[user_id + 1]
                    seen = self._fit_indices[start:end]
                    scores[seen] = -np.inf
                top_n = np.argsort(scores)[::-1][:n]
                return top_n.astype(np.int32), scores[top_n].astype(np.float32)

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
            self._global_bias,
            self._user_biases,
            self._item_biases,
        )
        return np.asarray(ids), np.asarray(scores)

    @classmethod
    def from_factors(
        cls,
        user_factors: Any,
        item_factors: Any,
        *,
        user_labels: Any = None,
        item_labels: Any = None,
        global_bias: float = 0.0,
        user_biases: Any = None,
        item_biases: Any = None,
    ) -> ALS:
        """Build a scoring-only ALS from factor matrices computed elsewhere.

        Training and scoring are often separate jobs: one fits the model and
        persists the factors (a Delta table, a feature store, a parquet file),
        another loads them later to score. Without this, that second job has
        no way to reach :meth:`batch_recommend` and ends up re-implementing
        the blocked top-N in Python.

        The returned model supports scoring (``batch_recommend``,
        ``recommend_items``, ``similar_items``) but not ``fit`` -- there is no
        interaction matrix behind it, so pass ``exclude=`` to
        ``batch_recommend`` if items need suppressing.

        Parameters
        ----------
        user_factors, item_factors : array-like
            ``(n_users, factors)`` and ``(n_items, factors)``, row-major.
        user_labels, item_labels : array-like, optional
            External ids, so results come back in the caller's id space
            instead of internal indices.
        global_bias, user_biases, item_biases : optional
            Bias terms, if the original model was fitted with them.

        Examples
        --------
        >>> model = ALS.from_factors(u, i, user_labels=customer_sks, item_labels=article_sks)
        >>> model.batch_recommend(n=20, exclude=already_bought, format="pandas")
        """
        import numpy as np

        uf = np.ascontiguousarray(user_factors, dtype=np.float32)
        itf = np.ascontiguousarray(item_factors, dtype=np.float32)
        if uf.ndim != 2 or itf.ndim != 2:
            raise ValueError("user_factors and item_factors must both be 2-D")
        if uf.shape[1] != itf.shape[1]:
            raise ValueError(
                f"factor dimension mismatch: user_factors has {uf.shape[1]}, item_factors has {itf.shape[1]}"
            )

        model = cls(factors=uf.shape[1])
        model._user_factors = uf
        model._item_factors = itf
        model._n_users, model._n_items = uf.shape[0], itf.shape[0]
        model._global_bias = float(global_bias)
        model._user_biases = None if user_biases is None else np.ascontiguousarray(user_biases, dtype=np.float32)
        model._item_biases = None if item_biases is None else np.ascontiguousarray(item_biases, dtype=np.float32)
        for name, labels, expected in (
            ("user_labels", user_labels, model._n_users),
            ("item_labels", item_labels, model._n_items),
        ):
            if labels is None:
                continue
            seq = list(labels)
            if len(seq) != expected:
                raise ValueError(f"{name} has {len(seq)} entries but there are {expected} rows")
            setattr(model, f"_{name}", seq)
        # ponytail: no _fit_indptr/_fit_indices — there is no training matrix,
        # so exclude_seen has nothing to exclude; callers pass exclude=.
        model.fitted = True
        return model

    def batch_recommend(
        self,
        n: int = 10,
        exclude_seen: bool = True,
        format: Literal["pandas", "polars", "spark"] = "polars",
        exclude: Any = None,
    ) -> Any:
        """Top-N items for all users efficiently computed in parallel.

        Parameters
        ----------
        n : int, default=10
            The number of items to recommend per user.
        exclude_seen : bool, default=True
            Whether to exclude items the user has already interacted with
            during training.
        format : str, default="polars"
            The DataFrame format to return. One of "pandas", "polars", or "spark".
        exclude : sparse matrix, optional
            Explicit ``(n_users, n_items)`` mask of items to exclude per user,
            in internal index space. Use this when the set to suppress is not
            the training matrix -- e.g. scoring cross-sell potential against
            "everything this customer has ever bought or been quoted", which is
            a wider set than what the model was fitted on. Takes precedence
            over ``exclude_seen``.

        Returns
        -------
        DataFrame
            A DataFrame with columns `user_id`, `item_id`, and `score`.
        """
        import numpy as np

        self._check_fitted()
        if exclude is not None:
            from scipy import sparse as sp

            exc = exclude if sp.isspmatrix_csr(exclude) else sp.csr_matrix(exclude)
            if exc.shape != (self._n_users, self._n_items):
                raise ValueError(f"exclude must have shape ({self._n_users}, {self._n_items}), got {exc.shape}")
            exc_indptr = np.asarray(exc.indptr, dtype=np.int64)
            exc_indices = np.asarray(exc.indices, dtype=np.int32)
        elif exclude_seen and self._fit_indptr is not None and self._fit_indices is not None:
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
            self._global_bias,
            self._user_biases,
            self._item_biases,
        )

        u_ids_arr = np.asarray(u_ids)
        i_ids_arr = np.asarray(i_ids)
        scores_arr = np.asarray(scores)

        if format == "pandas":
            # ponytail: build the pandas frame directly from numpy instead of
            # round-tripping through polars — callers that only have pandas
            # installed (e.g. Spark UDF workers) shouldn't be forced to import
            # polars just to immediately call .to_pandas() on the result.
            from rusket._dependencies import import_optional_dependency

            pd = import_optional_dependency("pandas")

            user_ids_out: np.ndarray = u_ids_arr
            item_ids_out: np.ndarray = i_ids_arr
            if self._user_labels is not None and len(self._user_labels) == self._n_users:
                user_ids_out = np.asarray(self._user_labels, dtype=object)[u_ids_arr]
            if self._item_labels is not None and len(self._item_labels) == self._n_items:
                item_ids_out = np.asarray(self._item_labels, dtype=object)[i_ids_arr]
            return pd.DataFrame({"user_id": user_ids_out, "item_id": item_ids_out, "score": scores_arr})

        from rusket._dependencies import import_optional_dependency

        pl = import_optional_dependency("polars")

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
        elif format == "spark":
            from rusket._dependencies import import_optional_dependency

            pyspark_sql = import_optional_dependency("pyspark.sql", "pyspark")
            SparkSession = pyspark_sql.SparkSession

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
            self._global_bias,
            self._user_biases,
            self._item_biases,
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

    @property
    def global_bias(self) -> float:
        """Global bias μ (scalar). Zero when use_biases=False."""
        self._check_fitted()
        return self._global_bias

    @property
    def user_biases(self) -> Any:
        """User bias vector b_u (n_users,). Zeros when use_biases=False."""
        self._check_fitted()
        return self._user_biases

    @property
    def item_biases(self) -> Any:
        """Item bias vector b_i (n_items,). Zeros when use_biases=False."""
        self._check_fitted()
        return self._item_biases

    def build_ann_index(
        self,
        backend: str = "native",
        index_type: str = "hnsw",
        **kwargs: Any,
    ) -> Any:
        """Build an Approximate Nearest Neighbor index from item factors.

        Parameters
        ----------
        backend : str
            ``"native"`` uses the built-in Rust random-projection forest
            (:class:`~rusket.ApproximateNearestNeighbors`).
            ``"faiss"`` uses FAISS (requires ``pip install faiss-cpu``).
        index_type : str
            For ``"faiss"`` backend: ``"flat"``, ``"hnsw"``, ``"ivfflat"``, ``"ivfpq"``.
            Ignored for ``"native"`` backend.
        **kwargs
            Additional arguments passed to the index builder.

        Returns
        -------
        index
            A fitted ANN index with a ``query()`` / ``kneighbors()`` method.
        """
        self._check_fitted()

        if backend == "native":
            from ..export.ann import ApproximateNearestNeighbors

            n_trees = kwargs.pop("n_trees", 10)
            leaf_size = kwargs.pop("leaf_size", 30)
            seed = kwargs.pop("seed", 42)
            idx = ApproximateNearestNeighbors(n_trees=n_trees, leaf_size=leaf_size, seed=seed)
            return idx.fit(self._item_factors)
        elif backend == "faiss":
            from ..export.faiss_ann import FAISSIndex

            return FAISSIndex(index_type=index_type, **kwargs).build(self._item_factors)
        else:
            raise ValueError(f"Unknown backend: '{backend}'. Must be 'native' or 'faiss'.")

    def _check_fitted(self) -> None:
        if self._user_factors is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")


class eALS(ALS):
    """Element-wise ALS (eALS) collaborative filtering model.

    A convenience wrapper around :class:`ALS` that sets ``use_eals=True`` by default.
    eALS updates latent factors element-by-element rather than block-wise, which
    is often faster and less memory-intensive for implicit datasets while yielding
    comparable or better recommendation quality.

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
    eals_iters : int
        Number of inner iterations for eALS (default 1).
    **kwargs
        Additional arguments passed to :class:`ALS`.
    """

    def __init__(self, *args: Any, use_eals: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, use_eals=use_eals, **kwargs)

    def __repr__(self) -> str:
        return (
            f"eALS(factors={self.factors}, regularization={self.regularization}, "
            f"alpha={self.alpha}, iterations={self.iterations}, eals_iters={self.eals_iters})"
        )
