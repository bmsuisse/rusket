"""Non-negative Matrix Factorization (NMF) recommender."""

from __future__ import annotations

import typing
from typing import Any

import numpy as np

from .. import _rusket as _rust  # type: ignore
from ..model import ImplicitRecommender


class NMF(ImplicitRecommender):
    """Non-negative Matrix Factorization for collaborative filtering.

    Decomposes the user-item interaction matrix **R** into two non-negative
    matrices **W** (users × factors) and **H** (factors × items) such that
    ``R ≈ W @ H``.  The multiplicative update rules guarantee non-negativity
    without a projection step.

    Parameters
    ----------
    factors : int, default=64
        Number of latent factors.
    iterations : int, default=100
        Number of multiplicative update iterations.
    regularization : float, default=0.01
        L2 regularisation penalty applied to both W and H.
    seed : int, default=42
        Random seed for initialisation.
    verbose : int, default=0
        Verbosity level.
    use_gpu : bool
        If True, use GPU acceleration (CuPy or PyTorch) for recommendation.
        Falls back to CPU if no GPU backend found. Default False.
    """

    def __init__(
        self,
        factors: int = 64,
        iterations: int = 100,
        regularization: float = 0.01,
        seed: int = 42,
        verbose: int = 0,
        use_cuda: bool | None = None,
        **kwargs: Any,
    ) -> None:
        _use_cuda = kwargs.pop("use_gpu", use_cuda)  # backward compat
        super().__init__(**kwargs)
        self.factors = factors
        self.iterations = iterations
        self.regularization = float(regularization)
        from .._internal._config import _resolve_cuda

        self.use_cuda = _resolve_cuda(_use_cuda)
        self.seed = seed
        self.verbose = verbose

        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._n_users: int = 0
        self._n_items: int = 0
        self._fit_indptr: np.ndarray | None = None
        self._fit_indices: np.ndarray | None = None
        self.fitted: bool = False

    def __repr__(self) -> str:
        return f"NMF(factors={self.factors}, iterations={self.iterations}, regularization={self.regularization})"

    # ── fit ────────────────────────────────────────────────────────────

    def fit(self, interactions: Any = None) -> NMF:
        """Fit via multiplicative update rules (Rust-accelerated).

        Parameters
        ----------
        interactions : sparse matrix or numpy array, optional
            User-item interaction matrix.  If *None*, uses the matrix
            prepared by ``from_transactions()``.

        Returns
        -------
        NMF
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
            V = sp.csr_matrix(interactions, dtype=np.float64)
        elif isinstance(interactions, np.ndarray):
            V = sp.csr_matrix(interactions.astype(np.float64))
        else:
            raise TypeError(f"Expected scipy sparse matrix or numpy array, got {type(interactions)}")

        if not isinstance(V, sp.csr_matrix):
            V = V.tocsr()

        n_users, n_items = typing.cast(tuple[int, int], V.shape)

        indptr = np.asarray(V.indptr, dtype=np.int64)
        indices = np.asarray(V.indices, dtype=np.int32)
        data = np.asarray(V.data, dtype=np.float64)

        user_factors, item_factors = _rust.nmf_fit(
            indptr,
            indices,
            data,
            n_users,
            n_items,
            self.factors,
            self.iterations,
            self.regularization,
            self.seed,
            self.verbose > 0,
        )

        self._user_factors = user_factors
        self._item_factors = item_factors
        self._n_users = n_users
        self._n_items = n_items

        # Store CSR data for seen-item exclusion
        self._fit_indptr = indptr
        self._fit_indices = indices
        self.fitted = True

        if self.verbose:
            print("NMF fit complete.")

        return self

    # ── recommend ──────────────────────────────────────────────────────

    def recommend_items(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-N items for a user via W @ H^T.

        Parameters
        ----------
        user_id : int
            Internal user index.
        n : int, default=10
            Number of items to return.
        exclude_seen : bool, default=True
            Whether to exclude already-seen items.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(item_ids, scores)`` sorted by descending score.
        """
        self._check_fitted()
        assert self._user_factors is not None
        assert self._item_factors is not None

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
                if exclude_seen and self._fit_indptr is not None and self._fit_indices is not None:
                    start = self._fit_indptr[user_id]
                    end = self._fit_indptr[user_id + 1]
                    seen = self._fit_indices[start:end]
                    scores[seen] = -np.inf
                top_n = np.argsort(scores)[::-1][:n]
                return top_n.astype(np.intp), scores[top_n].astype(np.float32)

        scores = self._user_factors[user_id] @ self._item_factors.T

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
    def user_factors(self) -> np.ndarray:
        """User factor matrix (n_users, factors)."""
        self._check_fitted()
        assert self._user_factors is not None
        return self._user_factors

    @property
    def item_factors(self) -> np.ndarray:
        """Item factor matrix (n_items, factors)."""
        self._check_fitted()
        assert self._item_factors is not None
        return self._item_factors
