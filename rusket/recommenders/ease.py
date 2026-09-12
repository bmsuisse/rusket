"""EASE (Embarrassingly Shallow Autoencoders) collaborative filtering recommender."""

from __future__ import annotations

import typing
from typing import Any

from .. import _rusket as _rust  # type: ignore
from ..model import ImplicitRecommender

# ponytail: peak-memory guard for EASE.fit(). EASE builds a dense
# n_items x n_items Gram matrix and Cholesky-inverts it -- O(n_items^2)
# in both time and memory, which is the algorithm, not a bug. During the
# Rust solve, two n_items x n_items f64 buffers are briefly alive at once
# (see the `drop()` calls in `ease_compute_weights` in src/ease.rs), so
# peak transient bytes is ~= 2 * 8 * n_items**2 = 16 * n_items**2.
# At the real production catalog size of 46,560 items that is
# 16 * 46560**2 / 1024**3 ~= 32.3 GiB -- before even counting the f32
# output matrix retained afterward. Rather than let that OOM-kill the
# process 10 minutes into a Cholesky solve, we estimate this up front and
# fail immediately with a clear, actionable error.
_EASE_PEAK_BYTES_PER_ITEM_SQUARED = 16.0
# Default cap chosen so the default is safe on a typical dev machine
# (~16-32 GB RAM) while still covering catalogs up to ~22k items
# (16 * 22000**2 / 1024**3 ~= 8.0 GiB), comfortably above what EASE is
# normally applied to. Pass a larger `max_memory_gb` explicitly to opt in
# on a bigger machine.
_EASE_DEFAULT_MAX_MEMORY_GB = 8.0


def _estimate_peak_bytes(n_items: int) -> float:
    return _EASE_PEAK_BYTES_PER_ITEM_SQUARED * float(n_items) ** 2


class EASE(ImplicitRecommender):
    """Embarrassingly Shallow Autoencoders for Sparse Data (EASE).

    An implicit collaborative filtering algorithm that computes a closed-form
    item-item similarity matrix by solving a ridge regression problem. EASE
    often achieves state-of-the-art recommendation quality and very fast
    inference, particularly on datasets with strong item-item correlations.

    Parameters
    ----------
    regularization : float
        L2 regularization weight (lambda). Higher values encourage smaller weights
        and reduce overfitting. Default is 500.0.
    use_gpu : bool
        If True, use GPU acceleration (CuPy or PyTorch) for recommendation.
        Falls back to CPU if no GPU backend found. Default False.
    """

    def __init__(
        self,
        regularization: float = 500.0,
        verbose: int = 0,
        use_cuda: bool | None = None,
        **kwargs: Any,
    ) -> None:
        _use_cuda = kwargs.pop("use_gpu", use_cuda)  # backward compat
        super().__init__(data=None, **kwargs)
        self.regularization = float(regularization)
        self.verbose = verbose
        from .._internal._config import _resolve_cuda

        self.use_cuda = _resolve_cuda(_use_cuda)

        self.item_weights: Any = None
        self._n_users: int = 0
        self._n_items: int = 0
        self._fit_indptr: Any = None
        self._fit_indices: Any = None
        self._fit_data: Any = None

        self._user_labels: list[Any] | None = None
        self._item_labels: list[Any] | None = None
        self.fitted: bool = False

    def __repr__(self) -> str:
        return f"EASE(regularization={self.regularization})"

    def fit(self, interactions: Any = None, max_memory_gb: float = _EASE_DEFAULT_MAX_MEMORY_GB) -> EASE:
        """Fit the model to the user-item interaction matrix (Rust-accelerated).

        Parameters
        ----------
        interactions : sparse matrix or numpy array, optional
            If None, uses the matrix prepared by ``from_transactions()``.
        max_memory_gb : float
            Safety cap on the estimated *peak transient* memory (GiB) EASE
            will need during the Cholesky solve, checked before any large
            allocation happens. EASE is inherently O(n_items^2) in memory:
            estimated peak bytes ~= 16 * n_items**2 (two dense
            n_items x n_items f64 matrices briefly alive at once inside the
            Rust solver). The default of 8 GiB caps this at ~22,000 items,
            which is safe on a typical dev machine and covers most
            catalogs. Raise this explicitly (e.g. to ~40 GiB) if you have a
            larger machine and truly need to fit EASE on a bigger catalog
            (the real 46,560-item production catalog needs ~32 GiB); on
            catalogs of that size, consider ItemKNN or ALS instead, which
            don't require a dense n_items^2 matrix.
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

        n_users, n_items = typing.cast(tuple[int, int], csr.shape)

        estimated_gb = _estimate_peak_bytes(n_items) / (1024**3)
        if estimated_gb > max_memory_gb:
            raise MemoryError(
                f"EASE.fit() refused to start: {n_items} items would need an estimated "
                f"{estimated_gb:.1f} GiB of peak memory for the dense n_items x n_items "
                f"Gram matrix and Cholesky solve, which exceeds the max_memory_gb="
                f"{max_memory_gb:.1f} safety cap. EASE is O(n_items^2) in memory, so this "
                "only gets worse with more items. Options: (1) reduce the item catalog "
                "(e.g. drop long-tail items), (2) use ItemKNN or ALS instead, which scale "
                "far better with item count, or (3) if you have enough RAM and really need "
                "EASE at this size, pass a larger max_memory_gb explicitly to opt in."
            )

        indptr = np.asarray(csr.indptr, dtype=np.int64)
        indices = np.asarray(csr.indices, dtype=np.int32)
        data = np.asarray(csr.data, dtype=np.float32)

        if self.verbose:
            print(f"EASE fitting {n_items} items via Rust (Cholesky)...")

        self.item_weights = _rust.ease_fit(
            indptr,
            indices,
            data,
            n_items,
            float(self.regularization),
        )

        self._n_users = n_users
        self._n_items = n_items
        self._fit_indptr = indptr
        self._fit_indices = indices
        self._fit_data = data
        self.fitted = True

        if self.verbose:
            print("EASE fit complete.")

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

        if self.use_cuda:
            from ..integrations.cuda import get_cuda_backend_safe, gpu_sparse_dense_matmul

            gpu = get_cuda_backend_safe()
            if gpu is not None:
                backend, lib = gpu
                # User's interaction row (sparse) @ item_weights (dense)
                start = self._fit_indptr[user_id]
                end = self._fit_indptr[user_id + 1]
                user_indices = self._fit_indices[start:end]
                user_data = self._fit_data[start:end]
                # Build single-row CSR
                row_indptr = np.array([0, len(user_indices)], dtype=np.int64)
                scores = gpu_sparse_dense_matmul(
                    user_data,
                    user_indices,
                    row_indptr,
                    (1, self._n_items),
                    self.item_weights,
                    backend,
                    lib,
                ).ravel()
                if exclude_seen:
                    scores[user_indices] = -np.inf
                top_n = np.argsort(scores)[::-1][:n]
                return top_n.astype(np.int32), scores[top_n].astype(np.float32)

        if exclude_seen and self._fit_indptr is not None and self._fit_indices is not None:
            exc_indptr = self._fit_indptr
            exc_indices = self._fit_indices
        else:
            exc_indptr = np.zeros(self._n_users + 1, dtype=np.int64)
            exc_indices = np.array([], dtype=np.int32)

        ids, scores = _rust.ease_recommend_items(  # type: ignore[attr-defined]
            self.item_weights,
            self._fit_indptr,
            self._fit_indices,
            self._fit_data,
            user_id,
            n,
            exc_indptr,
            exc_indices,
        )
        return np.asarray(ids), np.asarray(scores)

    @property
    def item_factors(self) -> Any:
        """Item factor matrix (n_items, n_items)."""
        self._check_fitted()
        return self.item_weights

    def _check_fitted(self) -> None:
        if self.item_weights is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
