from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from rusket._rusket import lightgcn_fit  # type: ignore[attr-defined]

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


class LightGCN:
    """LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.

    A state-of-the-art collaborative filtering model that propagates embeddings
    over the user–item bipartite graph without non-linear transformations.

    Typical training time on ml-100k: < 0.5s/epoch.

    Args:
        factors: Embedding dimensionality (latent factors).
        k_layers: Number of graph-propagation layers (1–4).
        learning_rate: Adam learning rate.
        lambda_: L2 regularization coefficient.
        iterations: Number of training epochs.
        random_state: Seed for reproducible training.
        verbose: Print training progress.
    """

    def __init__(
        self,
        factors: int = 64,
        k_layers: int = 3,
        learning_rate: float = 1e-3,
        lambda_: float = 1e-4,
        iterations: int = 20,
        random_state: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.factors = factors
        self.k_layers = k_layers
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.iterations = iterations
        self.random_state = random_state
        self.verbose = verbose

        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._user_map: dict[int, int] = {}
        self._item_map: dict[int, int] = {}
        self._rev_item_map: dict[int, int] = {}

    @classmethod
    def from_transactions(
        cls,
        data: pd.DataFrame | pl.DataFrame,
        user_col: str | None = None,
        item_col: str | None = None,
        **kwargs,
    ) -> LightGCN:
        model = cls(**kwargs)
        model._fit_from_df(data, user_col, item_col)
        return model

    def _fit_from_df(
        self,
        data: pd.DataFrame | pl.DataFrame,
        user_col: str | None,
        item_col: str | None,
    ) -> None:

        if hasattr(data, "to_pandas"):
            data = data.to_pandas()

        user_col = user_col or "user_id"
        item_col = item_col or "item_id"

        users = np.asarray(data[user_col])  # type: ignore[call-overload]
        items = np.asarray(data[item_col])  # type: ignore[call-overload]

        unique_users = np.unique(users.astype(object))  # type: ignore[arg-type]
        unique_items = np.unique(items.astype(object))  # type: ignore[arg-type]

        self._user_map = {u: i for i, u in enumerate(unique_users)}
        self._item_map = {it: i for i, it in enumerate(unique_items)}
        self._rev_item_map = {i: it for it, i in self._item_map.items()}

        u_idx = np.array([self._user_map[u] for u in users], dtype=np.int32)
        i_idx = np.array([self._item_map[it] for it in items], dtype=np.int32)

        n_users = len(unique_users)
        n_items = len(unique_items)

        # Build CSR (user → items) and CSC-as-CSR (item → users)
        ui_csr = sp.csr_matrix(
            (np.ones(len(u_idx)), (u_idx, i_idx)),
            shape=(n_users, n_items),
        )
        ui_csr.sort_indices()

        iu_csr = ui_csr.T.tocsr()
        iu_csr.sort_indices()

        u_indptr = ui_csr.indptr.astype(np.int64)
        u_indices = ui_csr.indices.astype(np.int32)
        i_indptr = iu_csr.indptr.astype(np.int64)
        i_indices = iu_csr.indices.astype(np.int32)

        seed = self.random_state if self.random_state is not None else int(np.random.randint(1 << 31))

        eu, ei = lightgcn_fit(
            u_indptr,
            u_indices,
            i_indptr,
            i_indices,
            n_users,
            n_items,
            self.factors,
            self.k_layers,
            float(self.learning_rate),
            float(self.lambda_),
            self.iterations,
            int(seed),
            bool(self.verbose),
        )

        # Compute final propagated embeddings for scoring
        self._user_factors = np.asarray(eu, dtype=np.float32)
        self._item_factors = np.asarray(ei, dtype=np.float32)
        self._n_users = n_users
        self._n_items = n_items

    def recommend_items(self, user_id: int, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Return top-n recommended item IDs and scores for a given user.

        Args:
            user_id: Original user ID (before encoding).
            n: Number of recommendations.

        Returns:
            Tuple of (item_ids, scores) arrays sorted by descending score.
        """
        if self._user_factors is None:
            raise RuntimeError("Model not fitted yet.")
        uid = self._user_map.get(user_id)
        if uid is None:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        assert self._item_factors is not None
        scores = self._user_factors[uid] @ self._item_factors.T
        top_idx = np.argsort(scores)[::-1][:n]
        original_ids = np.array([self._rev_item_map[i] for i in top_idx], dtype=np.int64)
        return original_ids, scores[top_idx]
