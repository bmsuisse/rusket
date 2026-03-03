"""Factorizing Personalized Markov Chains (FPMC) for Sequential Recommendation."""

from __future__ import annotations

from typing import Any

from . import _rusket as _rust  # type: ignore
from .model import SequentialRecommender


class FPMC(SequentialRecommender):
    """Factorizing Personalized Markov Chains (FPMC) model for sequential recommendation.

    FPMC combines Matrix Factorization (modeling user preferences) and Markov Chains
    (modeling sequential transitions between items). It is highly effective for tasks
    where both personal taste and sequential behavior matter (e.g., next-basket delivery).

    Parameters
    ----------
    factors : int
        Number of latent factors (default: 64).
    learning_rate : float
        SGD learning rate (default: 0.05).
    regularization : float
        L2 regularization weight (default: 0.01).
    iterations : int
        Number of passes over the transitions (default: 150).
    seed : int
        Random seed for sampling (default: 42).
    verbose : bool
        Whether to print training progress (default: False).
    use_gpu : bool
        If True, use GPU acceleration (CuPy or PyTorch) for recommendation.
        Falls back to CPU if no GPU backend found. Default False.
    """

    def __init__(
        self,
        factors: int = 64,
        learning_rate: float = 0.05,
        regularization: float = 0.01,
        iterations: int = 150,
        seed: int = 42,
        time_aware: bool = False,
        max_time_steps: int = 256,
        verbose: int = 0,
        use_gpu: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(data=None, **kwargs)
        self.factors = factors
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.iterations = iterations
        self.seed = seed
        self.time_aware = time_aware
        self.max_time_steps = max_time_steps
        self.verbose = verbose
        from ._config import _resolve_gpu

        self.use_gpu = _resolve_gpu(use_gpu)

        # Vu, Viu, Vil, Vli, Vtime
        self._vu: Any = None
        self._viu: Any = None
        self._vil: Any = None
        self._vli: Any = None
        self._vtime: Any = None

        import numpy as np

        # Last items for users to make predictions
        self._user_last_items: dict[int, int] = {}
        self._user_seen_items: dict[int, np.ndarray] = {}
        self._user_last_timestamps: dict[int, int] | None = {} if time_aware else None

        self.fitted: bool = False
        self._pending_sequences: list[list[int]] | None = None
        self._pending_timestamps: list[list[int]] | None = None

    def __repr__(self) -> str:
        return (
            f"FPMC(factors={self.factors}, learning_rate={self.learning_rate}, "
            f"regularization={self.regularization}, iterations={self.iterations})"
        )

    def fit(
        self,
        sequences: list[list[int]] | None = None,
        timestamps: list[list[int]] | None = None,
        n_items: int | None = None,
    ) -> FPMC:
        """Fit the FPMC model to a list of sequential interactions.

        Parameters
        ----------
        sequences : list of list of int, optional
            List of item sequences, where each sequence belongs to a unique user.
            Users are assigned IDs from 0 to len(sequences)-1.
            If None, uses data prepared by ``from_transactions()``.
        timestamps : list of list of int, optional
            Corresponding unix timestamps for sequences if time_aware is True.
        n_items : int | None
            Maximum number of items. If None, it is inferred from data.
        """
        if sequences is None:
            sequences = getattr(self, "_pending_sequences", None)
            timestamps = getattr(self, "_pending_timestamps", None)
            if sequences is None:
                raise ValueError("No sequences provided. Pass sequences or use from_transactions() first.")
        import numpy as np

        if self.fitted:
            raise RuntimeError("Model is already fitted.")

        if not isinstance(sequences, list) or not all(isinstance(seq, list) for seq in sequences):
            raise TypeError("Expected a list of lists of integers representing sequences.")

        self._n_users = len(sequences)
        max_item_in_data = max((max(seq) for seq in sequences if seq), default=-1)
        self._n_items = max_item_in_data + 1 if n_items is None else n_items

        if max_item_in_data >= self._n_items:
            raise ValueError(f"Observed item ID {max_item_in_data} but n_items is {self._n_items}.")

        # Flatten sequences into CSR-like structure for Rust
        indptr = np.zeros(self._n_users + 1, dtype=np.int64)
        indices = []
        flat_timestamps = [] if self.time_aware and timestamps is not None else None

        if self.time_aware and timestamps is not None:
            for u, (seq, ts_seq) in enumerate(zip(sequences, timestamps, strict=False)):
                if seq:
                    self._user_last_items[u] = seq[-1]
                    self._user_last_timestamps[u] = ts_seq[-1] if self._user_last_timestamps is not None else None
                    self._user_seen_items[u] = np.unique(seq)
                    indices.extend(seq)
                    flat_timestamps.extend(ts_seq)  # type: ignore
                indptr[u + 1] = len(indices)
        else:
            for u, seq in enumerate(sequences):
                if seq:
                    self._user_last_items[u] = seq[-1]
                    self._user_seen_items[u] = np.unique(seq)
                    indices.extend(seq)
                indptr[u + 1] = len(indices)

        indices_arr = np.array(indices, dtype=np.int32)
        timestamps_arr = np.array(flat_timestamps, dtype=np.int64) if flat_timestamps is not None else None

        self._vu, self._viu, self._vil, self._vli, self._vtime = _rust.fpmc_fit(  # type: ignore[attr-defined]
            indptr,
            indices_arr,
            timestamps_arr,
            self._n_users,
            self._n_items,
            self.factors,
            self.max_time_steps if self.time_aware else 0,
            self.learning_rate,
            self.regularization,
            self.iterations,
            self.seed,
            bool(self.verbose),
        )

        self.fitted = True
        return self

    def recommend_items(
        self,
        user_id: int,
        timestamp: int | None = None,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> tuple[Any, Any]:
        """Top-N sequential items for a user."""
        import numpy as np

        self._check_fitted()
        if user_id < 0 or user_id >= self._n_users:
            raise ValueError(f"user_id {user_id} is out of bounds for model with {self._n_users} users.")

        prev_item = self._user_last_items.get(user_id, -1)
        prev_ts = self._user_last_timestamps.get(user_id) if self._user_last_timestamps else None

        # Calculate scores: x_{u,l,i} = dot(V_u, V_iu) + dot(V_il, V_li) if prev_item >= 0 else dot(V_u, V_iu)
        v_u = self._vu[user_id]

        # MF term
        scores = np.dot(self._viu, v_u)

        # Markov Chain term
        if prev_item >= 0:
            v_l = self._vli[prev_item]

            if self.time_aware and self._vtime is not None:
                eff_v_l = v_l.copy()
                if timestamp is not None and prev_ts is not None:
                    time_diff = max(0, timestamp - prev_ts)
                    days_diff = int(time_diff // 86400)
                    t_diff_adjusted = min(days_diff, self.max_time_steps - 1)
                    eff_v_l += self._vtime[t_diff_adjusted]
                scores += np.dot(self._vil, eff_v_l)
            else:
                scores += np.dot(self._vil, v_l)

        # Top-N selection
        if exclude_seen:
            seen_items = self._user_seen_items.get(user_id)
            if seen_items is not None:
                scores[seen_items] = -np.inf

        idx = np.argpartition(scores, -min(n, len(scores)))[-min(n, len(scores)) :]
        sorted_idx = idx[np.argsort(scores[idx])[::-1]]

        return sorted_idx, scores[sorted_idx]

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    @property
    def item_factors(self) -> Any:
        """Item factor matrix (n_items, factors). Returns V_iu (MF item embeddings)."""
        self._check_fitted()
        return self._viu

    @classmethod
    def from_transactions(
        cls,
        data: Any,
        user_col: str = "user_id",
        item_col: str = "item_id",
        timestamp_col: str | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> FPMC:
        """Instantiate from transaction DataFrame."""
        import numpy as np

        users = data[user_col].values
        items = data[item_col].values

        model = cls(verbose=verbose, **kwargs)

        timestamps = None
        if model.time_aware:
            if not timestamp_col:
                raise ValueError("timestamp_col is required when time_aware=True")
            ts_series = data[timestamp_col]
            if not np.issubdtype(ts_series.dtype, np.number):
                import pandas as pd

                ts_series = pd.to_datetime(ts_series).astype("int64") // 10**9
            timestamps = ts_series.values

        # Only map items (users are implicit 0...N based on unique order or sequence grouping)
        # Note: In standard usage FPMC sequences are grouped beforehand. Usually this class
        # requires passing sequential data. To properly group for FPMC, we need to sort and group.

        # Sort values
        if timestamps is not None:
            sort_idx = np.lexsort((timestamps, users))
        else:
            sort_idx = np.argsort(users)

        sorted_users = users[sort_idx]
        sorted_items = items[sort_idx]
        sorted_ts = timestamps[sort_idx] if timestamps is not None else None

        unique_items = np.unique(sorted_items)
        item_map = {it: i for i, it in enumerate(unique_items)}

        # We need sequential maps for unique users
        unique_users = np.unique(sorted_users)
        user_map = {u: i for i, u in enumerate(unique_users)}

        sequences: dict[int, list[int]] = {i: [] for i in range(len(unique_users))}
        ts_dict: dict[int, list[int]] = {i: [] for i in range(len(unique_users))}

        if timestamps is not None and sorted_ts is not None:
            for u, it, ts in zip(sorted_users, sorted_items, sorted_ts, strict=False):
                u_idx = user_map[u]
                sequences[u_idx].append(item_map[it])
                ts_dict[u_idx].append(int(ts))
            model._pending_timestamps = list(ts_dict.values())
        else:
            for u, it in zip(sorted_users, sorted_items, strict=False):
                sequences[user_map[u]].append(item_map[it])
            model._pending_timestamps = None

        model._pending_sequences = list(sequences.values())
        return model
