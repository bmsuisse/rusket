"""SASRec – Self-Attentive Sequential Recommendation."""

from __future__ import annotations

from typing import Any

import numpy as np

from rusket._rusket import sasrec_fit  # type: ignore[attr-defined]

from .model import SequentialRecommender


class SASRec(SequentialRecommender):
    """SASRec – Self-Attentive Sequential Recommendation.

    Applies a causal Transformer to user interaction sequences to predict
    the next item. Significantly outperforms Markov-chain methods like FPMC
    on long sequences.

    Parameters
    ----------
    factors : int
        Embedding dimensionality.
    n_layers : int
        Number of Transformer blocks.
    max_seq : int
        Maximum input sequence length (older items are dropped).
    learning_rate : float
        SGD learning rate (decays during training).
    lambda\\_ : float
        L2 regularization.
    iterations : int
        Number of training epochs.
    seed : int or None
        Seed for reproducibility.
    time_aware : bool
        If true, incorporates timestamp deltas into sequential modeling.
    max_time_steps : int
        Maximum number of time bins (e.g. days) to consider for time-awareness.
    verbose : int
        Print epoch progress.
    use_gpu : bool
        If True, use GPU acceleration (CuPy or PyTorch) for recommendation.
        Falls back to CPU if no GPU backend found. Default False.
    """

    def __init__(
        self,
        factors: int = 64,
        n_layers: int = 2,
        max_seq: int = 50,
        learning_rate: float = 5e-4,
        lambda_: float = 1e-4,
        iterations: int = 20,
        seed: int | None = None,
        time_aware: bool = False,
        max_time_steps: int = 256,
        verbose: int = 0,
        use_gpu: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.factors = factors
        self.n_layers = n_layers
        self.max_seq = max_seq
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.iterations = iterations
        self.seed = seed
        self.time_aware = time_aware
        self.max_time_steps = max_time_steps
        self.verbose = verbose
        self.use_gpu = use_gpu

        self._item_emb: np.ndarray | None = None
        self._time_emb: np.ndarray | None = None
        self._item_map: dict[int, int] = {}
        self._rev_item_map: dict[int, int] = {}

        self._user_sequences: dict[int, list[int]] = {}
        self._user_timestamps: dict[int, list[int]] | None = None if not time_aware else {}
        self._n_items: int = 0
        self.fitted: bool = False
        self._pending_sequences: list[list[int]] | None = None
        self._pending_timestamps: list[list[int]] | None = None

    def __repr__(self) -> str:
        return (
            f"SASRec(factors={self.factors}, n_layers={self.n_layers}, "
            f"max_seq={self.max_seq}, iterations={self.iterations})"
        )

    def fit(self, sequences: list[list[int]] | None = None, timestamps: list[list[int]] | None = None) -> SASRec:
        if sequences is None:
            sequences = getattr(self, "_pending_sequences", None)
            timestamps = getattr(self, "_pending_timestamps", None)
            if sequences is None:
                raise ValueError("No sequences provided. Pass sequences or use from_transactions() first.")

        if self.fitted:
            raise RuntimeError("Model is already fitted.")

        n_items = max(max(s) for s in sequences if s) + 1 if any(sequences) else 0
        seed = self.seed if self.seed is not None else int(np.random.randint(1 << 31))

        self._item_emb, time_emb_opt = sasrec_fit(
            sequences,
            timestamps if self.time_aware else None,
            n_items,
            self.factors,
            self.n_layers,
            self.max_seq,
            self.max_time_steps if self.time_aware else 0,
            float(self.learning_rate),
            float(self.lambda_),
            self.iterations,
            int(seed),
            bool(self.verbose),
        )
        self._n_items = n_items
        self._time_emb = time_emb_opt

        self._user_sequences = dict(enumerate(sequences))
        if timestamps is not None:
            self._user_timestamps = dict(enumerate(timestamps))

        self.fitted = True

        return self

    @classmethod
    def from_transactions(
        cls,
        data: Any,
        transaction_col: str | None = None,
        item_col: str | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> SASRec:
        """Prepare SASRec from a transactions DataFrame.

        Prepares sequences but does **not** fit the model.
        Call ``.fit()`` explicitly to train.

        Parameters
        ----------
        data : pd.DataFrame | pl.DataFrame | pyspark.sql.DataFrame
            Event log containing user IDs, item IDs, and optionally timestamps.
        transaction_col : str, optional
            Column name identifying the user ID (aliases ``user_col``).
        item_col : str, optional
            Column name identifying the item ID.
        timestamp_col : str, optional
            Column name identifying the timestamp (required if time_aware is True).
        verbose : int, default=0
            Verbosity level.
        **kwargs
            Model hyperparameters (e.g., ``factors``, ``n_layers``).
            Can also include ``user_col`` and ``timestamp_col``.

        Returns
        -------
        SASRec
            The configured (unfitted) model.
        """
        user_col = kwargs.pop("user_col", transaction_col)
        timestamp_col = kwargs.pop("timestamp_col", None)

        if hasattr(data, "to_pandas"):
            data = data.to_pandas()

        user_col = user_col or str(data.columns[0])
        item_col = item_col or str(data.columns[1])

        if timestamp_col:
            data = data.sort_values([user_col, timestamp_col])

        users = data[user_col].values
        items = data[item_col].values

        model = cls(verbose=verbose, **kwargs)

        timestamps = None
        # Convert timestamp to unix seconds if time_aware is enabled
        if model.time_aware:
            if not timestamp_col:
                raise ValueError("timestamp_col is required when time_aware=True")
            ts_series = data[timestamp_col]
            if not np.issubdtype(ts_series.dtype, np.number):
                # Ensure it's passed as seconds since epoch
                import pandas as pd

                ts_series = pd.to_datetime(ts_series).astype("int64") // 10**9
            timestamps = ts_series.values

        unique_items = np.unique(items)
        item_map = {it: i + 1 for i, it in enumerate(unique_items)}  # 1-indexed; 0 = pad
        rev_item_map = {i: it for it, i in item_map.items()}

        model = cls(verbose=verbose, **kwargs)
        model._item_map = item_map
        model._rev_item_map = rev_item_map

        sequences: dict[Any, list[int]] = {}
        ts_dict: dict[Any, list[int]] = {}

        if model.time_aware and timestamps is not None:
            for u, it, ts in zip(users, items, timestamps, strict=False):
                sequences.setdefault(u, []).append(model._item_map[it])
                ts_dict.setdefault(u, []).append(int(ts))
            model._pending_timestamps = list(ts_dict.values())
        else:
            for u, it in zip(users, items, strict=False):
                sequences.setdefault(u, []).append(model._item_map[it])
            model._pending_timestamps = None

        model._pending_sequences = list(sequences.values())
        return model

    def recommend_items(
        self, user_id: int | list[int], timestamps: list[int] | None = None, n: int = 10, exclude_seen: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-N items for a user or an ad-hoc sequence.

        Parameters
        ----------
        user_id : int or list[int]
            The ID of the user (implicitly 0 to len(sequences)-1 from fit),
            or a list of items representing an ad-hoc sequence.
        timestamps : list[int], optional
            Corresponding unix timestamps if user_id is a list of items and time_aware=True.
        n : int, default=10
            Number of recommendations.
        exclude_seen : bool, default=True
            Whether to exclude items the user has already interacted with.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(item_ids, scores)`` sorted by descending score.
        """
        self._check_fitted()
        assert self._item_emb is not None

        if isinstance(user_id, int):
            user_sequence = self._user_sequences.get(user_id, [])
            user_ts = self._user_timestamps.get(user_id, []) if self._user_timestamps else []
        else:
            user_sequence = user_id
            user_ts = timestamps or []

        # Filter out 0s if they are padding
        valid_indices = [idx for idx, i in enumerate(user_sequence) if i > 0]
        seq = [user_sequence[idx] for idx in valid_indices]
        ts_seq = [user_ts[idx] for idx in valid_indices] if user_ts else []

        if not seq:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        sequences_param = [seq]
        timestamps_param = [ts_seq] if ts_seq else None

        from rusket._rusket import sasrec_predict  # type: ignore[attr-defined]

        out_ids, out_scores = sasrec_predict(
            self._item_emb,
            self._time_emb,
            sequences_param,
            timestamps_param,
            self.max_seq,
            self.max_time_steps if self.time_aware else 0,
            exclude_seen,
            self._n_items,
            n,
        )

        original_ids = np.array([self._rev_item_map.get(int(i), int(i)) for i in out_ids[0]], dtype=np.int64)
        return original_ids, out_scores[0]

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
