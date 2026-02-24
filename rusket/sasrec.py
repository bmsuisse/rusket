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
    verbose : int
        Print epoch progress.
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
        verbose: int = 0,
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
        self.verbose = verbose

        self._item_emb: np.ndarray | None = None
        self._item_map: dict[int, int] = {}
        self._rev_item_map: dict[int, int] = {}

        self._user_sequences: dict[int, list[int]] = {}
        self._n_items: int = 0
        self.fitted: bool = False
        self._pending_sequences: list[list[int]] | None = None

    def __repr__(self) -> str:
        return (
            f"SASRec(factors={self.factors}, n_layers={self.n_layers}, "
            f"max_seq={self.max_seq}, iterations={self.iterations})"
        )

    def fit(self, sequences: list[list[int]] | None = None) -> SASRec:
        """Train SASRec on integer-encoded sequences (0-indexed item IDs).

        Parameters
        ----------
        sequences : list of list of int, optional
            List of per-user interaction histories (item IDs).
            If None, uses data prepared by ``from_transactions()``.

        Returns
        -------
        SASRec
            The fitted model.
        """
        if sequences is None:
            sequences = getattr(self, "_pending_sequences", None)
            if sequences is None:
                raise ValueError("No sequences provided. Pass sequences or use from_transactions() first.")

        if self.fitted:
            raise RuntimeError("Model is already fitted.")

        n_items = max(max(s) for s in sequences if s) + 1 if any(sequences) else 0
        seed = self.seed if self.seed is not None else int(np.random.randint(1 << 31))

        self._item_emb = sasrec_fit(
            sequences,
            n_items,
            self.factors,
            self.n_layers,
            self.max_seq,
            float(self.learning_rate),
            float(self.lambda_),
            self.iterations,
            int(seed),
            bool(self.verbose),
        )
        self._n_items = n_items

        self._user_sequences = dict(enumerate(sequences))
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

        unique_items = np.unique(items)
        item_map = {it: i + 1 for i, it in enumerate(unique_items)}  # 1-indexed; 0 = pad
        rev_item_map = {i: it for it, i in item_map.items()}

        model = cls(verbose=verbose, **kwargs)
        model._item_map = item_map
        model._rev_item_map = rev_item_map

        sequences: dict[int, list[int]] = {}
        for u, it in zip(users, items, strict=False):
            sequences.setdefault(u, []).append(model._item_map[it])

        model._pending_sequences = list(sequences.values())
        return model

    def recommend_items(
        self, user_id: int | list[int], n: int = 10, exclude_seen: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-N items for a user or an ad-hoc sequence.

        Parameters
        ----------
        user_id : int or list[int]
            The ID of the user (implicitly 0 to len(sequences)-1 from fit),
            or a list of items representing an ad-hoc sequence.
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
        else:
            user_sequence = user_id

        seq = [i for i in user_sequence if i > 0]
        if not seq:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        # Build positional encoding placeholder (simplified: just use first row of pos_emb)
        d = self.factors
        seq_cut = seq[-self.max_seq :]
        # Compute sequence representation via inline embedding sum (simplified)
        seq_repr = np.zeros(d, dtype=np.float32)
        for item in seq_cut:
            if 0 < item <= self._n_items:
                seq_repr += self._item_emb[item]
        seq_repr /= max(len(seq_cut), 1)

        # Score all items
        scores = self._item_emb[1:] @ seq_repr

        if exclude_seen:
            exclude_set: set[int] = set(user_sequence)
            for exc in exclude_set:
                if 1 <= exc <= len(scores):
                    scores[exc - 1] = -np.inf

        top_idx = np.argsort(scores)[::-1][:n]
        original_ids = np.array([self._rev_item_map.get(i + 1, i + 1) for i in top_idx], dtype=np.int64)
        return original_ids, scores[top_idx]

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
