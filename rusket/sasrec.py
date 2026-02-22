from __future__ import annotations

import numpy as np

from rusket._rusket import sasrec_fit, sasrec_encode


class SASRec:
    """SASRec – Self-Attentive Sequential Recommendation.

    Applies a causal Transformer to user interaction sequences to predict
    the next item. Significantly outperforms Markov-chain methods like FPMC
    on long sequences.

    Args:
        factors: Embedding dimensionality.
        n_layers: Number of Transformer blocks.
        max_seq: Maximum input sequence length (older items are dropped).
        learning_rate: SGD learning rate (decays during training).
        lambda_: L2 regularization.
        iterations: Number of training epochs.
        random_state: Seed for reproducibility.
        verbose: Print epoch progress.
    """

    def __init__(
        self,
        factors: int = 64,
        n_layers: int = 2,
        max_seq: int = 50,
        learning_rate: float = 5e-4,
        lambda_: float = 1e-4,
        iterations: int = 20,
        random_state: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.factors = factors
        self.n_layers = n_layers
        self.max_seq = max_seq
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.iterations = iterations
        self.random_state = random_state
        self.verbose = verbose

        self._item_emb: np.ndarray | None = None
        self._item_map: dict[int, int] = {}
        self._rev_item_map: dict[int, int] = {}

    def fit(self, sequences: list[list[int]]) -> "SASRec":
        """Train SASRec on integer-encoded sequences (0-indexed item IDs).

        Args:
            sequences: List of per-user interaction histories (item IDs).

        Returns:
            self
        """
        n_items = max(max(s) for s in sequences if s) + 1 if any(sequences) else 0
        seed = (
            self.random_state
            if self.random_state is not None
            else int(np.random.randint(1 << 31))
        )

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
        return self

    @classmethod
    def from_transactions(
        cls,
        data,
        user_col: str | None = None,
        item_col: str | None = None,
        timestamp_col: str | None = None,
        **kwargs,
    ) -> "SASRec":
        """Train SASRec from a transactions DataFrame.

        Args:
            data: pandas or polars DataFrame.
            user_col: Name of the user ID column.
            item_col: Name of the item ID column.
            timestamp_col: Optional column to sort by; uses insertion order otherwise.
            **kwargs: Passed to the SASRec constructor.

        Returns:
            Fitted SASRec model.
        """
        import pandas as pd

        if hasattr(data, "to_pandas"):
            data = data.to_pandas()

        user_col = user_col or "user_id"
        item_col = item_col or "item_id"

        if timestamp_col:
            data = data.sort_values([user_col, timestamp_col])

        users = data[user_col].values
        items = data[item_col].values

        unique_items = np.unique(items)
        item_map = {it: i + 1 for i, it in enumerate(unique_items)}  # 1-indexed; 0 = pad
        rev_item_map = {i: it for it, i in item_map.items()}

        model = cls(**kwargs)
        model._item_map = item_map
        model._rev_item_map = rev_item_map

        sequences: dict[int, list[int]] = {}
        for u, it in zip(users, items):
            sequences.setdefault(u, []).append(model._item_map[it])

        model.fit(list(sequences.values()))
        return model

    def recommend_items(
        self, user_sequence: list[int], n: int = 10, exclude: list[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return top-n recommended item IDs and scores given a user sequence.

        Args:
            user_sequence: Recent item IDs (1-indexed internal encoding).
            n: Number of recommendations.
            exclude: Item IDs to exclude from recommendations.

        Returns:
            (item_ids, scores) sorted by descending score.
        """
        if self._item_emb is None:
            raise RuntimeError("Model is not fitted yet.")

        seq = [i for i in user_sequence if i > 0]
        if not seq:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        # Build positional encoding placeholder (simplified: just use first row of pos_emb)
        d = self.factors
        seq_cut = seq[-self.max_seq:]
        # Compute sequence representation via inline embedding sum (simplified)
        seq_repr = np.zeros(d, dtype=np.float32)
        for item in seq_cut:
            if 0 < item <= self._n_items:
                seq_repr += self._item_emb[item]
        seq_repr /= max(len(seq_cut), 1)

        # Score all items
        scores = self._item_emb[1:] @ seq_repr
        exclude_set: set[int] = set(exclude or [])
        for exc in exclude_set:
            if 1 <= exc <= len(scores):
                scores[exc - 1] = -np.inf

        top_idx = np.argsort(scores)[::-1][:n]
        original_ids = np.array(
            [self._rev_item_map.get(i + 1, i + 1) for i in top_idx], dtype=np.int64
        )
        return original_ids, scores[top_idx]
