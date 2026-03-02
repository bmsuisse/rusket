"""BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from rusket._rusket import bert4rec_fit, bert4rec_predict  # type: ignore[attr-defined]

from .model import SequentialRecommender

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


class BERT4Rec(SequentialRecommender):
    """BERT4Rec: Sequential Recommendation with Bidirectional Attention.

    Unlike SASRec, which uses causal (left-to-right) attention, BERT4Rec uses
    bidirectional attention to learn sequence representations, trained via a Cloze
    (Masked Item Prediction) objective.

    Parameters
    ----------
    factors : int
        Embedding dimensionality (latent factors).
    n_layers : int
        Number of transformer blocks.
    max_seq : int
        Maximum sequence length to consider.
    mask_prob : float, default=0.2
        Probability of masking an item during the Cloze task training.
    learning_rate : float
        Adam learning rate.
    lambda\\_ : float
        L2 regularization coefficient.
    iterations : int
        Number of training epochs.
    seed : int or None
        Seed for reproducible training.
    verbose : int
        Print training progress.
    use_gpu : bool
        If True, use GPU acceleration (CuPy or PyTorch) for recommendation.
        Falls back to CPU if no GPU backend found. Default False.
    """

    def __init__(
        self,
        factors: int = 64,
        n_layers: int = 2,
        max_seq: int = 50,
        mask_prob: float = 0.2,
        learning_rate: float = 1e-3,
        lambda_: float = 1e-4,
        iterations: int = 20,
        seed: int | None = None,
        verbose: int = 0,
        use_gpu: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.factors = factors
        self.n_layers = n_layers
        self.max_seq = max_seq
        self.mask_prob = mask_prob
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.iterations = iterations
        self.seed = seed
        self.verbose = verbose
        self.use_gpu = use_gpu

        self._item_factors: np.ndarray | None = None
        self._item_map: dict[Any, int] = {}
        self._rev_item_map: dict[int, Any] = {}
        self._n_items: int = 0
        self.fitted: bool = False

        self._pending_df: Any = None
        self._pending_user_col: str | None = None
        self._pending_item_col: str | None = None

    def __repr__(self) -> str:
        return (
            f"BERT4Rec(factors={self.factors}, n_layers={self.n_layers}, "
            f"max_seq={self.max_seq}, mask_prob={self.mask_prob}, "
            f"iterations={self.iterations})"
        )

    @classmethod
    def from_transactions(
        cls,
        data: pd.DataFrame | pl.DataFrame | Any,
        transaction_col: str | None = None,
        item_col: str | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> BERT4Rec:
        """Initialize the BERT4Rec model from a long-format DataFrame.

        Prepares the data but does **not** fit the model.
        Call ``.fit()`` explicitly to train.

        Parameters
        ----------
        data : pd.DataFrame | pl.DataFrame | pyspark.sql.DataFrame
            Event log containing user IDs and item IDs. Should be sorted by time
            if sequential order matters (which it does for BERT4Rec).
        transaction_col : str, optional
            Column name identifying the user ID (aliases ``user_col``).
        item_col : str, optional
            Column name identifying the item ID.
        verbose : int, default=0
            Verbosity level.
        **kwargs
            Model hyperparameters (e.g., ``factors``, ``n_layers``).

        Returns
        -------
        BERT4Rec
            The configured (unfitted) model.
        """
        user_col = kwargs.pop("user_col", transaction_col)
        model = cls(verbose=verbose, **kwargs)
        model._pending_df = data
        model._pending_user_col = user_col
        model._pending_item_col = item_col
        return model

    def fit(self, sequences: Sequence[Sequence[Any]] | None = None) -> BERT4Rec:
        """Fit the model to a list of item sequences.

        Parameters
        ----------
        sequences : list of list of items, optional
            Each inner list represents a user's chronological interaction history.
            If None, uses data prepared by ``from_transactions()``.

        Returns
        -------
        BERT4Rec
            The fitted model.
        """
        pending_df = self._pending_df
        if sequences is None and pending_df is not None:
            self._fit_from_df(pending_df, self._pending_user_col, self._pending_item_col)
            self._pending_df = None
            return self

        if sequences is None:
            raise ValueError("No sequences provided. Pass a list of lists or use from_transactions() first.")

        if self.fitted:
            raise RuntimeError("Model is already fitted. Create a new instance to refit.")

        # Build vocabulary dynamically
        unique_items = set()
        for seq in sequences:
            unique_items.update(seq)

        sorted_items = sorted(unique_items)
        # 1-based indexing! 0 is padding, len(sorted_items) + 1 is [MASK]
        self._item_map = {it: i + 1 for i, it in enumerate(sorted_items)}
        self._rev_item_map = {i + 1: it for i, it in enumerate(sorted_items)}
        self._n_items = len(sorted_items)

        encoded_seqs = [[self._item_map[it] for it in seq] for seq in sequences]

        seed = self.seed if self.seed is not None else int(np.random.randint(1 << 31))

        encoded_seqs_rust = [list(map(int, seq)) for seq in encoded_seqs]

        ei = bert4rec_fit(
            encoded_seqs_rust,
            self._n_items,
            self.factors,
            self.n_layers,
            self.max_seq,
            float(self.mask_prob),
            float(self.learning_rate),
            float(self.lambda_),
            self.iterations,
            int(seed),
            bool(self.verbose),
        )

        self._item_factors = np.asarray(ei, dtype=np.float32)
        self.item_names = [str(it) for it in sorted_items]
        self.fitted = True

        return self

    def _fit_from_df(
        self,
        data: pd.DataFrame | pl.DataFrame | Any,
        user_col: str | None,
        item_col: str | None,
    ) -> None:
        """Internal: fit from a DataFrame with user/item columns."""
        if hasattr(data, "to_pandas"):
            data = data.to_pandas()

        user_col = user_col or str(data.columns[0])
        item_col = item_col or str(data.columns[1])

        grouped = data.groupby(user_col, sort=False)[item_col].apply(list).tolist()
        self.fit(grouped)

    def recommend_items(
        self, sequence: Sequence[Any], n: int = 10, exclude_seen: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-N next-item recommendations for a sequence.

        Parameters
        ----------
        sequence : Sequence[Any]
            The user's recent interaction history.
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
        assert self._item_factors is not None

        encoded_seq = [self._item_map[it] for it in sequence if it in self._item_map]
        encoded_seq_rust = [int(i) for i in encoded_seq]

        # Use batch predict, even for a single user
        ids, scores = bert4rec_predict(
            self._item_factors,
            [encoded_seq_rust],
            self.max_seq,
            bool(exclude_seen),
            self._n_items,
            int(n),
        )

        top_ids = ids[0]
        top_scores = scores[0]

        # Filter out 0s (padding)
        valid_mask = top_ids > 0
        valid_ids = top_ids[valid_mask]
        valid_scores = top_scores[valid_mask]

        original_ids = np.array([self._rev_item_map[i] for i in valid_ids], dtype=object)
        return original_ids, valid_scores

    @property
    def item_factors(self) -> Any:
        """Item factor matrix (n_items + 2, factors). Row 0 is padding, row n_items+1 is [MASK]."""
        self._check_fitted()
        return self._item_factors

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
