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
    """

    def __init__(
        self,
        factors: int = 64,
        learning_rate: float = 0.05,
        regularization: float = 0.01,
        iterations: int = 150,
        seed: int = 42,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(data=None, **kwargs)
        self.factors = factors
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.iterations = iterations
        self.seed = seed
        self.verbose = verbose

        # Vu, Viu, Vil, Vli
        self._vu: Any = None
        self._viu: Any = None
        self._vil: Any = None
        self._vli: Any = None

        import numpy as np

        # Last items for users to make predictions
        self._user_last_items: dict[int, int] = {}
        self._user_seen_items: dict[int, np.ndarray] = {}

        self.fitted: bool = False

    def __repr__(self) -> str:
        return (
            f"FPMC(factors={self.factors}, learning_rate={self.learning_rate}, "
            f"regularization={self.regularization}, iterations={self.iterations})"
        )

    def fit(self, sequences: list[list[int]], n_items: int | None = None) -> FPMC:
        """Fit the FPMC model to a list of sequential interactions.

        Parameters
        ----------
        sequences : list of list of int
            List of item sequences, where each sequence belongs to a unique user.
            Users are assigned IDs from 0 to len(sequences)-1.
        n_items : int | None
            Maximum number of items. If None, it is inferred from data.
        """
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

        for u, seq in enumerate(sequences):
            if seq:
                self._user_last_items[u] = seq[-1]
                self._user_seen_items[u] = np.unique(seq)
                indices.extend(seq)
            indptr[u + 1] = len(indices)

        indices_arr = np.array(indices, dtype=np.int32)

        self._vu, self._viu, self._vil, self._vli = _rust.fpmc_fit(  # type: ignore[attr-defined]
            indptr,
            indices_arr,
            self._n_users,
            self._n_items,
            self.factors,
            self.learning_rate,
            self.regularization,
            self.iterations,
            self.seed,
            self.verbose,
        )

        self.fitted = True
        return self

    def recommend_items(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> tuple[Any, Any]:
        """Top-N sequential items for a user."""
        import numpy as np

        self._check_fitted()
        if user_id < 0 or user_id >= self._n_users:
            raise ValueError(f"user_id {user_id} is out of bounds for model with {self._n_users} users.")

        prev_item = self._user_last_items.get(user_id, -1)

        # Calculate scores: x_{u,l,i} = dot(V_u, V_iu) + dot(V_il, V_li) if prev_item >= 0 else dot(V_u, V_iu)
        v_u = self._vu[user_id]

        # MF term
        scores = np.dot(self._viu, v_u)

        # Markov Chain term
        if prev_item >= 0:
            v_l = self._vli[prev_item]
            scores += np.dot(self._vil, v_l)

        # Top-N selection
        if exclude_seen:
            seen_items = self._user_seen_items.get(user_id)
            if seen_items is not None:
                scores[seen_items] = -np.inf

        idx = np.argpartition(scores, -n)[-n:]
        sorted_idx = idx[np.argsort(scores[idx])[::-1]]

        return sorted_idx, scores[sorted_idx]

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
