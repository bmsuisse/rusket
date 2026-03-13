from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

from .. import _rusket as _rust  # type: ignore
from ..model import ImplicitRecommender


class RuleBasedRecommender(ImplicitRecommender):
    """
    A recommender that injects explicit business rules or curated lists.

    Rules can be defined as a mapping from item to a list of associated items,
    or as a pandas DataFrame containing antecedent, consequent, and optionally score.

    This class integrates perfectly with ``HybridRecommender``, allowing you to
    blend business logic with algorithmic models like ALS or BPR.
    """

    def __init__(
        self,
        rules: dict[Any, list[Any]] | pd.DataFrame,
        antecedent_col: str = "antecedent",
        consequent_col: str = "consequent",
        score_col: str | None = "score",
        default_score: float = 1.0,
        verbose: int = 0,
        **kwargs: Any,
    ):
        super().__init__()
        self.rules_input = rules
        self.antecedent_col = antecedent_col
        self.consequent_col = consequent_col
        self.score_col = score_col
        self.default_score = default_score
        self.verbose = verbose

        self.w_indptr: np.ndarray | None = None
        self.w_indices: np.ndarray | None = None
        self.w_data: np.ndarray | None = None
        self.fitted: bool = False

    def __repr__(self) -> str:
        return f"RuleBasedRecommender(rules_count={len(self.rules_input)})"

    def fit(self, interactions: Any = None) -> RuleBasedRecommender:
        """
        Builds the item-to-item recommendation matrix based on the provided explicit rules.

        Parameters
        ----------
        interactions : scipy.sparse.csr_matrix, optional
            A sparse matrix of shape (n_users, n_items). If None, uses the
            matrix prepared by ``from_transactions()``.

        Returns
        -------
        RuleBasedRecommender
            The fitted model.
        """
        if interactions is None:
            interactions = getattr(self, "_prepared_interactions", None)
            if interactions is None:
                raise ValueError("No interactions provided. Pass a matrix or use from_transactions() first.")

        import scipy.sparse as sp

        if not sp.isspmatrix_csr(interactions):
            interactions = interactions.tocsr()

        n_users, n_items = interactions.shape
        self._n_users = n_users
        self._n_items = n_items

        # Store fit interactions to omit seen items in recommend_items
        self._fit_indptr = interactions.indptr
        self._fit_indices = interactions.indices
        self._fit_data = interactions.data

        # We must map the external item labels to the internal integer IDs (0 to n_items-1)
        item_labels = getattr(self, "_item_labels", None)
        if item_labels is None:
            # If no labels, assume rules use internal integer IDs directly
            item_map: dict[Any, int] | None = None
        else:
            item_map = {label: idx for idx, label in enumerate(item_labels)}

        # Extract rules into (source_id, target_id, score)
        sources: list[int] = []
        targets: list[int] = []
        scores: list[float] = []

        from rusket._dependencies import import_optional_dependency

        _pd = import_optional_dependency("pandas")

        if isinstance(self.rules_input, dict):
            # Format: {item_A: [item_B, item_C], ...}
            for src, tgt_list in self.rules_input.items():
                src_id = (
                    item_map[src] if item_map is not None and src in item_map else (src if item_map is None else None)
                )
                if src_id is None or (isinstance(src_id, int) and (src_id < 0 or src_id >= n_items)):
                    continue

                for tgt in tgt_list:
                    tgt_id = (
                        item_map[tgt]
                        if item_map is not None and tgt in item_map
                        else (tgt if item_map is None else None)
                    )
                    if tgt_id is None or (isinstance(tgt_id, int) and (tgt_id < 0 or tgt_id >= n_items)):
                        continue

                    sources.append(int(src_id))
                    targets.append(int(tgt_id))
                    scores.append(self.default_score)

        elif isinstance(self.rules_input, _pd.DataFrame):
            # Format: DataFrame with antecedent, consequent, and optional score
            for row in self.rules_input.itertuples(index=False):  # type: ignore
                row_dict = row._asdict()
                src = row_dict[self.antecedent_col]
                tgt = row_dict[self.consequent_col]
                score = (
                    row_dict[self.score_col]
                    if self.score_col is not None and self.score_col in row_dict
                    else self.default_score
                )

                src_id = (
                    item_map[src] if item_map is not None and src in item_map else (src if item_map is None else None)
                )
                if src_id is None or (isinstance(src_id, int) and (src_id < 0 or src_id >= n_items)):
                    continue

                tgt_id = (
                    item_map[tgt] if item_map is not None and tgt in item_map else (tgt if item_map is None else None)
                )
                if tgt_id is None or (isinstance(tgt_id, int) and (tgt_id < 0 or tgt_id >= n_items)):
                    continue

                sources.append(int(src_id))
                targets.append(int(tgt_id))
                scores.append(float(score))
        else:
            raise TypeError("rules must be a dict or a pandas DataFrame.")

        # If duplicate rules exist, we sum their scores or take max? We'll construct a CSR matrix which sums duplicates.
        import numpy as np

        if len(sources) > 0:
            W = sp.csr_matrix((scores, (sources, targets)), shape=(n_items, n_items))
            W.eliminate_zeros()
        else:
            # Empty rules matrix
            W = sp.csr_matrix((n_items, n_items))

        # We don't prune to top-K here because rules are usually explicit and sparse enough.
        # But we could optionally. For now, just store W.
        self.w_indptr = W.indptr.astype(np.int64)
        self.w_indices = W.indices.astype(np.int32)
        self.w_data = W.data.astype(np.float32)

        self.fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")

    def recommend_items(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> tuple[Any, Any]:
        """Top-N items for a user based on explicit rules.

        Parameters
        ----------
        user_id : int
            The user ID to generate recommendations for.
        n : int, default=10
            Number of items to return.
        exclude_seen : bool, default=True
            Whether to exclude items the user has already interacted with.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(item_ids, scores)`` sorted by descending score.
        """
        self._check_fitted()

        import numpy as np

        if user_id < 0 or user_id >= self._n_users:
            raise ValueError(f"user_id {user_id} is out of bounds for model with {self._n_users} users.")

        if (
            exclude_seen
            and getattr(self, "_fit_indptr", None) is not None
            and getattr(self, "_fit_indices", None) is not None
        ):
            exc_indptr = self._fit_indptr
            exc_indices = self._fit_indices
        else:
            exc_indptr = np.zeros(self._n_users + 1, dtype=np.int64)
            exc_indices = np.array([], dtype=np.int32)

        if getattr(self, "_fit_data", None) is None:
            user_data = np.ones_like(exc_indices, dtype=np.float32)
        else:
            user_data = self._fit_data

        ids, scores = _rust.itemknn_recommend_items(  # type: ignore[attr-defined]
            self.w_indptr,
            self.w_indices,
            self.w_data,
            getattr(self, "_fit_indptr", np.zeros(self._n_users + 1, dtype=np.int64)).astype(np.int64),
            getattr(self, "_fit_indices", np.array([], dtype=np.int32)).astype(np.int32),
            user_data.astype(np.float32),
            user_id,
            n,
            exc_indptr.astype(np.int64),
            exc_indices.astype(np.int32),
            self._n_items,
        )
        return np.asarray(ids), np.asarray(scores)
