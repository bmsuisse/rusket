"""Content-based recommender using TF-IDF and cosine similarity."""

from __future__ import annotations

from typing import Any

import numpy as np

from .model import BaseModel


class ContentBased(BaseModel):
    """Content-based recommender using TF-IDF vectorization and cosine similarity.

    Recommends items similar to a given item based on textual features
    (descriptions, tags, genres, etc.).

    Parameters
    ----------
    max_features : int, default=5000
        Maximum number of TF-IDF features to extract.
    ngram_range : tuple[int, int], default=(1, 2)
        Range of n-grams for TF-IDF vectorisation.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        **kwargs: Any,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range

        self._item_ids: list[Any] | None = None
        self._item_id_to_idx: dict[Any, int] | None = None
        self._tfidf_matrix: Any = None
        self._similarity_matrix: np.ndarray | None = None
        self._texts: list[str] | None = None
        self.fitted: bool = False

    def __repr__(self) -> str:
        return f"ContentBased(max_features={self.max_features}, ngram_range={self.ngram_range})"

    # ── construction ──────────────────────────────────────────────────

    @classmethod
    def from_transactions(
        cls,
        data: Any,
        transaction_col: str | None = None,
        item_col: str | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> ContentBased:
        """Not applicable for content-based filtering.

        Use :meth:`from_dataframe` instead, which requires item metadata
        with text columns.
        """
        raise NotImplementedError(
            "ContentBased does not support from_transactions(). "
            "Use ContentBased.from_dataframe(df, item_col, text_col) instead."
        )

    @classmethod
    def from_dataframe(
        cls,
        df: Any,
        item_col: str,
        text_col: str,
        **kwargs: Any,
    ) -> ContentBased:
        """Build a content-based recommender from item metadata.

        Parameters
        ----------
        df : pd.DataFrame | pl.DataFrame
            A DataFrame containing item identifiers and text descriptions.
        item_col : str
            Column with item IDs.
        text_col : str
            Column with text data (descriptions, tags, genres, etc.).
        **kwargs
            Additional keyword arguments forwarded to the constructor
            (e.g. ``max_features``, ``ngram_range``).

        Returns
        -------
        ContentBased
            A ready-to-fit model instance.
        """
        from rusket._dependencies import import_optional_dependency

        pd = import_optional_dependency("pandas")

        # Convert Polars → Pandas if needed
        if not isinstance(df, pd.DataFrame):
            if hasattr(df, "to_pandas"):
                df = df.to_pandas()
            else:
                raise TypeError(f"Expected a Pandas or Polars DataFrame, got {type(df)}")

        model = cls(**kwargs)
        item_ids = df[item_col].tolist()
        model._item_ids = item_ids
        model._item_id_to_idx = {item: idx for idx, item in enumerate(item_ids)}
        model._texts = df[text_col].fillna("").astype(str).tolist()
        return model

    # ── fit ────────────────────────────────────────────────────────────

    def fit(self) -> ContentBased:  # type: ignore[override]
        """Compute TF-IDF vectors and the pairwise cosine similarity matrix.

        Returns
        -------
        ContentBased
            The fitted model.
        """
        if self._texts is None:
            raise ValueError("No text data. Use ContentBased.from_dataframe() first.")

        if self.fitted:
            raise RuntimeError("Model is already fitted. Create a new instance to refit.")

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english",
        )
        self._tfidf_matrix = vectorizer.fit_transform(self._texts)
        sim = cosine_similarity(self._tfidf_matrix).astype(np.float32)

        # Zero the diagonal so an item doesn't recommend itself
        np.fill_diagonal(sim, 0.0)
        self._similarity_matrix = sim

        self.fitted = True
        return self

    # ── recommend ──────────────────────────────────────────────────────

    def recommend_similar(
        self,
        item: Any,
        n: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find the *n* most similar items to a given item.

        Parameters
        ----------
        item : Any
            Item ID (as it appeared in ``item_col`` of the source DataFrame).
        n : int, default=10
            Number of similar items to return.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(item_ids, similarity_scores)`` sorted by descending similarity.
        """
        self._check_fitted()
        assert self._similarity_matrix is not None
        assert self._item_id_to_idx is not None
        assert self._item_ids is not None

        if item not in self._item_id_to_idx:
            raise ValueError(f"Item {item!r} not found in the item catalogue.")

        idx = self._item_id_to_idx[item]
        sim_scores = self._similarity_matrix[idx]
        top_n = np.argsort(sim_scores)[::-1][:n]
        item_ids = np.array([self._item_ids[i] for i in top_n])
        return item_ids, sim_scores[top_n]

    # ── helpers ────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

    @property
    def similarity_matrix(self) -> np.ndarray:
        """Full pairwise cosine similarity matrix (n_items × n_items)."""
        self._check_fitted()
        assert self._similarity_matrix is not None
        return self._similarity_matrix
