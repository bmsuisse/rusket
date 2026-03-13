"""Base classes for recommender models (implicit and sequential)."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from typing_extensions import Self

from .._internal._embedding_mixin import EmbeddingMixin
from .._internal._type_utils import try_import_polars
from ._base import BaseModel


class ImplicitRecommender(BaseModel, EmbeddingMixin):
    """Base class for implicit feedback recommender models.

    Inherited by ALS and BPR.
    """

    def __init__(self, **kwargs: Any):
        self._user_labels: list[Any] | None = None
        self._item_labels: list[Any] | None = None
        self.item_names: list[Any] | None = None

    @classmethod
    def from_transactions(
        cls,
        data: Any,
        transaction_col: str | None = None,
        item_col: str | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> Self:
        """Initialize the model from a long-format DataFrame.

        Prepares the interaction matrix but does **not** fit the model.
        Call ``.fit()`` explicitly to train.

        Parameters
        ----------
        data : pd.DataFrame | pl.DataFrame | pyspark.sql.DataFrame
            Event log containing users, items, and ratings.
        transaction_col : str, optional
            Column name identifying the user ID (aliases user_col).
        item_col : str, optional
            Column name identifying the item ID.
        verbose : int, optional
            Verbosity level.
        **kwargs
            Model hyperparameters (e.g., factors, learning_rate) passed to __init__.
            Can also include `user_col` and `rating_col`.
        """
        user_col = kwargs.pop("user_col", transaction_col)
        rating_col = kwargs.pop("rating_col", None)
        model = cls(verbose=bool(verbose), **kwargs)
        return model._prepare_transactions(data, user_col, item_col, rating_col)

    def fit_transactions(
        self,
        data: Any,
        user_col: str | None = None,
        item_col: str | None = None,
        rating_col: str | None = None,
    ) -> ImplicitRecommender:
        import warnings

        warnings.warn(
            "fit_transactions is deprecated. Use from_transactions() instead.", DeprecationWarning, stacklevel=2
        )
        return self._fit_transactions(data, user_col, item_col, rating_col)

    def _prepare_transactions(
        self,
        data: Any,
        user_col: str | None = None,
        item_col: str | None = None,
        rating_col: str | None = None,
    ) -> ImplicitRecommender:
        """Prepare interaction matrix from a long-format DataFrame without fitting."""
        import numpy as np

        from rusket._dependencies import import_optional_dependency

        _pd = import_optional_dependency("pandas")
        from scipy import sparse as sp

        from .._internal._compat import to_dataframe

        data = to_dataframe(data)

        cols = list(data.columns)
        u_col = user_col or str(cols[0])
        i_col = item_col or str(cols[1])

        pl, is_polars_available = try_import_polars()
        is_polars = is_polars_available and isinstance(data, pl.DataFrame)

        if not (isinstance(data, _pd.DataFrame) or is_polars):
            raise TypeError(f"Expected Pandas/Polars/Spark DataFrame, got {type(data)}")

        u_data = data[u_col].to_numpy() if is_polars else data[u_col]
        i_data = data[i_col].to_numpy() if is_polars else data[i_col]

        user_codes, user_uniques = _pd.factorize(u_data, sort=False)
        item_codes, item_uniques = _pd.factorize(i_data, sort=True)
        n_users = len(user_uniques)
        n_items = len(item_uniques)

        values = (
            np.asarray(data[rating_col], dtype=np.float32)
            if rating_col is not None
            else np.ones(len(user_codes), dtype=np.float32)
        )

        csr = sp.csr_matrix(
            (values, (user_codes.astype(np.int64), item_codes.astype(np.int64))),
            shape=(n_users, n_items),
        )
        self._user_labels = list(user_uniques)
        self._item_labels = list(item_uniques)
        self.item_names = self._item_labels
        self._prepared_interactions = csr
        return self

    def _fit_transactions(
        self,
        data: Any,
        user_col: str | None = None,
        item_col: str | None = None,
        rating_col: str | None = None,
    ) -> ImplicitRecommender:
        """Prepare and fit from a long-format DataFrame (backward compat)."""
        self._prepare_transactions(data, user_col, item_col, rating_col)
        return self.fit(self._prepared_interactions)

    @abstractmethod
    def fit(self, interactions: Any = None) -> ImplicitRecommender:
        """Fit the model to a user-item interaction matrix.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def recommend_items(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> tuple[Any, Any]:
        """Top-N items for a user.

        Must be implemented by subclasses.
        """
        pass

    def predict(self, user_id: int, item_id: int) -> float:
        """Predict the score for a user-item pair.

        Parameters
        ----------
        user_id : int
            User index.
        item_id : int
            Item index.

        Returns
        -------
        float
            Predicted score.
        """
        import numpy as np

        ids, scores = self.recommend_items(user_id, n=self._n_items, exclude_seen=False)  # type: ignore[attr-defined]
        idx = np.where(ids == item_id)[0]
        if len(idx) == 0:
            return 0.0
        return float(scores[idx[0]])

    def recommend_users(self, item_id: int, n: int = 10) -> tuple[Any, Any]:
        """Top-N users for an item.

        Override in subclasses that support this operation.

        Raises
        ------
        NotImplementedError
            If the subclass does not support this operation.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support recommend_users.")

    @property
    def item_factors(self) -> Any:
        """Item factor matrix (n_items, factors)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement item_factors.")

    @property
    def user_factors(self) -> Any:
        """User factor matrix (n_users, factors)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement user_factors.")

    @property
    def item_embeddings(self) -> Any:
        """Alias for item_factors, commonly used in GenAI/LLM contexts."""
        return self.item_factors

    def similar_items(self, item_id: int, n: int = 5) -> tuple[Any, Any]:
        """Find the most similar items to a given item ID.

        Computes cosine similarity between the specified item's latent vector
        and all other item vectors in the ``item_factors`` matrix.

        Parameters
        ----------
        item_id : int
            The internal integer index of the target item.
        n : int, default=5
            Number of most similar items to return.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(item_ids, cosine_similarities)`` sorted in descending order.
        """
        from .._internal.similarity import similar_items

        return similar_items(self, item_id, n)

    def export_factors(
        self,
        include_labels: bool = True,
        normalize: bool = False,
        format: Literal["pandas", "polars", "spark"] = "pandas",
    ) -> Any:
        """Exports latent item factors as a DataFrame for Vector DBs.

        Parameters
        ----------
        include_labels : bool, default=True
            Whether to include the string item labels (if available).
        normalize : bool, default=False
            Whether to L2-normalize the factors before export.
        format : str, default="pandas"
            The DataFrame format to return. One of "pandas", "polars", or "spark".

        Returns
        -------
        Any
            A DataFrame with columns ``item_id``, optionally ``item_label``,
            and ``vector``.
        """
        from ..export.factors import export_item_factors

        return export_item_factors(
            self,
            include_labels=include_labels,
            normalize=normalize,
            format=format,
        )

    def export_user_factors(
        self,
        include_labels: bool = True,
        normalize: bool = False,
        format: Literal["pandas", "polars", "spark"] = "pandas",
    ) -> Any:
        """Exports latent user factors as a DataFrame.

        Parameters
        ----------
        include_labels : bool, default=True
            Whether to include the string user labels (if available).
        normalize : bool, default=False
            Whether to L2-normalize the factors before export.
        format : str, default="pandas"
            The DataFrame format to return. One of "pandas", "polars", or "spark".

        Returns
        -------
        Any
            A DataFrame with columns ``user_id``, optionally ``user_label``,
            and ``vector``.
        """
        from ..export.factors import export_user_factors

        return export_user_factors(
            self,
            include_labels=include_labels,
            normalize=normalize,
            format=format,
        )

    def visualize_factors(self, labels: bool = True, n_items: int | None = None) -> Any:
        """Visualizes the item latent space in 3D using PCA.

        Requires ``plotly``.

        Parameters
        ----------
        labels : bool, default=True
            Whether to show item labels on hover.
        n_items : int, optional
            Limit visualization to the first N items.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from ..viz.plots import visualize_latent_space

        return visualize_latent_space(self, labels=labels, n_items=n_items)


class SequentialRecommender(BaseModel, EmbeddingMixin):
    """Base class for sequential recommendation models.

    Inherited by FPMC.
    """

    def __init__(self, **kwargs: Any):
        self._user_labels: list[Any] | None = None
        self._item_labels: list[Any] | None = None
        self.item_names: list[Any] | None = None

    @classmethod
    def from_transactions(
        cls,
        data: Any,
        transaction_col: str | None = None,
        item_col: str | None = None,
        timestamp_col: str | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> SequentialRecommender:
        raise NotImplementedError("from_transactions not yet implemented for SequentialRecommender")

    @property
    def item_factors(self) -> Any:
        """Item factor matrix (n_items, factors)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement item_factors.")
