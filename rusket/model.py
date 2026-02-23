from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from typing_extensions import Self


class RuleMinerMixin:
    """Mixin for association rules and recommendations on frequent itemset models."""

    # Cache: (metric, min_threshold) -> rules DataFrame
    _rules_cache: dict[tuple[str, float], Any] | None = None

    def _invalidate_rules_cache(self) -> None:
        """Clear the cached association rules (call after re-mining)."""
        self._rules_cache = None

    def association_rules(
        self,
        metric: str = "confidence",
        min_threshold: float = 0.8,
        return_metrics: list[str] | None = None,
    ) -> pd.DataFrame:
        """Generate association rules from the mined frequent itemsets.

        Parameters
        ----------
        metric : str, default='confidence'
            The metric to evaluate if a rule is of interest.
        min_threshold : float, default=0.8
            The minimum threshold for the evaluation metric.
        return_metrics : list[str] | None, default=None
            List of metrics to include in the resulting DataFrame. Defaults to all available metrics.

        Returns
        -------
        pd.DataFrame
            DataFrame of strong association rules.
        """
        from .association_rules import _ALL_METRICS
        from .association_rules import association_rules as _assoc_rules

        if return_metrics is None:
            return_metrics = _ALL_METRICS

        # self.mine() must be implemented by the subclass
        df_freq = self.mine()  # type: ignore

        is_empty = False
        if df_freq is None:
            is_empty = True
        elif hasattr(df_freq, "empty"):
            is_empty = df_freq.empty
        elif hasattr(df_freq, "is_empty"):
            # Polars
            is_empty = df_freq.is_empty()
        elif hasattr(df_freq, "isEmpty"):
            # Spark
            is_empty = df_freq.isEmpty()
        else:
            try:
                is_empty = len(df_freq) == 0
            except Exception:
                pass

        if is_empty:
            import pandas as pd

            empty_df = pd.DataFrame(columns=["antecedents", "consequents"] + return_metrics)
            if hasattr(self, "_convert_to_orig_type"):
                return self._convert_to_orig_type(empty_df)  # type: ignore[attr-defined]
            return empty_df

        # self._num_itemsets must be set by Model.__init__
        result_df = _assoc_rules(
            df_freq,
            num_itemsets=getattr(self, "_num_itemsets", len(df_freq)),
            metric=metric,
            min_threshold=min_threshold,
            return_metrics=return_metrics,
        )
        if hasattr(self, "_convert_to_orig_type"):
            return self._convert_to_orig_type(result_df)  # type: ignore[attr-defined]
        return result_df

    def recommend_items(self, items: list[Any], n: int = 5) -> list[Any]:
        """Suggest items to add to an active cart using association rules.

        Parameters
        ----------
        items : list[Any]
            The items currently in the cart or basket.
        n : int, default=5
            The maximum number of items to recommend.

        Returns
        -------
        list[Any]
            List of recommended items, ordered by lift and then confidence.

        Notes
        -----
        Rules are computed once and cached with the key ``(metric="lift",
        min_threshold=1.0)``.  Call :meth:`_invalidate_rules_cache` to force
        a re-computation after re-mining.
        """
        cache_key = ("lift", 1.0)
        if self._rules_cache is None:
            self._rules_cache = {}
        if cache_key not in self._rules_cache:
            self._rules_cache[cache_key] = self.association_rules(metric="lift", min_threshold=1.0)

        rules_df = self._rules_cache[cache_key]

        is_empty = False
        if rules_df is None:
            is_empty = True
        elif hasattr(rules_df, "empty"):
            is_empty = rules_df.empty
        elif hasattr(rules_df, "is_empty"):
            # Polars
            is_empty = rules_df.is_empty()
        elif hasattr(rules_df, "isEmpty"):
            # Spark
            is_empty = rules_df.isEmpty()
        else:
            try:
                is_empty = len(rules_df) == 0
            except Exception:
                pass

        if is_empty:
            return []

        # Ensure we are working with Pandas for recommend_items logic since we use .apply()
        import pandas as pd
        if not isinstance(rules_df, pd.DataFrame):
            try:
                if hasattr(rules_df, "to_pandas"):
                    rules_df = rules_df.to_pandas()
                else:
                    rules_df = rules_df.toPandas()
            except Exception:
                pass

        cart_set = frozenset(items)
        valid_rules = rules_df[
            rules_df["antecedents"].apply(lambda ant: frozenset(ant).issubset(cart_set))
        ].sort_values(by=["lift", "confidence"], ascending=False)  # type: ignore

        if valid_rules.empty:
            return []

        suggestions: list[Any] = []
        for consequents in valid_rules["consequents"]:
            for item in consequents:
                if item not in cart_set and item not in suggestions:
                    suggestions.append(item)
                    if len(suggestions) >= n:
                        return suggestions
        return suggestions


class BaseModel(ABC):
    """Abstract base class for all rusket algorithms.

    Provides unified data ingestion methods (from_transactions, from_pandas, etc.)
    for any downstream Miner or Recommender.
    """

    @classmethod
    @abstractmethod
    def from_transactions(
        cls,
        data: Any,
        transaction_col: str | None = None,
        item_col: str | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> Self:
        """Initialize the model from a long-format DataFrame or sequences.

        Must be implemented by subclasses.
        """
        pass

    def __dir__(self) -> list[str]:
        """Provides a clean public API surface for AI code assistants and REPLs.
        Filters out internal properties starting with underscores.
        """
        return [k for k in super().__dir__() if not k.startswith("_")]

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        transaction_col: str | None = None,
        item_col: str | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> Self:
        """Shorthand for ``from_transactions(df, transaction_col, item_col)``."""
        return cls.from_transactions(df, transaction_col=transaction_col, item_col=item_col, verbose=verbose, **kwargs)

    @classmethod
    def from_polars(
        cls,
        df: pl.DataFrame,
        transaction_col: str | None = None,
        item_col: str | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> Self:
        """Shorthand for ``from_transactions(df, transaction_col, item_col)``."""
        return cls.from_transactions(df, transaction_col=transaction_col, item_col=item_col, verbose=verbose, **kwargs)

    @classmethod
    def from_spark(
        cls,
        df: Any,
        transaction_col: str | None = None,
        item_col: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Shorthand for ``from_transactions(df, transaction_col, item_col)``."""
        return cls.from_transactions(df, transaction_col=transaction_col, item_col=item_col, **kwargs)


class Miner(BaseModel):
    """Base class for all pattern mining algorithms.

    Inherited by FPGrowth, Eclat, AutoMiner, PrefixSpan, and HUPM.
    """

    def __init__(self, data: pd.DataFrame | Any, item_names: list[str] | None = None, **kwargs: Any):
        """Initialize the miner with pre-formatted data.

        Parameters
        ----------
        data : pd.DataFrame | Any
            A one-hot encoded dataset (e.g. Pandas DataFrame, SciPy sparse matrix).
        item_names : list[str], optional
            Column names if data is a raw numpy/scipy array.
            If not provided, and data is a DataFrame, columns are inferred.
        **kwargs
            Algorithm-specific mining parameters (min_support, max_len, etc.).
        """
        self.data = data
        self.item_names = (
            item_names if item_names is not None else (list(data.columns) if hasattr(data, "columns") else None)
        )
        self.kwargs = kwargs

        # Keep track of the number of transactions for metric calculations later
        if hasattr(self.data, "shape") and len(self.data.shape) > 0:
            self._num_itemsets = self.data.shape[0]
        else:
            try:
                self._num_itemsets = len(self.data)
            except TypeError:
                self._num_itemsets = 0  # Fallback for unknown iterables

        # Store the original dataframe type to convert outputs back
        _type = type(self.data)
        self._orig_df_type: str = "pandas"
        if _type.__name__ == "DataFrame":
            mod_name = getattr(_type, "__module__", "")
            if mod_name.startswith("pyspark"):
                self._orig_df_type = "spark"
            elif mod_name.startswith("polars"):
                self._orig_df_type = "polars"

    def _convert_to_orig_type(self, df: pd.DataFrame) -> Any:
        """Helper to convert the resulting pandas DataFrame back to the input DataFrame type."""
        import pandas as pd

        if df is None or not isinstance(df, pd.DataFrame):
            return df

        if self._orig_df_type == "polars":
            import polars as pl

            # Convert frozensets to lists for pyarrow compatibility
            for col in ["antecedents", "consequents", "itemsets"]:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: list(x) if isinstance(x, (frozenset, set)) else x)

            return pl.from_pandas(df)
        elif self._orig_df_type == "spark":
            # Best-effort conversion to Spark
            try:
                from pyspark.sql import SparkSession

                # Convert frozensets to lists for Spark schema compatibility
                for col in ["antecedents", "consequents", "itemsets"]:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: list(x) if isinstance(x, (frozenset, set)) else x)

                spark = SparkSession.getActiveSession()
                if spark is not None:
                    # Arrow conversion requires types_mapper trick for ArrowDtype
                    # Since we are returning strings/floats/lists, basic createDataFrame usually works
                    return spark.createDataFrame(df)
            except ImportError:
                pass
        return df

    @classmethod
    def from_transactions(
        cls,
        data: pd.DataFrame | pl.DataFrame | Sequence[Sequence[str | int]] | Any,
        transaction_col: str | None = None,
        item_col: str | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> Self:
        """Load long-format transactional data into the algorithm.

        Parameters
        ----------
        data
            One of:

            - **Pandas / Polars / Spark DataFrame** with (at least) two columns:
              one for the transaction identifier and one for the item.
            - **List of lists** where each inner list contains the items of a
              single transaction, e.g. ``[["bread", "milk"], ["bread", "eggs"]]``.
        transaction_col
            Name of the column that identifies transactions. If ``None`` the
            first column is used. Ignored for list-of-lists input.
        item_col
            Name of the column that contains item values. If ``None`` the
            second column is used. Ignored for list-of-lists input.
        verbose : int, default=0
            Whether to print progress details.
        **kwargs
            Algorithm-specific parameters saved into the Miner (e.g., ``min_support``).

        Returns
        -------
        Miner
            Configured miner instance, ready to call ``.mine()``.
        """
        from ._compat import to_dataframe
        from .transactions import _from_dataframe, _from_list

        data = to_dataframe(data)
        _type = type(data)
        _orig_type = "pandas"
        if _type.__name__ == "DataFrame":
            mod_name = getattr(_type, "__module__", "")
            if mod_name.startswith("pyspark"):
                _orig_type = "spark"
            elif mod_name.startswith("polars"):
                _orig_type = "polars"

        if isinstance(data, (list, tuple)):
            sparse_df = _from_list(data, verbose=verbose)
            miner = cls(sparse_df, **kwargs)
            miner._orig_df_type = "pandas"
            return miner

        import pandas as _pd
        import polars as _pl

        if not isinstance(data, (_pd.DataFrame, _pl.DataFrame)):
            raise TypeError(f"Expected a Pandas/Polars/Spark DataFrame or list of lists, got {type(data)}")

        if isinstance(data, _pl.DataFrame):
            data = data.to_pandas()

        sparse_df = _from_dataframe(data, transaction_col, item_col, verbose=verbose)
        miner = cls(sparse_df, **kwargs)
        miner._orig_df_type = _orig_type
        return miner

    @abstractmethod
    def mine(self, **kwargs: Any) -> pd.DataFrame:
        """Execute the mining algorithm and return frequent patterns.

        Must be implemented by subclasses.
        """
        pass


class ImplicitRecommender(BaseModel):
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
    ) -> ImplicitRecommender:
        """Initialize and fit the model from a long-format DataFrame.

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
        return model._fit_transactions(data, user_col, item_col, rating_col)

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

    def _fit_transactions(
        self,
        data: Any,
        user_col: str | None = None,
        item_col: str | None = None,
        rating_col: str | None = None,
    ) -> ImplicitRecommender:
        """Fit from a long-format Pandas/Polars/Spark DataFrame."""
        import numpy as np
        import pandas as _pd
        from scipy import sparse as sp

        from ._compat import to_dataframe

        data = to_dataframe(data)

        cols = list(data.columns)
        u_col = user_col or str(cols[0])
        i_col = item_col or str(cols[1])

        try:
            import polars as pl

            is_polars = isinstance(data, pl.DataFrame)
        except ImportError:
            is_polars = False

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
        self._item_labels = [str(c) for c in item_uniques]
        self.item_names = self._item_labels
        return self.fit(csr)

    @abstractmethod
    def fit(self, interactions: Any) -> ImplicitRecommender:
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

    @property
    def item_factors(self) -> Any:
        """Item factor matrix (n_items, factors)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement item_factors.")

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
        from .similarity import similar_items

        return similar_items(self, item_id, n)

    def export_factors(self, include_labels: bool = True) -> pd.DataFrame:
        """Exports latent item factors as a Pandas DataFrame for Vector DBs.

        Parameters
        ----------
        include_labels : bool, default=True
            Whether to include the string item labels (if available).

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns ``item_id``, optionally ``item_label``,
            and ``vector``.
        """
        from .export import export_item_factors

        return export_item_factors(self, include_labels=include_labels)

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
        from .viz import visualize_latent_space

        return visualize_latent_space(self, labels=labels, n_items=n_items)


class SequentialRecommender(BaseModel):
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
        verbose: int = 0,
        **kwargs: Any,
    ) -> SequentialRecommender:
        raise NotImplementedError("from_transactions not yet implemented for SequentialRecommender")
