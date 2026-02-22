from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


class RuleMinerMixin:
    """Mixin for association rules and recommendations on frequent itemset models."""

    def association_rules(
        self,
        metric: str = "confidence",
        min_threshold: float = 0.8,
        return_metrics: list[str] | None = None,
    ) -> "pd.DataFrame":
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
        from .association_rules import association_rules as _assoc_rules, _ALL_METRICS

        if return_metrics is None:
            return_metrics = _ALL_METRICS

        # self.mine() must be implemented by the subclass
        df_freq = self.mine()  # type: ignore

        if df_freq is None or df_freq.empty:
            import pandas as pd
            return pd.DataFrame(columns=["antecedents", "consequents"] + return_metrics)

        # self._num_itemsets must be set by Model.__init__
        return _assoc_rules(
            df_freq,
            num_itemsets=getattr(self, "_num_itemsets", len(df_freq)),
            metric=metric,
            min_threshold=min_threshold,
            return_metrics=return_metrics,
        )

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
        """
        rules_df = self.association_rules(metric="lift", min_threshold=1.0)
        
        if rules_df.empty:
            return []

        cart_set = frozenset(items)
        valid_rules = rules_df[
            rules_df["antecedents"].apply(
                lambda ant: frozenset(ant).issubset(cart_set)
            )
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
    
    def __init__(self, data: "pd.DataFrame | Any", item_names: list[str] | None = None, **kwargs: Any):
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
        self.item_names = item_names if item_names is not None else (
            list(data.columns) if hasattr(data, "columns") else None
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

    @classmethod
    def from_transactions(
        cls,
        data: "pd.DataFrame | pl.DataFrame | Sequence[Sequence[str | int]] | Any",
        transaction_col: str | None = None,
        item_col: str | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> "BaseModel":
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

        if isinstance(data, (list, tuple)):
            sparse_df = _from_list(data, verbose=verbose)
            return cls(sparse_df, **kwargs)

        import pandas as _pd
        import polars as _pl

        if not isinstance(data, (_pd.DataFrame, _pl.DataFrame)):
            raise TypeError(
                f"Expected a Pandas/Polars/Spark DataFrame or list of lists, "
                f"got {type(data)}"
            )

        if isinstance(data, _pl.DataFrame):
            data = data.to_pandas()

        sparse_df = _from_dataframe(data, transaction_col, item_col, verbose=verbose)
        return cls(sparse_df, **kwargs)

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        transaction_col: str | None = None,
        item_col: str | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> "BaseModel":
        """Shorthand for ``from_transactions(df, transaction_col, item_col)``."""
        return cls.from_transactions(
            df, transaction_col=transaction_col, item_col=item_col, verbose=verbose, **kwargs
        )

    @classmethod
    def from_polars(
        cls,
        df: pl.DataFrame,
        transaction_col: str | None = None,
        item_col: str | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> "BaseModel":
        """Shorthand for ``from_transactions(df, transaction_col, item_col)``."""
        return cls.from_transactions(
            df, transaction_col=transaction_col, item_col=item_col, verbose=verbose, **kwargs
        )

    @classmethod
    def from_spark(
        cls,
        df: Any,
        transaction_col: str | None = None,
        item_col: str | None = None,
        **kwargs: Any,
    ) -> "BaseModel":
        """Shorthand for ``from_transactions(df, transaction_col, item_col)``."""
        return cls.from_transactions(
            df, transaction_col=transaction_col, item_col=item_col, **kwargs
        )

class Miner(BaseModel):
    """Base class for all pattern mining algorithms.
    
    Inherited by FPGrowth, Eclat, AutoMiner, PrefixSpan, and HUPM.
    """
    
    @abstractmethod
    def mine(self, **kwargs: Any) -> "pd.DataFrame":
        """Execute the mining algorithm and return frequent patterns.
        
        Must be implemented by subclasses.
        """
        pass


class ImplicitRecommender(BaseModel):
    """Base class for implicit feedback recommender models.
    
    Inherited by ALS and BPR.
    """
    
    @abstractmethod
    def fit(self, interactions: Any) -> "ImplicitRecommender":
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
