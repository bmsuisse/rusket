"""Mixin classes for pattern mining models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from .._type_utils import is_dataframe_empty


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
        from ..association_rules import _ALL_METRICS
        from ..association_rules import association_rules as _assoc_rules

        if return_metrics is None:
            return_metrics = _ALL_METRICS

        # self.fit() / self.mine() must be implemented by the subclass
        df_freq = self.fit().predict()  # type: ignore

        is_empty = is_dataframe_empty(df_freq)

        if is_empty:
            from rusket._dependencies import import_optional_dependency

            pd = import_optional_dependency("pandas")

            empty_df = pd.DataFrame(columns=["antecedents", "consequents"] + return_metrics)
            if hasattr(self, "_convert_to_orig_type"):
                return self._convert_to_orig_type(empty_df)  # type: ignore[attr-defined]
            return empty_df

        # self._num_itemsets must be set by Model.__init__
        num_itemsets = getattr(self, "_num_itemsets", None)
        if num_itemsets is None:
            try:
                num_itemsets = len(df_freq)
            except TypeError:
                num_itemsets = getattr(df_freq, "height", None)
                if num_itemsets is None:
                    num_itemsets = df_freq.count() if hasattr(df_freq, "count") else 0

        result_df = _assoc_rules(
            df_freq,
            num_itemsets=num_itemsets,
            metric=metric,
            min_threshold=min_threshold,
            return_metrics=return_metrics,
        )
        if hasattr(self, "_convert_to_orig_type"):
            return self._convert_to_orig_type(result_df)  # type: ignore[attr-defined]
        return result_df

    def rules_grouped(
        self,
        df_freq: Any,
        group_col: str,
        num_itemsets: dict[Any, int] | int | None = None,
        metric: str = "confidence",
        min_threshold: float = 0.8,
    ) -> Any:
        """Distribute Association Rule Mining across PySpark partitions.

        Parameters
        ----------
        df_freq : Any
            The PySpark ``DataFrame`` containing frequent itemsets (output of ``mine_grouped``).
        group_col : str
            The column to group by.
        num_itemsets : dict[Any, int] | int | None, optional
            A dictionary mapping group IDs to their total transaction count,
            or a single integer if all groups have the same number of transactions.
            If None, attempts to use the total transaction count of the original DataFrame (if known).
        metric : str, default='confidence'
            The metric to filter by (e.g. "confidence", "lift").
        min_threshold : float, default=0.8
            The minimal threshold for the evaluation metric.

        Returns
        -------
        pyspark.sql.DataFrame
            A PySpark DataFrame containing association rules for each group.
        """
        from ..spark import rules_grouped

        if num_itemsets is None:
            num_itemsets = getattr(self, "_num_itemsets", 0)

        return rules_grouped(
            df=df_freq,
            group_col=group_col,
            num_itemsets=num_itemsets,  # type: ignore[arg-type]
            metric=metric,
            min_threshold=min_threshold,
        )

    def recommend_for_cart(self, items: list[Any], n: int = 5) -> list[Any]:
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

        is_empty = is_dataframe_empty(rules_df)

        if is_empty:
            return []

        # Ensure we are working with Pandas for recommend_for_cart logic since we use .apply()
        from rusket._dependencies import import_optional_dependency

        pd = import_optional_dependency("pandas")

        if not isinstance(rules_df, pd.DataFrame):
            if hasattr(rules_df, "to_pandas"):
                rules_df = rules_df.to_pandas()
            else:
                rules_df = rules_df.toPandas()

        cart_set = set(items)
        valid_rules = rules_df[rules_df["antecedents"].apply(lambda ant: set(ant).issubset(cart_set))].sort_values(
            by=["lift", "confidence"], ascending=False
        )  # type: ignore

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

    def recommend_items(self, items: list[Any], n: int = 5) -> list[Any]:
        """Deprecated: use :meth:`recommend_for_cart` instead."""
        import warnings

        warnings.warn(
            "RuleMinerMixin.recommend_items() is deprecated. Use recommend_for_cart() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.recommend_for_cart(items, n)
