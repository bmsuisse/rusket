"""Shared base class for FP-Growth and FP-TDA estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import polars as pl

    class _SparkDataFrame:
        def toPandas(self) -> pd.DataFrame: ...


class FPBase(ABC):
    """Abstract base class for frequent-pattern estimators.

    Subclasses implement :meth:`_run_dense` and :meth:`_run_csr` to swap in
    different Rust backends (FP-Growth, FP-TDA, …) while sharing all data
    dispatch, validation, result building, and the Spark/list-of-items API.

    Parameters
    ----------
    min_support:
        Minimum support threshold (fraction of transactions) in ``(0, 1]``.
    min_confidence:
        Minimum confidence for association rules.  ``None`` skips rule
        generation.
    items_col:
        Column name containing item lists when the input is a list-of-items
        DataFrame (Spark-style).  Ignored for one-hot-encoded inputs.
    use_colnames:
        Replace integer column indices with actual column names in output.
    max_len:
        Maximum length of frequent itemsets.  ``None`` means no limit.
    """

    def __init__(
        self,
        min_support: float = 0.3,
        min_confidence: float | None = None,
        items_col: str = "items",
        use_colnames: bool = False,
        max_len: int | None = None,
    ) -> None:
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.items_col = items_col
        self.use_colnames = use_colnames
        self.max_len = max_len

        self._freq_itemsets: pd.DataFrame | None = None
        self._association_rules: pd.DataFrame | None = None
        self._n_transactions: int = 0

    # ------------------------------------------------------------------
    # Abstract interface — subclasses provide algorithm-specific call
    # ------------------------------------------------------------------

    @abstractmethod
    def _mine(
        self,
        df: pd.DataFrame | pl.DataFrame | np.ndarray | Any,
    ) -> pd.DataFrame:
        """Run the mining algorithm and return a ``support / itemsets`` DataFrame."""
        ...

    # ------------------------------------------------------------------
    # Class-method constructors (Spark, list-of-items, …)
    # ------------------------------------------------------------------

    @classmethod
    def from_spark(
        cls,
        spark_df: Any,
        min_support: float = 0.3,
        min_confidence: float | None = None,
        items_col: str = "items",
        use_colnames: bool = False,
        max_len: int | None = None,
    ) -> FPBase:
        """Create and fit from a Spark DataFrame.

        The Spark DataFrame is converted via ``.toPandas()`` and must be
        one-hot-encoded (bool / 0-1 integer columns).
        """
        model = cls(
            min_support=min_support,
            min_confidence=min_confidence,
            items_col=items_col,
            use_colnames=use_colnames,
            max_len=max_len,
        )
        import pandas as pd  # noqa: F401

        pandas_df: pd.DataFrame = spark_df.toPandas()
        return model.fit(pandas_df)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame | pl.DataFrame | np.ndarray | Any,
    ) -> FPBase:
        """Fit the model on a one-hot-encoded DataFrame (or array).

        Accepts pandas, Polars, NumPy array, or a PySpark DataFrame.
        """
        import typing

        t = type(df).__name__
        if t == "DataFrame" and getattr(df, "__module__", "").startswith("pyspark"):
            df = typing.cast(Any, df).toPandas()

        self._freq_itemsets = self._mine(df)

        t2 = type(df).__name__
        if t2 == "ndarray":
            self._n_transactions = int(df.shape[0])  # type: ignore[union-attr]
        elif hasattr(df, "shape"):
            self._n_transactions = int(df.shape[0])
        else:
            self._n_transactions = len(df)

        if self.min_confidence is not None:
            from .association_rules import association_rules as _ar

            self._association_rules = _ar(
                self._freq_itemsets,
                num_itemsets=self._n_transactions,
                metric="confidence",
                min_threshold=self.min_confidence,
            )
        else:
            self._association_rules = None

        return self

    # ------------------------------------------------------------------
    # Model attributes (Spark-compatible camelCase aliases provided)
    # ------------------------------------------------------------------

    @property
    def freq_itemsets(self) -> pd.DataFrame:
        """Frequent itemsets DataFrame (equivalent to Spark's ``freqItemsets``)."""
        if self._freq_itemsets is None:
            raise RuntimeError("Call fit() before accessing freq_itemsets.")
        return self._freq_itemsets

    @property
    def freqItemsets(self) -> pd.DataFrame:  # noqa: N802
        """Alias for :attr:`freq_itemsets` (Spark camelCase spelling)."""
        return self.freq_itemsets

    @property
    def association_rules_(self) -> pd.DataFrame:
        """Association rules DataFrame (requires *min_confidence* to be set)."""
        if self._association_rules is None:
            if self.min_confidence is None:
                raise RuntimeError(
                    "Set min_confidence in the constructor to generate rules."
                )
            raise RuntimeError("Call fit() before accessing association_rules_.")
        return self._association_rules

    @property
    def associationRules(self) -> pd.DataFrame:  # noqa: N802
        """Alias for :attr:`association_rules_` (Spark camelCase spelling)."""
        return self.association_rules_

    def __repr__(self) -> str:
        fitted = self._freq_itemsets is not None
        return (
            f"{type(self).__name__}("
            f"min_support={self.min_support}, "
            f"min_confidence={self.min_confidence}, "
            f"fitted={fitted})"
        )
