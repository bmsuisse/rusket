from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
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

        # self.fit() / self.mine() must be implemented by the subclass
        df_freq = self.fit().predict()  # type: ignore

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
        from .spark import rules_grouped

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

        # Ensure we are working with Pandas for recommend_for_cart logic since we use .apply()
        import pandas as pd

        if not isinstance(rules_df, pd.DataFrame):
            try:
                if hasattr(rules_df, "to_pandas"):
                    rules_df = rules_df.to_pandas()
                else:
                    rules_df = rules_df.toPandas()
            except Exception:
                pass

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

    @classmethod
    def from_arrow(
        cls,
        table: Any,
        transaction_col: str | None = None,
        item_col: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Shorthand for ``from_transactions(table, transaction_col, item_col)``.

        Parameters
        ----------
        table : pyarrow.Table
            An Arrow table with transaction and item columns.
        transaction_col : str, optional
            Name of the transaction ID column.
        item_col : str, optional
            Name of the item column.
        **kwargs
            Extra arguments forwarded to ``from_transactions``.
        """
        return cls.from_transactions(table, transaction_col=transaction_col, item_col=item_col, **kwargs)

    def save(self, path: str | Path) -> None:
        """Save the model to disk using pickle.

        Parameters
        ----------
        path : str or Path
            File path to write the model to (e.g. ``"model.pkl"``).
        """
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "__rusket_version__": 1,
            "class": type(self).__name__,
            "module": type(self).__module__,
            "state": self.__dict__,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a previously saved model from disk.

        Parameters
        ----------
        path : str or Path
            File path to load from.

        Returns
        -------
        Self
            The restored model.

        Raises
        ------
        TypeError
            If the file contains a different model class.
        """
        import pickle

        path = Path(path)
        with open(path, "rb") as f:
            payload = pickle.load(f)  # noqa: S301

        if isinstance(payload, dict) and "__rusket_version__" in payload:
            saved_cls_name = payload.get("class", "")
            state = payload["state"]
        else:
            # Legacy: plain pickled object
            if isinstance(payload, cls):
                return payload  # type: ignore[return-value]
            raise TypeError(f"Expected {cls.__name__}, got {type(payload).__name__}")

        # Construct an empty instance and restore state
        instance = cls.__new__(cls)  # type: ignore[arg-type]
        instance.__dict__.update(state)

        if saved_cls_name != cls.__name__:
            import warnings

            warnings.warn(
                f"Model was saved as {saved_cls_name} but loaded as {cls.__name__}. "
                "This may cause unexpected behaviour.",
                stacklevel=2,
            )

        return instance  # type: ignore[return-value]


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
        if _type.__name__ == "Table" and getattr(_type, "__module__", "").startswith("pyarrow"):
            self._orig_df_type = "pyarrow"
        elif _type.__name__ == "DataFrame":
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

        if self._orig_df_type == "pyarrow":
            import pyarrow as pa

            # Convert tuples to lists for Arrow compatibility
            for col in ["antecedents", "consequents", "itemsets"]:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: list(x) if isinstance(x, (tuple, set)) else x)
            return pa.Table.from_pandas(df)
        elif self._orig_df_type == "polars":
            import polars as pl

            # Convert tuples to lists for pyarrow compatibility
            for col in ["antecedents", "consequents", "itemsets"]:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: list(x) if isinstance(x, (tuple, set, tuple)) else x)

            return pl.from_pandas(df)
        elif self._orig_df_type == "spark":
            # Best-effort conversion to Spark
            try:
                from pyspark.sql import SparkSession

                # Convert tuples to lists for Spark schema compatibility
                for col in ["antecedents", "consequents", "itemsets"]:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: list(x) if isinstance(x, (tuple, set, tuple)) else x)

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

        _type = type(data)
        _orig_type = "pandas"
        if _type.__name__ == "Table" and getattr(_type, "__module__", "").startswith("pyarrow"):
            _orig_type = "pyarrow"
        elif _type.__name__ == "DataFrame":
            mod_name = getattr(_type, "__module__", "")
            if mod_name.startswith("pyspark"):
                _orig_type = "spark"
            elif mod_name.startswith("polars"):
                _orig_type = "polars"

        data = to_dataframe(data)

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

    def fit(self, **kwargs: Any) -> Self:
        """Sklearn-compatible alias for ``mine()``. Runs the mining algorithm.

        Returns
        -------
        self
        """
        self._result = self.mine(**kwargs)
        return self  # type: ignore[return-value]

    def predict(self, **kwargs: Any) -> pd.DataFrame:
        """Return the last mined result, or run ``fit()`` first.

        Returns
        -------
        pd.DataFrame
            The frequent itemsets / patterns.
        """
        if not hasattr(self, "_result") or self._result is None:
            self.fit(**kwargs)
        return self._result  # type: ignore[return-value]

    def mine_grouped(self, group_col: str, **kwargs: Any) -> Any:
        """Mine frequent itemsets independently for every group in a DataFrame.

        Works with **Pandas**, **Polars**, and **PySpark** DataFrames.
        The output type always matches the input type.

        Parameters
        ----------
        group_col : str
            The column to group by (e.g. ``store_id``).
        **kwargs
            Additional arguments such as ``min_support``, ``max_len``, ``use_colnames``.

        Returns
        -------
        pd.DataFrame | pl.DataFrame | pyspark.sql.DataFrame
            A DataFrame containing ``group_col``, ``support``, and ``itemsets``.
            The type mirrors the input ``data`` type.
        """
        import pandas as _pd

        min_support = kwargs.get("min_support", getattr(self, "min_support", 0.5))
        max_len = kwargs.get("max_len", getattr(self, "max_len", None))
        use_colnames = kwargs.get("use_colnames", getattr(self, "use_colnames", True))

        method = "auto"
        cls_name = type(self).__name__
        if cls_name == "FPGrowth":
            method = "fpgrowth"
        elif cls_name == "Eclat":
            method = "eclat"

        df = self.data

        # ── Spark path ────────────────────────────────────────────────────────
        if getattr(type(df), "__module__", "").startswith("pyspark"):
            from .spark import mine_grouped as _spark_mine_grouped

            return _spark_mine_grouped(
                df=df,
                group_col=group_col,
                min_support=min_support,
                max_len=max_len,
                method=method,
                use_colnames=use_colnames,
            )

        from .mine import mine as _mine

        # ── Polars path ───────────────────────────────────────────────────────
        try:
            import polars as _pl

            is_polars = isinstance(df, _pl.DataFrame)
        except ImportError:
            is_polars = False

        if is_polars:
            frames: list[Any] = []
            for g in df[group_col].unique().to_list():
                sub_pd = df.filter(_pl.col(group_col) == g).drop(group_col).to_pandas().astype(bool)
                res_pd = _mine(
                    sub_pd, min_support=min_support, max_len=max_len, method=method, use_colnames=use_colnames
                )
                if len(res_pd) == 0:
                    continue
                res_pd["itemsets"] = res_pd["itemsets"].apply(
                    lambda x: list(x) if isinstance(x, (tuple, set, tuple)) else x
                )
                res_pd.insert(0, group_col, g)
                frames.append(res_pd)

            if not frames:
                return _pl.DataFrame(
                    {
                        group_col: _pl.Series([], dtype=_pl.Utf8),
                        "support": _pl.Series([], dtype=_pl.Float64),
                        "itemsets": _pl.Series([], dtype=_pl.List(_pl.Utf8)),
                    }
                )

            return _pl.from_pandas(_pd.concat(frames, ignore_index=True)[[group_col, "support", "itemsets"]])

        # ── Pandas path ───────────────────────────────────────────────────────
        if not isinstance(df, _pd.DataFrame):
            raise TypeError(f"mine_grouped requires a Pandas, Polars, or PySpark DataFrame; got {type(df)}")

        frames_pd: list[_pd.DataFrame] = []
        for g, sub in df.groupby(group_col, sort=False):
            res = _mine(
                sub.drop(columns=[group_col]).astype(bool),
                min_support=min_support,
                max_len=max_len,
                method=method,
                use_colnames=use_colnames,
            )
            if len(res) == 0:
                continue
            res.insert(0, group_col, g)
            frames_pd.append(res)

        if not frames_pd:
            return _pd.DataFrame(columns=[group_col, "support", "itemsets"])

        return _pd.concat(frames_pd, ignore_index=True)

    def fit_grouped(self, group_col: str, **kwargs: Any) -> Any:
        """Sklearn-style alias for :meth:`mine_grouped`. Caches the result.

        Parameters
        ----------
        group_col : str
            The column to group by.
        **kwargs
            Forwarded to :meth:`mine_grouped`.

        Returns
        -------
        self
        """
        self._grouped_result = self.mine_grouped(group_col, **kwargs)
        return self


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
        from .similarity import similar_items

        return similar_items(self, item_id, n)

    def export_factors(
        self,
        include_labels: bool = True,
        normalize: bool = False,
        format: str = "pandas",
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
        from .export import export_item_factors

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
        format: str = "pandas",
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
        from .export import export_user_factors

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
        from .viz import visualize_latent_space

        return visualize_latent_space(self, labels=labels, n_items=n_items)

    def pca(self, n_components: int = 3, normalize: bool = True) -> Any:
        """Reduces the item embeddings to `n_components` dimensions using PCA.

        This enables a fluent visualization API:
        ```python
        model.fit().pca().plot()
        ```

        Parameters
        ----------
        n_components : int, default=3
            Number of principal components to keep.
        normalize : bool, default=True
            Whether to L2-normalize the item factors before PCA computation.
            Normalizing factors often creates a better visualization for cosine distance.

        Returns
        -------
        ProjectedSpace
            A wrapper object containing the projected coordinates, with a ``.plot()`` method.
        """
        import numpy as np

        from .pca import ProjectedSpace, pca

        factors = self.item_factors
        if normalize:
            norms = np.linalg.norm(factors, axis=1, keepdims=True)
            factors = factors / np.clip(norms, a_min=1e-10, a_max=None)

        coords = pca(factors, n_components=n_components)
        return ProjectedSpace(coords, self._item_labels)


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

    @property
    def item_factors(self) -> Any:
        """Item factor matrix (n_items, factors)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement item_factors.")

    def pca(self, n_components: int = 3, normalize: bool = True) -> Any:
        """Reduces the item embeddings to `n_components` dimensions using PCA.

        This enables a fluent visualization API:
        ```python
        model.fit().pca().plot()
        ```

        Parameters
        ----------
        n_components : int, default=3
            Number of principal components to keep.
        normalize : bool, default=True
            Whether to L2-normalize the item factors before PCA computation.
            Normalizing factors often creates a better visualization for cosine distance.

        Returns
        -------
        ProjectedSpace
            A wrapper object containing the projected coordinates, with a ``.plot()`` method.
        """
        import numpy as np

        from .pca import ProjectedSpace, pca

        factors = self.item_factors
        if normalize:
            norms = np.linalg.norm(factors, axis=1, keepdims=True)
            factors = factors / np.clip(norms, a_min=1e-10, a_max=None)

        coords = pca(factors, n_components=n_components)
        return ProjectedSpace(coords, self._item_labels)
