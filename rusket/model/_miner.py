"""Base class for pattern mining algorithms."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from typing_extensions import Self

from .._type_utils import detect_dataframe_type, to_list_if_collection, try_import_polars
from ._base import BaseModel


class Miner(BaseModel):
    """Base class for all pattern mining algorithms.

    Inherited by FPGrowth, Eclat, PrefixSpan, and HUPM.
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
        self._orig_df_type: str = detect_dataframe_type(self.data)

        self._result: Any = None
        self._grouped_result: Any = None

    def _convert_to_orig_type(self, df: pd.DataFrame) -> Any:
        """Helper to convert the resulting pandas DataFrame back to the input DataFrame type."""
        from rusket._dependencies import import_optional_dependency

        pd = import_optional_dependency("pandas")

        if df is None or not isinstance(df, pd.DataFrame):
            return df

        if self._orig_df_type == "pyarrow":
            pa = import_optional_dependency("pyarrow")

            # Convert tuples to lists for Arrow compatibility
            for col in ["antecedents", "consequents", "itemsets"]:
                if col in df.columns:
                    df[col] = df[col].apply(to_list_if_collection)
            return pa.Table.from_pandas(df)
        elif self._orig_df_type == "polars":
            pl = import_optional_dependency("polars")

            # Convert tuples to lists for pyarrow compatibility
            for col in ["antecedents", "consequents", "itemsets"]:
                if col in df.columns:
                    df[col] = df[col].apply(to_list_if_collection)

            return pl.from_pandas(df)
        elif self._orig_df_type == "spark":
            # Best-effort conversion to Spark
            try:
                pyspark_sql = import_optional_dependency("pyspark.sql", "pyspark")
                SparkSession = pyspark_sql.SparkSession

                # Convert tuples to lists for Spark schema compatibility
                for col in ["antecedents", "consequents", "itemsets"]:
                    if col in df.columns:
                        df[col] = df[col].apply(to_list_if_collection)

                spark = SparkSession.getActiveSession()
                if spark is not None:
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
        from .._compat import to_dataframe
        from ..transactions import _from_dataframe, _from_list

        _orig_type = detect_dataframe_type(data)

        data = to_dataframe(data)

        if isinstance(data, (list, tuple)):
            sparse_df = _from_list(data, verbose=verbose)
            miner = cls(sparse_df, **kwargs)
            miner._orig_df_type = "pandas"
            return miner

        from rusket._dependencies import import_optional_dependency

        _pd = import_optional_dependency("pandas")
        _pl = import_optional_dependency("polars")

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
        if self._result is None:
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
        from rusket._dependencies import import_optional_dependency

        _pd = import_optional_dependency("pandas")

        min_support = kwargs.get("min_support", getattr(self, "min_support", 0.5))
        max_len = kwargs.get("max_len", getattr(self, "max_len", None))
        use_colnames = kwargs.get("use_colnames", getattr(self, "use_colnames", True))

        method = "fpgrowth"
        cls_name = type(self).__name__
        if cls_name == "FPGrowth":
            method = "fpgrowth"
        elif cls_name == "Eclat":
            method = "eclat"

        df = self.data

        # ── Spark path ────────────────────────────────────────────────────────
        if getattr(type(df), "__module__", "").startswith("pyspark"):
            from ..spark import mine_grouped as _spark_mine_grouped

            return _spark_mine_grouped(
                df=df,
                group_col=group_col,
                min_support=min_support,
                max_len=max_len,
                method=method,
                use_colnames=use_colnames,
            )

        from ..mine import mine as _mine

        # ── Polars path ───────────────────────────────────────────────────────
        _pl, is_polars_available = try_import_polars()
        is_polars = is_polars_available and isinstance(df, _pl.DataFrame)

        if is_polars and _pl is not None:
            frames: list[Any] = []
            for g in list(df[group_col].unique()):
                sub_pd = df.filter(_pl.col(group_col) == g).drop(group_col).to_pandas().astype(bool)
                res_pd = _mine(
                    sub_pd, min_support=min_support, max_len=max_len, method=method, use_colnames=use_colnames
                )
                if len(res_pd) == 0:
                    continue
                res_pd["itemsets"] = res_pd["itemsets"].apply(to_list_if_collection)
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
