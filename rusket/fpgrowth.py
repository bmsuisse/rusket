from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from . import _rusket as _rust  # type: ignore
from ._fpbase import FPBase

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import polars as pl

    # pyspark is an optional dependency — only used for type hints
    class _SparkDataFrame:
        def toPandas(self) -> pd.DataFrame: ...


def fpgrowth(
    df: pd.DataFrame | pl.DataFrame | np.ndarray | Any,
    min_support: float = 0.5,
    null_values: bool = False,
    use_colnames: bool = False,
    max_len: int | None = None,
    verbose: int = 0,
) -> pd.DataFrame:
    if min_support <= 0.0:
        raise ValueError(
            "`min_support` must be a positive "
            "number within the interval `(0, 1]`. "
            "Got %s." % min_support
        )

    t = type(df).__name__
    import typing

    if t == "DataFrame" and getattr(df, "__module__", "").startswith("pyspark"):
        df = typing.cast(Any, df).toPandas()
        t = "DataFrame"

    if t == "DataFrame" and getattr(df, "__module__", "").startswith("polars"):
        return _fpgrowth_polars(
            typing.cast("pl.DataFrame", df), min_support, use_colnames, max_len
        )

    if t == "ndarray":
        import numpy as np

        df_nd = typing.cast("np.ndarray", df)
        n_rows, n_cols = df_nd.shape
        min_count = math.ceil(min_support * n_rows)
        col_names = [str(i) for i in range(n_cols)]
        data = np.ascontiguousarray(df_nd, dtype=np.uint8)
        raw = _rust.fpgrowth_from_dense(data, min_count, max_len)
        return _build_result(raw, n_rows, min_support, col_names, use_colnames)

    df_pd = typing.cast("pd.DataFrame", df)
    n_rows, n_cols = df_pd.shape
    min_count = math.ceil(min_support * n_rows)
    col_names = list(df_pd.columns)

    from ._validation import valid_input_check

    valid_input_check(df_pd, null_values)

    if hasattr(df_pd, "sparse"):
        raw = _fpgrowth_sparse_pandas(df_pd, n_rows, n_cols, min_count, max_len)
    else:
        raw = _fpgrowth_dense_pandas(df_pd, n_rows, n_cols, min_count, max_len)

    return _build_result(raw, n_rows, min_support, col_names, use_colnames)


def _fpgrowth_dense_pandas(
    df: pd.DataFrame, n_rows: int, n_cols: int, min_count: int, max_len: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np

    data = np.ascontiguousarray(df.values, dtype=np.uint8)
    return _rust.fpgrowth_from_dense(data, min_count, max_len)


def _fpgrowth_sparse_pandas(
    df: pd.DataFrame, n_rows: int, n_cols: int, min_count: int, max_len: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np

    csr = df.sparse.to_coo().tocsr()
    csr.eliminate_zeros()
    indptr = np.asarray(csr.indptr, dtype=np.int32)
    indices = np.asarray(csr.indices, dtype=np.int32)
    return _rust.fpgrowth_from_csr(indptr, indices, n_cols, min_count, max_len)


def _fpgrowth_polars(
    df: pl.DataFrame, min_support: float, use_colnames: bool, max_len: int | None
) -> pd.DataFrame:
    import numpy as np

    n_rows, n_cols = df.shape
    min_count = math.ceil(min_support * n_rows)
    arr = np.ascontiguousarray(df.to_numpy(), dtype=np.uint8)
    raw = _rust.fpgrowth_from_dense(arr, min_count, max_len)
    return _build_result(raw, n_rows, min_support, df.columns, use_colnames)


def _build_result(
    raw: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_rows: int,
    min_support: float,
    col_names: list | Any,
    use_colnames: bool,
) -> pd.DataFrame:
    import pandas as pd
    import pyarrow as pa

    supports_arr, offsets_arr, items_arr = raw

    if len(supports_arr) == 0:
        return pd.DataFrame(columns=["support", "itemsets"])

    supports = supports_arr / n_rows

    if use_colnames:
        col_array = pa.array(col_names)
        items_pa = pa.DictionaryArray.from_arrays(
            pa.array(items_arr, type=pa.int32()), col_array
        )
        item_type = col_array.type
    else:
        items_pa = pa.array(items_arr, type=pa.int32())
        item_type = pa.int32()

    offsets_pa = pa.array(offsets_arr, type=pa.int32())
    list_arr = pa.ListArray.from_arrays(offsets_pa, items_pa)

    result = pd.DataFrame(
        {
            "support": supports,
            "itemsets": pd.Series(list_arr, dtype=pd.ArrowDtype(pa.list_(item_type))),
        }
    )

    filtered_df = result[result["support"] >= min_support].reset_index(drop=True)
    import typing

    return typing.cast("pd.DataFrame", filtered_df)


class FPGrowth(FPBase):
    """Spark-compatible FP-Growth estimator.

    Mirrors the ``pyspark.ml.fpm.FPGrowth`` API so you can swap between the
    two backends with minimal code changes.

    Parameters
    ----------
    min_support:
        Minimum support threshold (fraction of transactions).  Same as
        Spark's ``minSupport``.
    min_confidence:
        Minimum confidence for association rules.  Same as Spark's
        ``minConfidence``.  Pass ``None`` (default) to skip rule generation.
    items_col:
        Name of the column that contains the item-set when the input is a
        *list-of-items* Spark/Pandas DataFrame (e.g. ``[["milk", "bread"]]``).
        Ignored when the input is a one-hot-encoded boolean DataFrame.
    use_colnames:
        Replace integer column indices with the actual column names in the
        output (same semantics as :func:`rusket.fpgrowth`).
    max_len:
        Maximum length of frequent itemsets to return.  ``None`` means no
        limit.

    Examples
    --------
    **Drop-in replacement for Spark MLlib:**

    .. code-block:: python

        from rusket import FPGrowth

        model = FPGrowth.from_spark(spark_df, min_support=0.3, min_confidence=0.6)
        freq  = model.freq_itemsets
        rules = model.association_rules_

    **Works with any supported input type:**

    .. code-block:: python

        model = FPGrowth(min_support=0.3).fit(pandas_df)
        model = FPGrowth(min_support=0.3).fit(polars_df)
        model = FPGrowth(min_support=0.3).fit(numpy_array)
    """

    def _mine(
        self,
        df: pd.DataFrame | pl.DataFrame | np.ndarray | Any,
    ) -> pd.DataFrame:
        return fpgrowth(
            df,
            min_support=self.min_support,
            use_colnames=self.use_colnames,
            max_len=self.max_len,
        )
