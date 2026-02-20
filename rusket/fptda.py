"""FP-TDA: functional API and FPTda estimator class."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from . import _rusket as _rust  # type: ignore
from ._fpbase import FPBase
from .fpgrowth import _build_result  # reuse the same result builder

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import polars as pl

    class _SparkDataFrame:
        def toPandas(self) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------
# Module-level functional API (mirrors rusket.fpgrowth)
# ---------------------------------------------------------------------------


def fptda(
    df: pd.DataFrame | pl.DataFrame | np.ndarray | Any,
    min_support: float = 0.5,
    null_values: bool = False,
    use_colnames: bool = False,
    max_len: int | None = None,
    verbose: int = 0,
) -> pd.DataFrame:
    """Mine frequent itemsets with the FP-TDA algorithm.

    FP-TDA (Frequent-Pattern Two-Dimensional Array) replaces the recursive
    conditional-subtree construction of FP-Growth with a right-to-left
    column projection on sorted transaction lists.  It produces the **same
    result** as :func:`rusket.fpgrowth` but with a different internal
    algorithm.

    Parameters and return value are identical to :func:`rusket.fpgrowth`.
    """
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
        return _fptda_polars(
            typing.cast("pl.DataFrame", df), min_support, use_colnames, max_len
        )

    if t == "ndarray":
        import numpy as np

        df_nd = typing.cast("np.ndarray", df)
        n_rows, n_cols = df_nd.shape
        min_count = math.ceil(min_support * n_rows)
        col_names = [str(i) for i in range(n_cols)]
        data = np.ascontiguousarray(df_nd, dtype=np.uint8)
        raw = _rust.fptda_from_dense(data, min_count, max_len)
        return _build_result(raw, n_rows, min_support, col_names, use_colnames)

    df_pd = typing.cast("pd.DataFrame", df)
    n_rows, n_cols = df_pd.shape
    min_count = math.ceil(min_support * n_rows)
    col_names = list(df_pd.columns)

    from ._validation import valid_input_check

    valid_input_check(df_pd, null_values)

    if hasattr(df_pd, "sparse"):
        raw = _fptda_sparse_pandas(df_pd, n_rows, n_cols, min_count, max_len)
    else:
        raw = _fptda_dense_pandas(df_pd, n_rows, n_cols, min_count, max_len)

    return _build_result(raw, n_rows, min_support, col_names, use_colnames)


def _fptda_dense_pandas(
    df: pd.DataFrame, n_rows: int, n_cols: int, min_count: int, max_len: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np

    data = np.ascontiguousarray(df.values, dtype=np.uint8)
    return _rust.fptda_from_dense(data, min_count, max_len)


def _fptda_sparse_pandas(
    df: pd.DataFrame, n_rows: int, n_cols: int, min_count: int, max_len: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np

    csr = df.sparse.to_coo().tocsr()
    csr.eliminate_zeros()
    indptr = np.asarray(csr.indptr, dtype=np.int32)
    indices = np.asarray(csr.indices, dtype=np.int32)
    return _rust.fptda_from_csr(indptr, indices, n_cols, min_count, max_len)


def _fptda_polars(
    df: pl.DataFrame, min_support: float, use_colnames: bool, max_len: int | None
) -> pd.DataFrame:
    import numpy as np

    n_rows, n_cols = df.shape
    min_count = math.ceil(min_support * n_rows)
    arr = np.ascontiguousarray(df.to_numpy(), dtype=np.uint8)
    raw = _rust.fptda_from_dense(arr, min_count, max_len)
    return _build_result(raw, n_rows, min_support, df.columns, use_colnames)


# ---------------------------------------------------------------------------
# OOP estimator — inherits all shared logic from FPBase
# ---------------------------------------------------------------------------


class FPTda(FPBase):
    """FP-TDA estimator — same API as :class:`rusket.FPGrowth`.

    Uses the Frequent-Pattern Two-Dimensional Array algorithm (IJISRT25NOV1256)
    instead of the standard FP-Growth tree construction.  Both produce identical
    results; FP-TDA trades tree memory for recursive column projections.

    Parameters
    ----------
    min_support, min_confidence, items_col, use_colnames, max_len:
        Same semantics as :class:`rusket.FPGrowth`.

    Examples
    --------
    .. code-block:: python

        from rusket import FPTda

        model = FPTda(min_support=0.3, min_confidence=0.6).fit(df)
        freq  = model.freq_itemsets
        rules = model.association_rules_

        # Spark drop-in:
        model = FPTda.from_spark(spark_df, min_support=0.3)

        # Polars:
        model = FPTda(min_support=0.3).fit(polars_df)
    """

    def _mine(
        self,
        df: pd.DataFrame | pl.DataFrame | np.ndarray | Any,
    ) -> pd.DataFrame:
        return fptda(
            df,
            min_support=self.min_support,
            use_colnames=self.use_colnames,
            max_len=self.max_len,
        )
