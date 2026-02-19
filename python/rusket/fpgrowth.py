"""FP-Growth frequent itemset mining backed by Rust + PyO3.

Dispatches to the optimal Rust entry-point depending on input type:
  - Dense pandas DataFrame  → fpgrowth_from_dense (flat u8 buffer, no Python loops)
  - Sparse pandas DataFrame → fpgrowth_from_csr   (raw CSR arrays, no densification)
  - Polars DataFrame        → fpgrowth_from_dense (via to_numpy, Arrow zero-copy)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from ._validation import valid_input_check
from . import _rusket as _rust  # type: ignore[import-untyped, import-not-found]

if TYPE_CHECKING:
    import polars as pl


def fpgrowth(
    df: "pd.DataFrame | pl.DataFrame",
    min_support: float = 0.5,
    null_values: bool = False,
    use_colnames: bool = False,
    max_len: Optional[int] = None,
    verbose: int = 0,
) -> pd.DataFrame:
    """Find frequent itemsets using the FP-Growth algorithm (Rust-backed).

    Parameters
    ----------
    df:
        One-hot encoded DataFrame.  Accepted types:
        * :class:`pandas.DataFrame` (dense or sparse, bool/int 0-1 values)
        * :class:`polars.DataFrame` (bool/int 0-1 values)
    min_support:
        Minimum support threshold in ``(0, 1]``.
    null_values:
        Allow NaN values in *df* (pandas only).
    use_colnames:
        If ``True``, itemsets contain column names instead of column indices.
    max_len:
        Maximum itemset length.  ``None`` means unlimited.
    verbose:
        Verbosity level (currently unused, kept for API compatibility).

    Returns
    -------
    pandas.DataFrame
        Columns ``['support', 'itemsets']``.  Each itemset is a
        :class:`frozenset` of column indices (or names when *use_colnames*).
    """
    # ------------------------------------------------------------------ #
    # Spark branch — collect to pandas on the driver, then mine in Rust  #
    # ------------------------------------------------------------------ #
    try:
        from pyspark.sql import DataFrame as SparkDataFrame  # type: ignore[import]
        if isinstance(df, SparkDataFrame):
            import warnings
            warnings.warn(
                "Spark DataFrames are collected to the driver before mining. "
                "Make sure the dataset fits in local memory.",
                stacklevel=2,
            )
            df = df.toPandas()  # type: ignore[attr-defined]
            # fall through to the pandas branch below
    except ImportError:
        pass

    # ------------------------------------------------------------------ #
    # Polars branch — convert to numpy once, all mining happens in Rust  #
    # ------------------------------------------------------------------ #
    try:
        import polars as pl  # type: ignore[import]
        if isinstance(df, pl.DataFrame):
            return _fpgrowth_polars(df, min_support, use_colnames, max_len)
    except ImportError:
        pass

    # ------------------------------------------------------------------ #
    # Numpy ndarray branch — direct 2-D boolean/uint8 array              #
    # ------------------------------------------------------------------ #
    if isinstance(df, np.ndarray):
        if df.ndim != 2:
            raise ValueError(f"numpy array must be 2-D, got shape {df.shape}")
        n_rows, n_cols = df.shape
        min_count = math.ceil(min_support * n_rows)
        col_names = [str(i) for i in range(n_cols)]
        data = np.ascontiguousarray(df, dtype=np.uint8)
        raw = _rust.fpgrowth_from_dense(data, min_count, max_len)
        return _build_result(raw, n_rows, min_support, col_names, use_colnames)

    # ------------------------------------------------------------------ #
    # Pandas branch                                                       #
    # ------------------------------------------------------------------ #
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, polars DataFrame, numpy array, "
            f"or Spark DataFrame — got {type(df)}"
        )

    valid_input_check(df, null_values)

    if min_support <= 0.0:
        raise ValueError(
            "`min_support` must be a positive "
            "number within the interval `(0, 1]`. "
            "Got %s." % min_support
        )

    n_rows, n_cols = df.shape
    min_count = math.ceil(min_support * n_rows)
    col_names = list(df.columns)

    if hasattr(df, "sparse"):
        raw = _fpgrowth_sparse_pandas(df, n_rows, n_cols, min_count, max_len)
    else:
        raw = _fpgrowth_dense_pandas(df, n_rows, n_cols, min_count, max_len)

    return _build_result(raw, n_rows, min_support, col_names, use_colnames)


# ------------------------------------------------------------------ #
# Private helpers — each is a thin extraction + Rust call
# ------------------------------------------------------------------ #

def _fpgrowth_dense_pandas(
    df: pd.DataFrame,
    n_rows: int,
    n_cols: int,
    min_count: int,
    max_len: Optional[int],
) -> list[tuple[int, list[int]]]:
    """Dense pandas path: pass 2D numpy uint8 array directly — zero-copy in Rust."""
    # np.ascontiguousarray guarantees C-order; astype(uint8) is zero-copy for bool arrays.
    data: np.ndarray = np.ascontiguousarray(df.values, dtype=np.uint8)
    return _rust.fpgrowth_from_dense(data, min_count, max_len)


def _fpgrowth_sparse_pandas(
    df: pd.DataFrame,
    n_rows: int,
    n_cols: int,
    min_count: int,
    max_len: Optional[int],
) -> list[tuple[int, list[int]]]:
    """Sparse pandas path: pass int32 CSR numpy arrays directly — zero-copy in Rust."""
    csr = df.sparse.to_coo().tocsr()
    csr.eliminate_zeros()
    # Rust accepts PyReadonlyArray1<i32> — scipy CSR uses int32 by default.
    indptr = np.asarray(csr.indptr, dtype=np.int32)
    indices = np.asarray(csr.indices, dtype=np.int32)
    return _rust.fpgrowth_from_csr(indptr, indices, n_cols, min_count, max_len)


def _fpgrowth_polars(
    df: "pl.DataFrame",
    min_support: float,
    use_colnames: bool,
    max_len: Optional[int],
) -> pd.DataFrame:
    """Polars path: Arrow-backed numpy buffer → Rust dense path."""
    n_rows, n_cols = df.shape
    if min_support <= 0.0:
        raise ValueError(
            "`min_support` must be a positive "
            "number within the interval `(0, 1]`. "
            "Got %s." % min_support
        )
    min_count = math.ceil(min_support * n_rows)
    col_names = df.columns

    # to_numpy() uses Arrow memory directly — zero-copy for numeric dtypes.
    arr: np.ndarray = np.ascontiguousarray(df.to_numpy(), dtype=np.uint8)
    raw = _rust.fpgrowth_from_dense(arr, min_count, max_len)
    return _build_result(raw, n_rows, min_support, col_names, use_colnames)


def _build_result(
    raw: list[tuple[int, list[int]]],
    n_rows: int,
    min_support: float,
    col_names: list,
    use_colnames: bool,
) -> pd.DataFrame:
    """Convert Rust output to a pandas DataFrame with frozenset itemsets."""
    if not raw:
        return pd.DataFrame(columns=["support", "itemsets"])

    supports = [count / n_rows for count, _ in raw]
    itemsets: list[frozenset] = [frozenset(iset) for _, iset in raw]

    if use_colnames:
        col_map = {idx: col for idx, col in enumerate(col_names)}
        itemsets = [frozenset(col_map[i] for i in iset) for iset in itemsets]

    result = pd.DataFrame({"support": supports, "itemsets": itemsets})
    result = result[result["support"] >= min_support].reset_index(drop=True)
    return result  # type: ignore[return-value]
