from __future__ import annotations

import time
import typing
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, overload

from ._compat import to_dataframe

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from pyspark.sql import DataFrame as SparkDataFrame

    from ._compat import DataFrame

# ---------------------------------------------------------------------------
# Overloaded signatures — tell pyright/mypy the exact return type per input
# ---------------------------------------------------------------------------


@overload
def from_transactions(
    data: pd.DataFrame,
    transaction_col: str | None = None,
    item_col: str | None = None,
    min_item_count: int = 1,
    verbose: int = 0,
) -> pd.DataFrame: ...


@overload
def from_transactions(
    data: pl.DataFrame,
    transaction_col: str | None = None,
    item_col: str | None = None,
    min_item_count: int = 1,
    verbose: int = 0,
) -> pl.DataFrame: ...


@overload
def from_transactions(
    data: SparkDataFrame,
    transaction_col: str | None = None,
    item_col: str | None = None,
    min_item_count: int = 1,
    verbose: int = 0,
) -> SparkDataFrame: ...


@overload
def from_transactions(
    data: pa.Table,
    transaction_col: str | None = None,
    item_col: str | None = None,
    min_item_count: int = 1,
    verbose: int = 0,
) -> pa.Table: ...


@overload
def from_transactions(
    data: Sequence[Sequence[str | int]],
    transaction_col: str | None = None,
    item_col: str | None = None,
    min_item_count: int = 1,
    verbose: int = 0,
) -> pd.DataFrame: ...


def from_transactions(
    data: DataFrame | Sequence[Sequence[str | int]] | Any,
    transaction_col: str | None = None,
    item_col: str | None = None,
    min_item_count: int = 1,
    verbose: int = 0,
) -> Any:
    """Convert long-format transactional data to a one-hot boolean matrix.

    The return type mirrors the input type:

    - **Polars** ``DataFrame`` → **Polars** ``DataFrame``
    - **Pandas** ``DataFrame`` → **Pandas** ``DataFrame``
    - **Spark** ``DataFrame``  → **Spark**  ``DataFrame``
    - ``list[list[...]]``      → **Pandas** ``DataFrame``

    Parameters
    ----------
    data
        One of:

        - **Pandas / Polars / Spark DataFrame** with (at least) two columns:
          one for the transaction identifier and one for the item.
        - **List of lists** where each inner list contains the items of a
          single transaction, e.g. ``[["bread", "milk"], ["bread", "eggs"]]``.

    transaction_col
        Name of the column that identifies transactions.  If ``None`` the
        first column is used.  Ignored for list-of-lists input.

    item_col
        Name of the column that contains item values.  If ``None`` the
        second column is used.  Ignored for list-of-lists input.

    min_item_count
        Minimum number of times an item must appear to be included in the
        resulting one-hot-encoded matrix. Default is 1.

    Returns
    -------
    DataFrame
        A boolean DataFrame (same type as input) ready for
        :func:`rusket.fpgrowth` or :func:`rusket.eclat`.
        Column names correspond to the unique items.

    Examples
    --------
    >>> import rusket
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "order_id": [1, 1, 1, 2, 2, 3],
    ...     "item": [3, 4, 5, 3, 5, 8],
    ... })
    >>> ohe = rusket.from_transactions(df)
    >>> freq = rusket.fpgrowth(ohe, min_support=0.5, use_colnames=True)
    """
    # --- PyArrow Table — zero-copy round-trip ---
    _type = type(data)
    _type_name = _type.__name__
    _mod = getattr(_type, "__module__", "") or ""
    _is_arrow = _type_name == "Table" and _mod.startswith("pyarrow")

    if _is_arrow:
        import pandas as _pd
        import polars as _pl
        import pyarrow as _pa

        pl_df = _pl.from_arrow(typing.cast("_pa.Table", data))
        pandas_df = typing.cast("_pd.DataFrame", pl_df.to_pandas())
        result_pd = _from_dataframe(pandas_df, transaction_col, item_col, min_item_count=min_item_count, verbose=verbose)
        return _pa.Table.from_pandas(result_pd.astype(bool))

    # --- Spark Detection MUST happen before coercion to Polars in to_dataframe() ---
    _is_spark = _type_name == "DataFrame" and _mod.startswith("pyspark")

    if _is_spark:
        if min_item_count > 1:
            if verbose:
                print(f"[{time.strftime('%X')}] Spark: Filtering items with < {min_item_count} occurrences...")
            import pyspark.sql.functions as F

            spark_df = typing.cast("SparkDataFrame", data)
            _itm_c = item_col or spark_df.columns[1]
            item_counts = spark_df.groupBy(_itm_c).count()
            valid_items = item_counts.filter(F.col("count") >= min_item_count).select(_itm_c)
            data = spark_df.join(valid_items, on=_itm_c, how="inner").select(*spark_df.columns)

        if hasattr(data, "toArrow"):
            import pandas as pd

            pandas_df = data.toArrow().to_pandas(types_mapper=pd.ArrowDtype)  # type: ignore[union-attr]
        else:
            pandas_df = data.toPandas()  # type: ignore[union-attr]

        result_pd = _from_dataframe(pandas_df, transaction_col, item_col, min_item_count=1, verbose=verbose)
        # Convert back via Arrow for efficiency
        import pyarrow as _pa

        spark = data.sparkSession  # type: ignore[union-attr]
        arrow_table = _pa.Table.from_pandas(result_pd.astype(bool))
        return spark.createDataFrame(arrow_table)

    # Standard coercion (coerces Spark to Polars, but we bypassed it above)
    data = to_dataframe(data)

    if isinstance(data, (list, tuple)):
        return _from_list(data, min_item_count=min_item_count, verbose=verbose)

    import pandas as _pd
    import polars as _pl

    # --- Polars ---
    if isinstance(data, _pl.DataFrame):
        pandas_df = data.to_pandas()
        result_pd = _from_dataframe(
            pandas_df, transaction_col, item_col, min_item_count=min_item_count, verbose=verbose
        )
        return _pl.from_pandas(result_pd.astype(bool))

    # --- Pandas ---
    if isinstance(data, _pd.DataFrame):
        return _from_dataframe(data, transaction_col, item_col, min_item_count=min_item_count, verbose=verbose)

    raise TypeError(f"Expected a Pandas/Polars/Spark/PyArrow DataFrame or list of lists, got {type(data)}")


def from_pandas(
    df: pd.DataFrame,
    transaction_col: str | None = None,
    item_col: str | None = None,
    min_item_count: int = 1,
    verbose: int = 0,
) -> pd.DataFrame:
    """Shorthand for ``from_transactions(df, transaction_col, item_col)``."""
    return from_transactions(
        df, transaction_col=transaction_col, item_col=item_col, min_item_count=min_item_count, verbose=verbose
    )


def from_polars(
    df: pl.DataFrame,
    transaction_col: str | None = None,
    item_col: str | None = None,
    min_item_count: int = 1,
    verbose: int = 0,
) -> pl.DataFrame:
    """Shorthand for ``from_transactions(df, transaction_col, item_col)``."""
    return from_transactions(
        df, transaction_col=transaction_col, item_col=item_col, min_item_count=min_item_count, verbose=verbose
    )


def from_spark(
    df: SparkDataFrame,
    transaction_col: str | None = None,
    item_col: str | None = None,
    min_item_count: int = 1,
    verbose: int = 0,
) -> SparkDataFrame:
    """Shorthand for ``from_transactions(df, transaction_col, item_col)``."""
    return from_transactions(df, transaction_col=transaction_col, item_col=item_col, min_item_count=min_item_count)


def from_arrow(
    table: pa.Table,
    transaction_col: str | None = None,
    item_col: str | None = None,
    min_item_count: int = 1,
    verbose: int = 0,
) -> pa.Table:
    """Convert a PyArrow Table in long format to a one-hot boolean PyArrow Table.

    This is a zero-copy-friendly shorthand for ``from_transactions(table, ...)``.  The
    input table must have at least two columns: one for the transaction identifier and
    one for the item.  The returned table has boolean columns (one per unique item).

    Parameters
    ----------
    table
        A ``pyarrow.Table`` with at least two columns (transaction id + item).
    transaction_col
        Name of the transaction-id column. Defaults to the first column.
    item_col
        Name of the item column. Defaults to the second column.
    min_item_count
        Minimum occurrences for an item to be included. Default is 1.
    verbose
        Verbosity level.

    Returns
    -------
    pyarrow.Table
        A boolean Table ready for :func:`rusket.fpgrowth` / :func:`rusket.eclat`.
    """
    return from_transactions(
        table,
        transaction_col=transaction_col,
        item_col=item_col,
        min_item_count=min_item_count,
        verbose=verbose,
    )


def _from_list(
    transactions: Sequence[Sequence[str | int]],
    min_item_count: int = 1,
    verbose: int = 0,
) -> pd.DataFrame:
    import numpy as np
    import pandas as pd
    from scipy import sparse as sp

    t0 = 0.0
    if verbose:
        print(f"[{time.strftime('%X')}] Extracting unique items from list of lists...")
        t0 = time.perf_counter()

    if min_item_count > 1:
        from collections import Counter

        counts = Counter()  # type: ignore[var-annotated]
        for txn in transactions:
            counts.update(txn)
        all_items_set = {item for item, count in counts.items() if count >= min_item_count}
    else:
        all_items_set_any: set[Any] = set()
        for txn in transactions:
            all_items_set_any.update(txn)
        all_items_set = all_items_set_any

    all_items = sorted(all_items_set, key=lambda x: (isinstance(x, str), x))
    item_to_idx = {item: i for i, item in enumerate(all_items)}

    n_txn = len(transactions)
    n_items = len(all_items)

    if verbose:
        print(f"[{time.strftime('%X')}] Found {n_items:,} unique items. Building COO coordinates...")

    row_idx: list[int] = []
    col_idx: list[int] = []
    iterator = enumerate(transactions)
    if verbose:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, total=n_txn, desc="Transactions")
        except ImportError:
            pass

    for i, txn in iterator:
        for item in txn:
            if item in item_to_idx:
                row_idx.append(i)
                col_idx.append(item_to_idx[item])

    data = np.ones(len(row_idx), dtype=bool)
    csr = sp.csr_matrix(
        (data, (np.array(row_idx, dtype=np.int64), np.array(col_idx, dtype=np.int64))),
        shape=(n_txn, n_items),
    )

    sparse_df = pd.DataFrame.sparse.from_spmatrix(
        csr,
        columns=[str(item) for item in all_items],
    ).astype(pd.SparseDtype("bool", fill_value=False))

    if verbose:
        print(f"[{time.strftime('%X')}] CSR generation completed in {time.perf_counter() - t0:.2f}s.")

    return sparse_df


def _from_dataframe(
    df: pd.DataFrame,
    transaction_col: str | None,
    item_col: str | None,
    min_item_count: int = 1,
    verbose: int = 0,
) -> pd.DataFrame:
    import numpy as np
    import pandas as pd
    from scipy import sparse as sp

    t0 = 0.0
    if verbose:
        print(f"[{time.strftime('%X')}] Parameterizing from DataFrame (shape={df.shape})...")
        t0 = time.perf_counter()

    cols = list(df.columns)

    if len(cols) < 2:
        raise ValueError(f"DataFrame must have at least 2 columns (transaction id + item), got {len(cols)}: {cols}")

    txn_col = transaction_col or str(cols[0])
    itm_col = item_col or str(cols[1])

    if txn_col not in df.columns:
        raise ValueError(f"Transaction column '{txn_col}' not found. Available columns: {cols}")
    if itm_col not in df.columns:
        raise ValueError(f"Item column '{itm_col}' not found. Available columns: {cols}")

    if min_item_count > 1:
        if verbose:
            print(f"[{time.strftime('%X')}] Pandas: Filtering items with < {min_item_count} occurrences...")
        raw_counts = df[itm_col].value_counts()
        counts: Any = raw_counts
        valid_items_idx = typing.cast("pd.Index", counts[counts >= min_item_count].index)
        df = df.loc[df[itm_col].isin(valid_items_idx)]

    txn_codes, _txn_uniques = pd.factorize(df[txn_col], sort=False)
    item_codes, item_uniques = pd.factorize(df[itm_col], sort=True)

    n_txn = int(txn_codes.max()) + 1
    n_items = len(item_uniques)

    data = np.ones(len(txn_codes), dtype=np.int8)
    csr = sp.csr_matrix(
        (data, (txn_codes.astype(np.int64), item_codes.astype(np.int64))),
        shape=(n_txn, n_items),
    )
    csr.data = np.minimum(csr.data, 1)

    item_names = [str(c) for c in item_uniques]
    res = pd.DataFrame.sparse.from_spmatrix(
        csr,
        columns=item_names,
    ).astype(pd.SparseDtype("bool", fill_value=False))

    if verbose:
        print(f"[{time.strftime('%X')}] DataFrame parameterization completed in {time.perf_counter() - t0:.2f}s.")

    return res


def from_transactions_csr(
    data: DataFrame | str | Any,
    transaction_col: str | None = None,
    item_col: str | None = None,
    chunk_size: int = 10_000_000,
) -> tuple[Any, list[str]]:
    """Convert long-format transactional data to a CSR matrix + column names.

    Unlike :func:`from_transactions`, this returns a raw
    ``scipy.sparse.csr_matrix`` that can be passed directly to
    :func:`rusket.fpgrowth` or :func:`rusket.eclat` — **no pandas overhead**.

    For billion-row datasets, this processes data in chunks of ``chunk_size``
    rows, keeping peak memory to one chunk + the running CSR.

    Parameters
    ----------
    data
        One of:

        - **Pandas DataFrame** with (at least) two columns.
        - **Polars DataFrame** or **Spark DataFrame** (converted internally).
        - **File path** (str / Path) to a Parquet file — read in chunks.

    transaction_col
        Name of the transaction-id column. Defaults to the first column.

    item_col
        Name of the item column. Defaults to the second column.

    chunk_size
        Number of rows per chunk. Lower values use less memory.
        Default: 10 million rows.

    Returns
    -------
    tuple[scipy.sparse.csr_matrix, list[str]]
        A CSR matrix and the list of column (item) names.  Pass directly::

            csr, names = from_transactions_csr(df)
            freq = fpgrowth(csr, min_support=0.001,
                            use_colnames=True, column_names=names)

    Examples
    --------
    >>> import rusket
    >>> csr, names = rusket.from_transactions_csr("orders.parquet")
    >>> freq = rusket.fpgrowth(csr, min_support=0.001,
    ...                        use_colnames=True, column_names=names)
    """
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from scipy import sparse as sp

    data = to_dataframe(data)

    if isinstance(data, (str, Path)):
        return _from_parquet_csr(str(data), transaction_col, item_col, chunk_size)

    if type(data).__name__ == "DataFrame" and getattr(data, "__module__", "").startswith("polars"):
        data = data.to_pandas()  # type: ignore[union-attr]

    df = typing.cast("pd.DataFrame", data)
    cols = list(df.columns)
    txn_col = transaction_col or str(cols[0])
    itm_col = item_col or str(cols[1])

    item_codes_global, item_uniques = pd.factorize(df[itm_col], sort=True)
    n_items = len(item_uniques)
    item_names = [str(c) for c in item_uniques]

    n = len(df)
    csr_parts: list[sp.csr_matrix] = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_txn = df[txn_col].iloc[start:end].values
        chunk_item_codes = item_codes_global[start:end]

        local_codes, _local_uniques = pd.factorize(chunk_txn, sort=False)
        n_txn_chunk = int(local_codes.max()) + 1 if len(local_codes) > 0 else 0

        data_arr = np.ones(len(local_codes), dtype=np.int8)
        chunk_csr = sp.csr_matrix(
            (
                data_arr,
                (local_codes.astype(np.int64), chunk_item_codes.astype(np.int64)),
            ),
            shape=(n_txn_chunk, n_items),
        )
        chunk_csr.data = np.minimum(chunk_csr.data, 1)
        csr_parts.append(chunk_csr)

    final_csr = sp.vstack(csr_parts, format="csr")
    return final_csr, item_names


def _from_parquet_csr(
    path: str,
    transaction_col: str | None,
    item_col: str | None,
    chunk_size: int,
) -> tuple[Any, list[str]]:
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from scipy import sparse as sp

    table = pq.read_table(path)
    cols = table.column_names
    txn_col_name = transaction_col or cols[0]
    itm_col_name = item_col or cols[1]

    item_series = table.column(itm_col_name).to_pandas()
    _item_codes_all, item_uniques = pd.factorize(item_series, sort=True)
    n_items = len(item_uniques)
    item_to_idx = {v: i for i, v in enumerate(item_uniques)}
    item_names = [str(c) for c in item_uniques]
    del item_series

    df_full = table.select([txn_col_name, itm_col_name]).to_pandas()
    del table

    n = len(df_full)
    csr_parts: list[sp.csr_matrix] = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = df_full.iloc[start:end]

        txn_codes, _txn_uniques = pd.factorize(chunk[txn_col_name], sort=False)
        item_codes = chunk[itm_col_name].map(item_to_idx).values
        n_txn_chunk = int(txn_codes.max()) + 1 if len(txn_codes) > 0 else 0

        data_arr = np.ones(len(txn_codes), dtype=np.int8)
        chunk_csr = sp.csr_matrix(
            (data_arr, (txn_codes.astype(np.int64), item_codes.astype(np.int64))),
            shape=(n_txn_chunk, n_items),
        )
        chunk_csr.data = np.minimum(chunk_csr.data, 1)
        csr_parts.append(chunk_csr)

    final_csr = sp.vstack(csr_parts, format="csr")
    return final_csr, item_names
