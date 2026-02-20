"""Utilities for converting transactional data to one-hot boolean matrices."""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


def from_transactions(
    data: pd.DataFrame | pl.DataFrame | Sequence[Sequence[str | int]] | Any,
    transaction_col: str | None = None,
    item_col: str | None = None,
) -> pd.DataFrame:
    """Convert long-format transactional data to a one-hot boolean matrix.

    Real-world data typically looks like this::

        order_id  item
        1         bread
        1         butter
        1         milk
        2         bread
        2         eggs
        3         milk

    This function groups items by transaction and pivots them into the
    boolean matrix that :func:`rusket.fpgrowth` and :func:`rusket.eclat`
    expect (rows = transactions, columns = items, values = ``True``/``False``).

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

    Returns
    -------
    pd.DataFrame
        A boolean DataFrame ready for :func:`rusket.fpgrowth` or
        :func:`rusket.eclat`.  Column names correspond to the unique items.

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
    t = type(data).__name__

    # --- Spark DataFrame → Pandas ------------------------------------------
    if t == "DataFrame" and getattr(data, "__module__", "").startswith("pyspark"):
        data = typing.cast(Any, data).toPandas()
        t = "DataFrame"

    # --- Polars DataFrame → Pandas -----------------------------------------
    if t == "DataFrame" and getattr(data, "__module__", "").startswith("polars"):
        pl_df = typing.cast("pl.DataFrame", data)
        data = pl_df.to_pandas()
        t = "DataFrame"

    # --- List of lists -----------------------------------------------------
    if isinstance(data, (list, tuple)):
        return _from_list(data)

    # --- Pandas DataFrame --------------------------------------------------
    import pandas as _pd

    if not isinstance(data, _pd.DataFrame):
        raise TypeError(
            f"Expected a Pandas/Polars/Spark DataFrame or list of lists, "
            f"got {type(data)}"
        )

    return _from_dataframe(data, transaction_col, item_col)


# ---------------------------------------------------------------------------
# Explicit convenience wrappers
# ---------------------------------------------------------------------------


def from_pandas(
    df: pd.DataFrame,
    transaction_col: str | None = None,
    item_col: str | None = None,
) -> pd.DataFrame:
    """Convert a long-format Pandas DataFrame to a one-hot boolean matrix.

    Shorthand for ``from_transactions(df, transaction_col, item_col)``.

    Parameters
    ----------
    df
        A Pandas DataFrame with (at least) two columns: one for the
        transaction identifier and one for the item.
    transaction_col
        Name of the transaction-id column.  Defaults to the first column.
    item_col
        Name of the item column.  Defaults to the second column.

    Returns
    -------
    pd.DataFrame
        Boolean one-hot matrix.
    """
    return from_transactions(df, transaction_col=transaction_col, item_col=item_col)


def from_polars(
    df: pl.DataFrame,
    transaction_col: str | None = None,
    item_col: str | None = None,
) -> pd.DataFrame:
    """Convert a long-format Polars DataFrame to a one-hot boolean matrix.

    Shorthand for ``from_transactions(df, transaction_col, item_col)``.

    Parameters
    ----------
    df
        A Polars DataFrame with (at least) two columns: one for the
        transaction identifier and one for the item.
    transaction_col
        Name of the transaction-id column.  Defaults to the first column.
    item_col
        Name of the item column.  Defaults to the second column.

    Returns
    -------
    pd.DataFrame
        Boolean one-hot matrix.
    """
    return from_transactions(df, transaction_col=transaction_col, item_col=item_col)


def from_spark(
    df: Any,
    transaction_col: str | None = None,
    item_col: str | None = None,
) -> pd.DataFrame:
    """Convert a long-format Spark DataFrame to a one-hot boolean matrix.

    Calls ``.toPandas()`` internally and then pivots to one-hot format.

    Parameters
    ----------
    df
        A PySpark DataFrame with (at least) two columns: one for the
        transaction identifier and one for the item.
    transaction_col
        Name of the transaction-id column.  Defaults to the first column.
    item_col
        Name of the item column.  Defaults to the second column.

    Returns
    -------
    pd.DataFrame
        Boolean one-hot matrix.
    """
    return from_transactions(df, transaction_col=transaction_col, item_col=item_col)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _from_list(
    transactions: Sequence[Sequence[str | int]],
) -> pd.DataFrame:
    """Convert a list of item-lists into a sparse boolean DataFrame (CSR-backed).

    Produces a ``pd.SparseDtype("bool", fill_value=False)`` DataFrame so that
    even 100M+ transactions × 200k+ items fit in memory.
    """
    import pandas as pd
    import numpy as np
    from scipy import sparse as sp

    # Build item vocabulary
    all_items_set: set[str | int] = set()
    for txn in transactions:
        all_items_set.update(txn)
    all_items = sorted(all_items_set, key=lambda x: (isinstance(x, str), x))
    item_to_idx = {item: i for i, item in enumerate(all_items)}

    n_txn = len(transactions)
    n_items = len(all_items)

    # Build COO data for the sparse matrix
    row_idx: list[int] = []
    col_idx: list[int] = []
    for i, txn in enumerate(transactions):
        for item in txn:
            row_idx.append(i)
            col_idx.append(item_to_idx[item])

    data = np.ones(len(row_idx), dtype=bool)
    csr = sp.csr_matrix(
        (data, (np.array(row_idx, dtype=np.int64), np.array(col_idx, dtype=np.int64))),
        shape=(n_txn, n_items),
    )

    # Convert CSR → sparse pandas DataFrame
    return pd.DataFrame.sparse.from_spmatrix(
        csr,
        columns=[str(item) for item in all_items],
    ).astype(pd.SparseDtype("bool", fill_value=False))


def _from_dataframe(
    df: pd.DataFrame,
    transaction_col: str | None,
    item_col: str | None,
) -> pd.DataFrame:
    """Convert a 2-column long-format DataFrame to sparse one-hot boolean.

    Uses vectorised groupby + COO→CSR conversion — no Python loops over rows.
    Scales to 100M+ transactions × 200k+ items.
    """
    import numpy as np
    import pandas as pd
    from scipy import sparse as sp

    cols = list(df.columns)

    if len(cols) < 2:
        raise ValueError(
            f"DataFrame must have at least 2 columns (transaction id + item), "
            f"got {len(cols)}: {cols}"
        )

    txn_col = transaction_col or str(cols[0])
    itm_col = item_col or str(cols[1])

    if txn_col not in df.columns:
        raise ValueError(
            f"Transaction column '{txn_col}' not found. "
            f"Available columns: {cols}"
        )
    if itm_col not in df.columns:
        raise ValueError(
            f"Item column '{itm_col}' not found. "
            f"Available columns: {cols}"
        )

    # Encode transaction IDs and items as integers (no Python loops)
    txn_cat = pd.Categorical(df[txn_col])
    item_cat = pd.Categorical(df[itm_col])

    row_idx = txn_cat.codes.astype(np.int64)
    col_idx = item_cat.codes.astype(np.int64)
    n_txn = len(txn_cat.categories)
    n_items = len(item_cat.categories)

    data = np.ones(len(row_idx), dtype=bool)
    csr = sp.csr_matrix((data, (row_idx, col_idx)), shape=(n_txn, n_items))

    # Build sparse DataFrame
    item_names = [str(c) for c in item_cat.categories]
    return pd.DataFrame.sparse.from_spmatrix(
        csr,
        columns=item_names,
    ).astype(pd.SparseDtype("bool", fill_value=False))

