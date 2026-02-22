"""Sequential Pattern Mining (PrefixSpan)."""

from __future__ import annotations
import pandas as pd
from typing import Any

from . import _rusket as _rust  # type: ignore


def prefixspan(
    sequences: list[list[int]],
    min_support: int,
    max_len: int | None = None,
) -> pd.DataFrame:
    """Mine sequential patterns using the PrefixSpan algorithm.

    This function discovers frequent sequences of items across multiple users/sessions.
    Currently, this assumes sequences where each event consists of a single item
    (e.g., a sequence of page views or a sequence of individual products bought over time).

    Parameters
    ----------
    sequences : list of list of int
        A list of sequences, where each sequence is a list of integers representing items.
        Example: `[[1, 2, 3], [1, 3], [2, 3]]`.
    min_support : int
        The minimum absolute support (number of sequences a pattern must appear in).
    max_len : int, optional
        The maximum length of the sequential patterns to mine.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing 'support' and 'sequence' columns.
    """
    supports, patterns = _rust.prefixspan_mine_py(sequences, min_support, max_len)

    return (
        pd.DataFrame(
            {
                "support": supports,
                "sequence": patterns,
            }
        )
        .sort_values(by="support", ascending=False)
        .reset_index(drop=True)
    )


def sequences_from_event_log(
    df: Any,
    user_col: str,
    time_col: str,
    item_col: str,
) -> tuple[list[list[int]], dict[int, Any]]:
    """Helper to convert an event log DataFrame into the sequence format required by PrefixSpan.

    Accepts Pandas, Polars, or PySpark DataFrames. Data is grouped by `user_col`,
    ordered by `time_col`, and `item_col` values are collected into sequences.

    Parameters
    ----------
    df : pd.DataFrame | pl.DataFrame | pyspark.sql.DataFrame
        Event log containing users, timestamps, and items.
    user_col : str
        Column name identifying the sequence (e.g., user_id or session_id).
    time_col : str
        Column name for ordering events.
    item_col : str
        Column name for the items.

    Returns
    -------
    tuple of (sequences, item_mapping)
        - sequences: The nested list of integers to pass to `prefixspan()`.
        - item_mapping: A dictionary mapping the integer IDs back to the original item labels.
    """
    from ._compat import to_dataframe
    import pandas as pd

    data = to_dataframe(df)

    try:
        import polars as pl

        is_polars = isinstance(data, pl.DataFrame)
    except ImportError:
        is_polars = False

    if is_polars:
        sorted_df = data.sort([user_col, time_col])
        unique_items = sorted_df[item_col].unique(maintain_order=True).to_list()
        
        item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        idx_to_item = {idx: item for idx, item in enumerate(unique_items)}

        # Map to integer IDs and group
        mapped = sorted_df.with_columns(
            pl.col(item_col).replace(item_to_idx).cast(pl.Int64).alias("_mapped_items")
        )
        grouped = mapped.group_by(user_col, maintain_order=True).agg(pl.col("_mapped_items"))
        
        # Rust pyo3 requires explicit list[list[int]], so we map elements natively
        sequences = [[int(x) for x in seq] for seq in grouped["_mapped_items"].to_list()]
        return sequences, idx_to_item

    elif isinstance(data, pd.DataFrame):
        df_sorted = data.sort_values(by=[user_col, time_col])

        unique_items = df_sorted[item_col].unique()
        item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        idx_to_item = {idx: item for idx, item in enumerate(unique_items)}

        mapped_items = df_sorted[item_col].map(lambda x: item_to_idx[x])
        grouped = mapped_items.groupby(df_sorted[user_col]).apply(list)
        
        sequences = grouped.tolist()
        return sequences, idx_to_item

    else:
        raise TypeError(f"Expected Pandas, Polars, or PySpark DataFrame, got {type(data)}")
