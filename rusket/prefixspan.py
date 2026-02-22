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
    df: pd.DataFrame,
    user_col: str,
    time_col: str,
    item_col: str,
) -> tuple[list[list[int]], dict[int, Any]]:
    """Helper to convert an event log DataFrame into the sequence format required by PrefixSpan.

    Parameters
    ----------
    df : pd.DataFrame
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
    df_sorted = df.sort_values(by=[user_col, time_col])

    # Map items to integers if they aren't already
    unique_items = df_sorted[item_col].unique()
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    idx_to_item = {idx: item for idx, item in enumerate(unique_items)}

    # Map column to indices using .map with a callable
    mapped_items = df_sorted[item_col].map(lambda x: item_to_idx[x])

    # Group by user and collect sequences
    # We use mapped_items to ensure we group the integer IDs.
    grouped = mapped_items.groupby(df_sorted[user_col]).apply(list)

    # Convert Pandas Series of lists to a regular Python list of lists
    sequences = grouped.tolist()

    return sequences, idx_to_item
