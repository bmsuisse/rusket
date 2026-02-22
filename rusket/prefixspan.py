"""Sequential Pattern Mining (PrefixSpan)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from . import _rusket as _rust  # type: ignore
from ._compat import to_dataframe
from .model import Miner


class PrefixSpan(Miner):
    """Sequential Pattern Mining (PrefixSpan) model.

    This class discovers frequent sequences of items across multiple users/sessions.
    """

    def __init__(
        self,
        data: list[list[int]],
        min_support: int,
        max_len: int | None = None,
        item_mapping: dict[int, Any] | None = None,
    ):
        """Initialize PrefixSpan with an already-formatted sequence list.

        Parameters
        ----------
        data : list of list of int
            A list of sequences, where each sequence is a list of integers representing items.
        min_support : int
            The minimum absolute support (number of sequences a pattern must appear in).
        max_len : int, optional
            The maximum length of the sequential patterns to mine.
        item_mapping : dict, optional
            A mapping from integer IDs back to original item names.
        """
        super().__init__(data, item_names=None)
        self.min_support = min_support
        self.max_len = max_len
        self.item_mapping = item_mapping

    @classmethod
    def from_transactions(
        cls,
        data: Any,
        transaction_col: str | None = None,
        item_col: str | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> PrefixSpan:
        """Initialize the PrefixSpan model from an event log DataFrame.

        Parameters
        ----------
        data : pd.DataFrame | pl.DataFrame | pyspark.sql.DataFrame
            Event log containing users, timestamps, and items.
        user_col : str
            Column name identifying the sequence (e.g., user_id or session_id).
        time_col : str
            Column name for ordering events.
        item_col : str
            Column name for the items.
        min_support : int
            The minimum absolute support required.
        max_len : int | None, default=None
            The maximum length of the sequential patterns to mine.
        """
        user_col = kwargs.get("user_col", transaction_col)
        if user_col is None:
            raise ValueError("user_col (or transaction_col) is required for PrefixSpan")

        time_col = kwargs.get("time_col")
        if time_col is None:
            raise ValueError("time_col is required for PrefixSpan")

        if item_col is None:
            raise ValueError("item_col is required for PrefixSpan")

        sequences, mapping = sequences_from_event_log(data, user_col, time_col, item_col)

        min_support = kwargs.get("min_support", 1)
        max_len = kwargs.get("max_len", None)

        return cls(sequences, min_support, max_len, mapping)

    def mine(self, **kwargs: Any) -> pd.DataFrame:
        """Mine sequential patterns using PrefixSpan.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing 'support' and 'sequence' columns.
            Sequences are mapped back to original item names if `from_transactions` was used.
        """
        # If already CSR-format from our internal loader:
        if isinstance(self.data, tuple) and len(self.data) == 2:
            indptr, indices = self.data
        else:
            from typing import cast

            # Fallback for manual user lists
            data_list = cast(list[list[int]], self.data)
            indptr = [0]
            indices = []
            for seq in data_list:
                indices.extend(seq)
                indptr.append(len(indices))

        supports, patterns = _rust.prefixspan_mine_py(indptr, indices, self.min_support, self.max_len)

        if self.item_mapping is not None:
            mapped_patterns = []
            for seq in patterns:
                mapped_patterns.append([self.item_mapping.get(idx, idx) for idx in seq])
            patterns = mapped_patterns

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
    import warnings

    warnings.warn(
        "rusket.prefixspan() is deprecated. Use PrefixSpan.from_transactions() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    indptr = [0]
    indices = []
    for seq in sequences:
        indices.extend(seq)
        indptr.append(len(indices))

    supports, patterns = _rust.prefixspan_mine_py(indptr, indices, min_support, max_len)

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
    """Convert an event log DataFrame into the sequence format required by PrefixSpan.

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
    tuple of (indptr, indices, item_mapping)
        - indptr: CSR-style index pointer list.
        - indices: Flattened item index list.
        - item_mapping: A dictionary mapping the integer IDs back to the original item labels.
    """
    import pandas as pd

    data = to_dataframe(df)

    try:
        import polars as pl

        is_polars = isinstance(data, pl.DataFrame)
    except ImportError:
        is_polars = False

    if is_polars:
        import polars as pl

        sorted_df = data.sort([user_col, time_col])
        unique_items = sorted_df[item_col].unique(maintain_order=True).to_list()

        item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        idx_to_item = dict(enumerate(unique_items))

        # Map to integer IDs and group
        mapped = sorted_df.with_columns(pl.col(item_col).replace(item_to_idx).cast(pl.Int64).alias("_mapped_items"))
        grouped = mapped.group_by(user_col, maintain_order=True).agg(pl.col("_mapped_items"))

        # Fast flat list representation via polars explode avoiding python objects
        indptr_series = grouped["_mapped_items"].list.len().cum_sum()
        indptr = [0] + indptr_series.to_list()
        indices = grouped["_mapped_items"].explode().to_list()

        return (indptr, indices), idx_to_item

    elif isinstance(data, pd.DataFrame):
        df_sorted = data.sort_values(by=[user_col, time_col])

        unique_items = df_sorted[item_col].unique()
        item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        idx_to_item = dict(enumerate(unique_items))

        mapped_items = df_sorted[item_col].map(lambda x: item_to_idx[x])
        grouped = mapped_items.groupby(df_sorted[user_col]).apply(list)

        sizes = grouped.apply(len)
        indptr = [0] + sizes.cumsum().tolist()
        indices = mapped_items.tolist()

        return (indptr, indices), idx_to_item

    else:
        raise TypeError(f"Expected Pandas, Polars, or PySpark DataFrame, got {type(data)}")
