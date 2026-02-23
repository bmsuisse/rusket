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
        min_support: int | float,
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
        min_supp = kwargs.get("min_support", self.min_support)
        max_len = kwargs.get("max_len", self.max_len)

        if isinstance(min_supp, float):
            n_seqs = min(1, len(self.data)) if isinstance(self.data, list) else 1  # simplistic fallback
            if isinstance(self.data, tuple) and len(self.data) == 2:
                n_seqs = len(self.data[0]) - 1  # len of indptr - 1 is number of sequences
            elif isinstance(self.data, list):
                n_seqs = len(self.data)

            min_supp = max(1, int(min_supp * n_seqs))

        # If already CSR-format from our internal loader:
        if isinstance(self.data, tuple) and len(self.data) == 2:
            indptr, indices = self.data
        else:
            from typing import cast

            import numpy as np

            # Fallback for manual user lists
            data_list = cast(list[list[int]], self.data)
            indptr_list = [0]
            indices_list = []
            for seq in data_list:
                indices_list.extend(seq)
                indptr_list.append(len(indices_list))

            indptr = np.array(indptr_list, dtype=np.uintp)
            indices = np.array(indices_list, dtype=np.uint32)

        supports, patterns = _rust.prefixspan_mine_py(indptr, indices, int(min_supp), max_len)  # type: ignore

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

    @classmethod
    def mine_grouped(
        cls,
        df: Any,
        group_col: str,
        user_col: str,
        time_col: str,
        item_col: str,
        min_support: int = 1,
        max_len: int | None = None,
    ) -> Any:
        """Distribute Sequential Pattern Mining (PrefixSpan) across PySpark partitions.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            The input PySpark DataFrame (event log).
        group_col : str
            The column to group by (e.g. `store_id`).
        user_col : str
            The column identifying the sequence within each group.
        time_col : str
            The column used for ordering events within a sequence.
        item_col : str
            The column containing the items.
        min_support : int | float, default=1
            The minimum absolute support (number of sequences), or a float percent.
        max_len : int | None, default=None
            Maximum length of the sequential patterns to mine.

        Returns
        -------
        pyspark.sql.DataFrame
            A PySpark DataFrame containing group_col, support, and sequence.
        """
        from .spark import prefixspan_grouped

        return prefixspan_grouped(
            df=df,
            group_col=group_col,
            user_col=user_col,
            time_col=time_col,
            item_col=item_col,
            min_support=min_support,
            max_len=max_len,
        )


def prefixspan(
    sequences: list[list[int]],
    min_support: int | float,
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
    min_support : int | float
        The minimum absolute support (number of sequences a pattern must appear in), or float percent.
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
    import numpy as np

    indptr_list = [0]
    indices_list = []
    for seq in sequences:
        indices_list.extend(seq)
        indptr_list.append(len(indices_list))

    indptr = np.array(indptr_list, dtype=np.uintp)
    indices = np.array(indices_list, dtype=np.uint32)

    if isinstance(min_support, float):
        min_support = max(1, int(min_support * len(sequences)))

    supports, patterns = _rust.prefixspan_mine_py(indptr, indices, int(min_support), max_len)  # type: ignore

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

        import numpy as np

        # Map to integer IDs
        mapped = sorted_df.with_columns(pl.col(item_col).replace(item_to_idx).cast(pl.UInt32).alias("_mapped_items"))

        # Fast flat list representation tracking sizes straight from dataframe
        indices = mapped["_mapped_items"].to_numpy()
        sizes = mapped.group_by(user_col, maintain_order=True).len()["len"].to_numpy()

        indptr = np.zeros(len(sizes) + 1, dtype=np.uintp)
        np.cumsum(sizes, out=indptr[1:])

        return (indptr, indices), idx_to_item  # type: ignore

    elif isinstance(data, pd.DataFrame):
        df_sorted = data.sort_values(by=[user_col, time_col])

        unique_items = df_sorted[item_col].unique()
        item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        idx_to_item = dict(enumerate(unique_items))

        import numpy as np

        mapped_items = df_sorted[item_col].map(lambda x: item_to_idx[x])
        indices = mapped_items.to_numpy(dtype=np.uint32)

        sizes = df_sorted.groupby(user_col, sort=False).size().to_numpy()
        indptr = np.zeros(len(sizes) + 1, dtype=np.uintp)
        np.cumsum(sizes, out=indptr[1:])

        return (indptr, indices), idx_to_item  # type: ignore

    else:
        raise TypeError(f"Expected Pandas, Polars, or PySpark DataFrame, got {type(data)}")
