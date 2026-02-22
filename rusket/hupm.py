"""High-Utility Pattern Mining (HUPM)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from . import _rusket as _rust  # type: ignore
from ._compat import to_dataframe
from .model import Miner


class HUPM(Miner):
    """High-Utility Pattern Mining (HUPM) model.

    This class discovers combinations of items that generate a high total utility
    (e.g., profit) across all transactions, even if they aren't the most frequent.
    """

    def __init__(
        self,
        transactions: list[list[int]],
        utilities: list[list[float]],
        min_utility: float,
        max_len: int | None = None,
    ):
        """Initialize HUPM with pre-formatted transactions and utilities.

        Parameters
        ----------
        transactions : list of list of int
            A list of transactions, where each transaction is a list of item IDs.
        utilities : list of list of float
            A list of identical structure to `transactions`, but containing the
            numeric utility (e.g., profit) of that item in that specific transaction.
        min_utility : float
            The minimum total utility required to consider a pattern "high-utility".
        max_len : int, optional
            The maximum length of the itemsets to mine.
        """
        super().__init__(data=transactions, item_names=None)
        self.utilities = utilities
        self.min_utility = min_utility
        self.max_len = max_len

    @classmethod
    def from_transactions(
        cls,
        data: Any,
        transaction_col: str | None = None,
        item_col: str | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ) -> HUPM:
        """Initialize the HUPM model from a long-format DataFrame.

        Parameters
        ----------
        data : pd.DataFrame | pl.DataFrame | pyspark.sql.DataFrame
            Event log containing transactions and items.
        transaction_col : str, optional
            Column name identifying the transaction ID.
        item_col : str, optional
            Column name identifying the item ID (must be numeric integers).
        verbose : int, optional
            Verbosity level.
        **kwargs
            Must contain `utility_col` (str) and `min_utility` (float).
            Can optionally contain `max_len` (int).
        """
        utility_col = kwargs["utility_col"]
        min_utility = kwargs["min_utility"]
        max_len = kwargs.get("max_len", None)

        if transaction_col is None or item_col is None:
            raise ValueError("transaction_col and item_col must be provided for HUPM.")
        data = to_dataframe(data)

        try:
            import polars as pl
            is_polars = isinstance(data, pl.DataFrame)
        except ImportError:
            is_polars = False

        if is_polars:
            import polars as pl
            grouped = data.group_by(transaction_col).agg([
                pl.col(item_col).alias("items"),
                pl.col(utility_col).alias("utils")
            ])
            transactions = grouped["items"].to_list()
            utilities = grouped["utils"].to_list()
        elif isinstance(data, pd.DataFrame):
            grouped = data.groupby(transaction_col).agg(
                items=(item_col, list),
                utils=(utility_col, list)
            )
            transactions = grouped["items"].tolist()
            utilities = grouped["utils"].tolist()
        else:
            raise TypeError(f"Expected Pandas or Polars DataFrame, got {type(data)}")

        return cls(transactions=transactions, utilities=utilities, min_utility=min_utility, max_len=max_len)

    def mine(self, **kwargs: Any) -> pd.DataFrame:
        """Mine high-utility itemsets.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing 'utility' and 'itemset' columns.
        """
        from typing import cast
        data_list = cast(list[list[int]], self.data)
        total_utils, patterns = _rust.hupm_mine_py(
            data_list, self.utilities, self.min_utility, self.max_len
        )

        return (
            pd.DataFrame(
                {
                    "utility": total_utils,
                    "itemset": patterns,
                }
            )
            .sort_values(by="utility", ascending=False)
            .reset_index(drop=True)
        )


def hupm(
    transactions: list[list[int]],
    utilities: list[list[float]],
    min_utility: float,
    max_len: int | None = None,
) -> pd.DataFrame:
    """Mine high-utility itemsets.

    This function discovers combinations of items that generate a high total utility
    (e.g., profit) across all transactions, even if they aren't the most frequent.

    Parameters
    ----------
    transactions : list of list of int
        A list of transactions, where each transaction is a list of item IDs.
    utilities : list of list of float
        A list of identical structure to `transactions`, but containing the
        numeric utility (e.g., profit) of that item in that specific transaction.
    min_utility : float
        The minimum total utility required to consider a pattern "high-utility".
    max_len : int, optional
        The maximum length of the itemsets to mine.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing 'utility' and 'itemset' columns.
    """
    import warnings
    warnings.warn(
        "rusket.hupm() is deprecated. Use HUPM.from_transactions() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    total_utils, patterns = _rust.hupm_mine_py(
        transactions, utilities, min_utility, max_len
    )

    return (
        pd.DataFrame(
            {
                "utility": total_utils,
                "itemset": patterns,
            }
        )
        .sort_values(by="utility", ascending=False)
        .reset_index(drop=True)
    )


def mine_hupm(
    data: Any,
    transaction_col: str,
    item_col: str,
    utility_col: str,
    min_utility: float,
    max_len: int | None = None,
) -> pd.DataFrame:
    """Mine high-utility itemsets from a long-format DataFrame.

    Converts a Pandas or Polars DataFrame into the required list-of-lists format
    and runs the High-Utility Pattern Mining (HUPM) algorithm.

    Parameters
    ----------
    data : pd.DataFrame or pl.DataFrame
        A long-format DataFrame where each row represents an item in a transaction.
    transaction_col : str
        Column name identifying the transaction ID.
    item_col : str
        Column name identifying the item ID (must be numeric integers).
    utility_col : str
        Column name identifying the numeric utility (e.g. price, profit) of the item.
    min_utility : float
        The minimum total utility required to consider a pattern "high-utility".
    max_len : int, optional
        Maximum length of the itemsets to mine.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing 'utility' and 'itemset' columns.
    """
    import warnings
    warnings.warn(
        "rusket.mine_hupm() is deprecated. Use HUPM.from_transactions() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    import pandas as pd

    from ._compat import to_dataframe

    data = to_dataframe(data)

    try:
        import polars as pl
        is_polars = isinstance(data, pl.DataFrame)
    except ImportError:
        is_polars = False

    if is_polars:
        import polars as pl
        # Polars native grouping
        grouped = data.group_by(transaction_col).agg([
            pl.col(item_col).alias("items"),
            pl.col(utility_col).alias("utils")
        ])
        transactions = grouped["items"].to_list()
        utilities = grouped["utils"].to_list()
    elif isinstance(data, pd.DataFrame):
        # Pandas native grouping
        grouped = data.groupby(transaction_col).agg(
            items=(item_col, list),
            utils=(utility_col, list)
        )
        transactions = grouped["items"].tolist()
        utilities = grouped["utils"].tolist()
    else:
        raise TypeError(f"Expected Pandas or Polars DataFrame, got {type(data)}")

    return hupm(
        transactions=transactions,
        utilities=utilities,
        min_utility=min_utility,
        max_len=max_len,
    )
