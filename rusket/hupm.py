"""High-Utility Pattern Mining (HUPM)."""

from __future__ import annotations
import pandas as pd

from . import _rusket as _rust  # type: ignore

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
    total_utils, patterns = _rust.hupm_mine_py(
        transactions, utilities, min_utility, max_len
    )
    
    return pd.DataFrame({
        "utility": total_utils,
        "itemset": patterns,
    }).sort_values(by="utility", ascending=False).reset_index(drop=True)
