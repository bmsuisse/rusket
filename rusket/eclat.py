from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import polars as pl


def eclat(
    df: pd.DataFrame | pl.DataFrame | np.ndarray | Any,
    min_support: float = 0.5,
    null_values: bool = False,
    use_colnames: bool = False,
    max_len: int | None = None,
    verbose: int = 0,
    column_names: list[str] | None = None,
) -> pd.DataFrame:
    """Find frequent itemsets using the Eclat algorithm.

    Eclat is typically faster than FP-growth on dense datasets due to
    efficient vertical bitset intersection logic.

    Parameters
    ----------
    df : pandas.DataFrame, polars.DataFrame, or numpy.ndarray
        The input dataset containing transactions.
    min_support : float, default=0.5
        The minimum support threshold `[0.0, 1.0]`. Calculates as a percentage of total transactions.
    null_values : bool, default=False
        If True, ignore missing/null values in pandas DataFrames.
    use_colnames : bool, default=False
        If True, returns itemsets containing actual item names (column names)
        rather than their column indices.
    max_len : int | None, default=None
        Maximum length of the itemsets generated. If None, no limit is applied.
    verbose : int, default=0
        If > 0, print progress details to standard output.
    column_names : list[str] | None, default=None
        Custom column names to use if input is a numpy array or scipy sparse matrix
        and `use_colnames=True`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns:
        - `support`: the support score.
        - `itemsets`: list of items (indices or column names).
    """
    if min_support <= 0.0:
        raise ValueError(
            f"`min_support` must be a positive number within the interval `(0, 1]`. Got {min_support}."
        )

    from ._core import dispatch

    return dispatch(
        df,
        min_support,
        null_values,
        use_colnames,
        max_len,
        "eclat",
        column_names,
        verbose,
    )
