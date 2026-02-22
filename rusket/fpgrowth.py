from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .model import Miner, RuleMinerMixin

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import polars as pl

class FPGrowth(Miner, RuleMinerMixin):
    """FP-Growth frequent itemset miner.

    This class wraps the fast, core Rust FP-Growth implementation.
    """

    def __init__(
        self,
        data: pd.DataFrame | pl.DataFrame | np.ndarray | Any,
        item_names: list[str] | None = None,
        min_support: float = 0.5,
        null_values: bool = False,
        use_colnames: bool = False,
        max_len: int | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ):
        """Initialize the FPGrowth miner.

        Parameters
        ----------
        data : pandas.DataFrame, polars.DataFrame, or numpy.ndarray
            The input dataset containing transactions. Can be a dense matrix of 0s and 1s,
            a scipy sparse matrix, or a boolean/integer DataFrame.
        item_names : list[str] | None, default=None
            Custom column names to use if input is a numpy array or scipy sparse matrix
            and `use_colnames=True`.
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
            If > 0, print progress and backend selection details to standard output.
        """
        super().__init__(data=data, item_names=item_names, **kwargs)
        self.min_support = min_support
        self.null_values = null_values
        self.use_colnames = use_colnames
        self.max_len = max_len
        self.verbose = verbose

    def mine(self, **kwargs: Any) -> pd.DataFrame:
        """Execute the FP-growth algorithm on the stored data.

        Returns
        -------
        pandas.DataFrame
            DataFrame with two columns:
            - `support`: the support score.
            - `itemsets`: list of items (indices or column names).
        """
        if self.min_support <= 0.0:
            raise ValueError(
                f"`min_support` must be a positive number within the interval `(0, 1]`. Got {self.min_support}."
            )

        from ._core import dispatch

        return dispatch(
            self.data,
            self.min_support,
            self.null_values,
            self.use_colnames,
            self.max_len,
            "fpgrowth",
            self.item_names,
            self.verbose,
        )  # type: ignore[arg-type]

def fpgrowth(
    df: pd.DataFrame | pl.DataFrame | np.ndarray | Any,
    min_support: float = 0.5,
    null_values: bool = False,
    use_colnames: bool = False,
    max_len: int | None = None,
    method: str = "fpgrowth",
    verbose: int = 0,
    column_names: list[str] | None = None,
) -> pd.DataFrame:
    """Find frequent itemsets using the FP-growth algorithm.

    This module-level function relies on the Object-Oriented APIs.
    """
    if method not in ("fpgrowth", "eclat", "auto"):
        raise ValueError(
            f"`method` must be 'fpgrowth', 'eclat', or 'auto'. Got: {method}"
        )

    if method == "eclat":
        from .eclat import Eclat
        return Eclat(
            data=df,
            item_names=column_names,
            min_support=min_support,
            null_values=null_values,
            use_colnames=use_colnames,
            max_len=max_len,
            verbose=verbose
        ).mine()
    elif method == "auto":
        from .mine import AutoMiner
        return AutoMiner(
            data=df,
            item_names=column_names,
            min_support=min_support,
            null_values=null_values,
            use_colnames=use_colnames,
            max_len=max_len,
            verbose=verbose
        ).mine()

    return FPGrowth(
        data=df,
        item_names=column_names,
        min_support=min_support,
        null_values=null_values,
        use_colnames=use_colnames,
        max_len=max_len,
        verbose=verbose,
    ).mine()
