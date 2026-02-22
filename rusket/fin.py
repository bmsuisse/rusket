from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .model import Miner, RuleMinerMixin

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import polars as pl


class FIN(Miner, RuleMinerMixin):
    """FIN (Fast Itemset per Nodeset) frequent itemset miner.

    This class wraps the fast core Rust FIN implementation.
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
        """Initialize the FIN miner.

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
            If > 0, print progress to standard output.
        """
        super().__init__(data=data, item_names=item_names, **kwargs)
        self.min_support = min_support
        self.null_values = null_values
        self.use_colnames = use_colnames
        self.max_len = max_len
        self.verbose = verbose

    def mine(self, **kwargs: Any) -> pd.DataFrame:
        """Execute the FIN algorithm on the stored data.

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
            "fin",
            self.item_names,
            self.verbose,
        )  # type: ignore[arg-type]
