"""NegFIN (Fastest exact mining algorithm with Parent Equivalence Pruning) for frequent itemset mining."""

from __future__ import annotations

from typing import Any

# Removed
import pandas as pd

from .model import Miner


class NegFIN(Miner):
    """NegFIN algorithm for frequent itemset mining.

    NegFIN is highly optimized for very dense datasets. It operates natively
    on bit-vectors (Bitsets) allowing for linear time exact intersection through
    hardware-native bitwise operations, and introduces Parent Equivalence Pruning (PEP)
    to mathematically prune redundant search spaces.
    """

    def __init__(
        self,
        data: pd.DataFrame | Any,
        item_names: list[str] | None = None,
        min_support: float = 0.5,
        null_values: bool = False,
        use_colnames: bool = True,
        max_len: int | None = None,
        verbose: int = 0,
        **kwargs: Any,
    ):
        super().__init__(data=data, item_names=item_names, **kwargs)
        self.min_support = min_support
        self.null_values = null_values
        self.use_colnames = use_colnames
        self.max_len = max_len
        self.verbose = verbose

    def mine(self, **kwargs: Any) -> pd.DataFrame:
        """Mine frequent itemsets using the NegFIN algorithm.

        Parameters
        ----------
        **kwargs : Any
            min_support: The minimum support threshold (default: self.kwargs.get("min_support", 0.5))
            use_colnames: Use item names instead of IDs in the output (default: self.kwargs.get("use_colnames", True))
            max_len: The maximum length of the itemsets (default: self.kwargs.get("max_len", None))

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'support' and 'itemsets'.
        """
        min_support = kwargs.get("min_support", getattr(self, "min_support", 0.5))
        null_values = kwargs.get("null_values", getattr(self, "null_values", False))
        use_colnames = kwargs.get("use_colnames", getattr(self, "use_colnames", True))
        max_len = kwargs.get("max_len", getattr(self, "max_len", None))
        verbose = kwargs.get("verbose", getattr(self, "verbose", 0))

        if min_support <= 0.0:
            raise ValueError(
                f"`min_support` must be a positive number within the interval `(0, 1]`. Got {min_support}."
            )

        from ._core import dispatch

        result_df = dispatch(
            self.data,
            min_support,
            null_values,
            use_colnames,
            max_len,
            "negfin",
            self.item_names,
            verbose,
        )  # type: ignore[arg-type]
        return self._convert_to_orig_type(result_df)
