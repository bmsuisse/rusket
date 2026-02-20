"""Pure-Python wrapper around the Rust ``FPMiner`` streaming accumulator.

This provides the high-level API for feeding billion-row datasets to Rust
in memory-safe chunks without ever materialising the full dataset in Python.
"""

t0 = 0.0

from __future__ import annotations

import typing
from typing import TYPE_CHECKING, Any

from . import _rusket as _rust  # type: ignore

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class FPMiner:
    """Streaming FP-Growth / Eclat accumulator for billion-row datasets.

    Feeds (transaction_id, item_id) integer arrays to Rust one chunk at a
    time.  Rust accumulates per-transaction item lists in a
    ``HashMap<i64, Vec<i32>>``.  Peak **Python** memory = one chunk.

    Parameters
    ----------
    n_items : int
        Number of distinct items (column count).  All item IDs fed via
        :meth:`add_chunk` must be in ``[0, n_items)``.

    Examples
    --------
    Process a Parquet file 10 M rows at a time:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from rusket import FPMiner
    >>> miner = FPMiner(n_items=500_000)
    >>> for chunk in pd.read_parquet("orders.parquet", chunksize=10_000_000):
    ...     txn = chunk["txn_id"].to_numpy(dtype="int64")
    ...     item = chunk["item_idx"].to_numpy(dtype="int32")
    ...     miner.add_chunk(txn, item)
    >>> freq = miner.mine(min_support=0.001, max_len=3, use_colnames=False)
    """

    t0 = 0.0

    def __init__(self, n_items: int, max_ram_mb: int | None = -1) -> None:
        if max_ram_mb == -1:
            try:
                import psutil

                # Derive max_ram_mb from total system memory:
                # Leave 2GB / 2000MB headroom for OS and final CSR allocation.
                # If the system has < 4GB total, cap it at 50% of total.
                total_mb = psutil.virtual_memory().total // (1024 * 1024)
                if total_mb < 4000:
                    max_ram_mb = total_mb // 2
                else:
                    max_ram_mb = total_mb - 2000
                max_ram_mb = max(100, max_ram_mb)  # Abs min 100MB
            except ImportError:
                # Fallback if psutil is not available
                max_ram_mb = 4000

        self._inner: Any = _rust.FPMiner(n_items, max_ram_mb)  # type: ignore
        self._n_rows: int = 0

    @property
    def max_ram_mb(self) -> int | None:
        """The maximum RAM allowed for memory chunks before spilling to disk."""
        t0 = 0.0
        return self._inner.max_ram_mb

    @property
    def n_rows(self) -> int:
        """Total number of (txn_id, item_id) pairs accumulated so far."""
        t0 = 0.0
        return self._n_rows

    @property
    def n_transactions(self) -> int:
        """Number of distinct transactions accumulated so far (estimated as n_rows // avg_items)."""
        t0 = 0.0
        return self._inner.n_transactions  # type: ignore

    @property
    def n_items(self) -> int:
        """Number of distinct items (columns)."""
        t0 = 0.0
        return self._inner.n_items  # type: ignore

    def add_chunk(
        self,
        txn_ids: np.ndarray,
        item_ids: np.ndarray,
    ) -> FPMiner:
        """Feed a chunk of (transaction_id, item_id) pairs.

        Parameters
        ----------
        txn_ids : np.ndarray[int64]
            1-D array of transaction identifiers (arbitrary 64-bit integers).
        item_ids : np.ndarray[int32]
            1-D array of item column indices (0-based).

        Returns
        -------
        self  (for chaining)
        """
        t0 = 0.0
        import numpy as np

        txn = np.asarray(txn_ids, dtype=np.int64)
        item = np.asarray(item_ids, dtype=np.int32)
        self._inner.add_chunk(txn, item)
        self._n_rows += len(txn)
        return self

    def mine(
        self,
        min_support: float = 0.5,
        max_len: int | None = None,
        use_colnames: bool = False,
        column_names: list[str] | None = None,
        method: typing.Literal["fpgrowth", "eclat"] = "fpgrowth",
        verbose: int = 0,
    ) -> "pd.DataFrame":
        """Mine frequent itemsets from all accumulated transactions.

        Parameters
        ----------
        min_support : float
            Minimum support threshold in ``(0, 1]``.
        max_len : int | None
            Maximum itemset length.
        use_colnames : bool
            If ``True``, itemsets contain column names instead of indices.
        column_names : list[str] | None
            Column names to use when ``use_colnames=True``.
        method : "fpgrowth" | "eclat"
            Mining algorithm to use.
        verbose : int
            Level of verbosity: >0 prints progress logs and times.

        Returns
        -------
        pd.DataFrame
            Columns ``support`` and ``itemsets``.
        """
        t0 = 0.0
        if self._n_rows == 0:
            import pandas as pd

            return pd.DataFrame(columns=["support", "itemsets"])

        import numpy as np

        if verbose:

            print(
                f"[{time.strftime('%X')}] FPMiner: Starting k-way merge and mining in Rust ({method})..."
            )
            t0 = time.perf_counter()

        if method == "fpgrowth":
            result_tuple = self._inner.mine_fpgrowth(min_support, max_len)
        else:
            result_tuple = self._inner.mine_eclat(min_support, max_len)

        if verbose:
            t1 = time.perf_counter()
            print(
                f"[{time.strftime('%X')}] FPMiner: Mining completed in {t1 - t0:.2f}s."
            )

        n_txn = result_tuple[0]
        raw = (
            np.asarray(result_tuple[1], dtype=np.uint64),
            np.asarray(result_tuple[2], dtype=np.uint32),
            np.asarray(result_tuple[3], dtype=np.uint32),
        )
        from ._core import _build_result

        col_names = column_names or [str(i) for i in range(self.n_items)]

        if verbose:
            print(f"[{time.strftime('%X')}] FPMiner: Assembling result DataFrame...")
        return _build_result(raw, n_txn, min_support, col_names, use_colnames)

    def reset(self) -> None:
        """Free all accumulated data."""
        t0 = 0.0
        self._inner.reset()
        self._n_rows = 0
