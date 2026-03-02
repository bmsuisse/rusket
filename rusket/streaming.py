"""Pure-Python wrapper around the Rust ``FPMiner`` streaming accumulator.

This provides the high-level API for feeding billion-row datasets to Rust
in memory-safe chunks without ever materialising the full dataset in Python.
"""

from __future__ import annotations

import time
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
    >>> freq = miner.mine(min_support=0.001, max_len=3, use_colnames=True)
    """

    def __init__(
        self,
        n_items: int,
        max_ram_mb: int | None = -1,
        hint_n_transactions: int | None = None,
    ) -> None:
        if max_ram_mb == -1:
            try:
                import psutil

                available_mb = psutil.virtual_memory().available // (1024 * 1024)
                max_ram_mb = max(100, int(available_mb * 0.90))
            except ImportError:
                max_ram_mb = 4000

        self._inner: Any = _rust.FPMiner(n_items, max_ram_mb, hint_n_transactions)  # type: ignore
        self._n_rows: int = 0

    @property
    def max_ram_mb(self) -> int | None:
        """The maximum RAM allowed for memory chunks before spilling to disk."""
        return self._inner.max_ram_mb

    @property
    def n_rows(self) -> int:
        """Total number of (txn_id, item_id) pairs accumulated so far."""
        return self._n_rows

    @property
    def n_transactions(self) -> int:
        """Number of distinct transactions accumulated so far."""
        return self._inner.n_transactions  # type: ignore

    @property
    def n_items(self) -> int:
        """Number of distinct items (columns)."""
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
        import numpy as np

        txn = np.asarray(txn_ids, dtype=np.int64)
        item = np.asarray(item_ids, dtype=np.int32)
        self._inner.add_chunk(txn, item)
        self._n_rows += len(txn)
        return self

    def add_arrow_batch(self, batch: Any, txn_col: str, item_col: str) -> FPMiner:
        """Feed a PyArrow RecordBatch directly into the miner.
        Zero-copy extraction is used if types match (Int64/Int32).
        """
        txn_array = batch.column(txn_col).to_numpy(zero_copy_only=False)
        item_array = batch.column(item_col).to_numpy(zero_copy_only=False)

        return self.add_chunk(txn_array, item_array)

    def mine(
        self,
        min_support: float = 0.5,
        max_len: int | None = None,
        use_colnames: bool = True,
        column_names: list[str] | None = None,
        method: typing.Literal["fpgrowth", "eclat"] = "fpgrowth",
        verbose: int = 0,
    ) -> pd.DataFrame:
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
        if self._n_rows == 0:
            from rusket._dependencies import import_optional_dependency

            pd = import_optional_dependency("pandas")

            return pd.DataFrame(columns=["support", "itemsets"])  # type: ignore

        import numpy as np

        t0 = 0.0
        if verbose:
            print(f"[{time.strftime('%X')}] FPMiner: Mining ({method})...")
            t0 = time.perf_counter()

        chosen_method = method
        if method == "fpgrowth":
            result_tuple = self._inner.mine_fpgrowth(min_support, max_len)
        else:
            result_tuple = self._inner.mine_eclat(min_support, max_len)

        if verbose:
            t1 = time.perf_counter()
            print(f"[{time.strftime('%X')}] FPMiner: Mining ({chosen_method}) completed in {t1 - t0:.2f}s.")

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
        self._inner.reset()
        self._n_rows = 0

    def fit(self, **kwargs: Any) -> FPMiner:
        """Sklearn-compatible alias for ``mine()``. Runs the mining algorithm.

        Returns
        -------
        self
        """
        self._result = self.mine(**kwargs)
        return self

    def predict(self, **kwargs: Any) -> pd.DataFrame:
        """Return the last mined result, or run ``fit()`` first.

        Returns
        -------
        pd.DataFrame
            The frequent itemsets.
        """
        if not hasattr(self, "_result") or self._result is None:
            self.fit(**kwargs)
        return self._result  # type: ignore[return-value]


def mine_duckdb(
    con: Any,
    query: str,
    n_items: int,
    txn_col: str,
    item_col: str,
    min_support: float = 0.5,
    max_len: int | None = None,
    chunk_size: int = 1_000_000,
) -> pd.DataFrame:
    """Stream directly from a DuckDB query via Arrow RecordBatches.

    This is extremely memory efficient, bypassing Pandas entirely.
    """
    miner = FPMiner(n_items=n_items)
    arrow_reader = con.execute(query).fetch_record_batch(chunk_size=chunk_size)

    for batch in arrow_reader:
        miner.add_arrow_batch(batch, txn_col, item_col)

    return miner.mine(min_support=min_support, max_len=max_len)


def mine_spark(
    spark_df: Any,
    n_items: int,
    txn_col: str,
    item_col: str,
    min_support: float = 0.5,
    max_len: int | None = None,
) -> pd.DataFrame:
    """Stream natively from a PySpark DataFrame on Databricks via Arrow.

    Uses `toLocalIterator()` to fetch Arrow chunks incrementally directly
    to the driver node, avoiding massive memory spikes.
    """
    miner = FPMiner(n_items=n_items)

    # Enable Arrow-based data transfers implicitly via spark-pandas interop
    # Spark 3.3+ supports fetching Arrow batches via mapInArrow/toLocalIterator

    try:
        # Databricks DBR 13+ / PySpark 3.4+ native Arrow stream
        batches = spark_df.select(txn_col, item_col).toArrow()
        # Returns an Arrow Table
        for batch in batches.to_batches():
            miner.add_arrow_batch(batch, txn_col, item_col)
    except Exception:
        # Fallback to older spark iterator
        for row_chunk in spark_df.select(txn_col, item_col).toLocalIterator():
            import numpy as np

            if hasattr(row_chunk, "asDict"):
                txn = np.array([row_chunk[txn_col]], dtype=np.int64)
                item = np.array([row_chunk[item_col]], dtype=np.int32)
                miner.add_chunk(txn, item)

    return miner.mine(min_support=min_support, max_len=max_len)
