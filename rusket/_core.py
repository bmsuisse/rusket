from __future__ import annotations

import math
import time
import typing
from typing import TYPE_CHECKING, Any, Literal

from . import _rusket as _rust  # type: ignore
from ._compat import to_dataframe

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import polars as pl

    from ._compat import DataFrame

Method = Literal["fpgrowth", "eclat", "fin", "lcm", "auto"]

_RUST_DENSE = {
    "fpgrowth": _rust.fpgrowth_from_dense,
    "eclat": _rust.eclat_from_dense,
    "fin": _rust.fin_from_dense,  # type: ignore[attr-defined]
    "lcm": _rust.lcm_from_dense,  # type: ignore[attr-defined]
}
_RUST_CSR = {
    "fpgrowth": _rust.fpgrowth_from_csr,
    "eclat": _rust.eclat_from_csr,
    "fin": _rust.fin_from_csr,  # type: ignore[attr-defined]
    "lcm": _rust.lcm_from_csr,  # type: ignore[attr-defined]
}


def _estimate_dense_memory(shape: tuple[int, int]) -> int:
    """Estimate memory (in bytes) of a dense uint8 array for the given shape."""
    return shape[0] * shape[1]


def _get_available_memory() -> int:
    """Get available system memory in bytes."""
    try:
        import psutil

        return psutil.virtual_memory().available
    except ImportError:
        # Fallback if psutil is not installed
        return 4 * 1024 * 1024 * 1024  # Assume 4GB


def _build_result(
    raw: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_rows: int,
    min_support: float,
    col_names: list | Any,
    use_colnames: bool,
) -> pd.DataFrame:
    import pandas as pd
    import pyarrow as pa

    supports_arr, offsets_arr, items_arr = raw

    if len(supports_arr) == 0:
        return pd.DataFrame(columns=["support", "itemsets"])  # type: ignore[arg-type]

    supports = supports_arr / n_rows

    if use_colnames:
        col_array = pa.array(col_names)
        items_pa = pa.DictionaryArray.from_arrays(pa.array(items_arr, type=pa.int32()), col_array)
        item_type = col_array.type
    else:
        items_pa = pa.array(items_arr, type=pa.int32())
        item_type = pa.int32()

    offsets_pa = pa.array(offsets_arr, type=pa.int32())
    list_arr = pa.ListArray.from_arrays(offsets_pa, items_pa)

    result = pd.DataFrame(
        {
            "support": supports,
            "itemsets": pd.Series(list_arr, dtype=pd.ArrowDtype(pa.list_(item_type))),
        }
    )

    filtered_df = result[result["support"] >= min_support].reset_index(drop=True)
    filtered_df.attrs["num_itemsets"] = n_rows
    return typing.cast("pd.DataFrame", filtered_df)


def _select_method(method: Method, density: float, verbose: int, label: str) -> Method:
    if method != "auto":
        return method
    chosen: Method = "eclat" if density < 0.15 else "fpgrowth"
    if verbose:
        print(f"[{time.strftime('%X')}] Auto-selected method: '{chosen}' ({label} density={density:.4f})")
    return chosen


def _run_fpminer_fallback(
    df: pd.DataFrame | pl.DataFrame | np.ndarray,
    min_support: float,
    max_len: int | None,
    use_colnames: bool,
    col_names: list[str],
    verbose: int = 0,
) -> pd.DataFrame:
    """Fallback to FPMiner (streaming) when memory is low."""
    if verbose:
        print(f"[{time.strftime('%X')}] Low memory detected! Falling back to FPMiner (streaming)...")

    from .streaming import FPMiner

    n_items = len(col_names)
    miner = FPMiner(n_items=n_items)

    # Convert to numpy chunk by chunk or as a whole if it's already an array
    # For now, we reuse the existing data since it's already in memory,
    # but we feed it to the streaming miner which is more memory-efficient
    # than the recursive FP-tree construction in some cases.
    # To TRULY save memory, we should iterate over the input, but here the input
    # is already in memory. The fallback is mostly to avoid the Peak RAM spike
    # of the FP-tree building.

    import numpy as np
    import pandas as pd
    import polars as pl

    if isinstance(df, pd.DataFrame):
        # We can iterate over rows without materializing a massive dense matrix
        # However, FPMiner expects (txn_id, item_id) pairs.
        # We'll use a simple loop for now.
        for i, row in enumerate(df.itertuples(index=False)):
            items = np.where(row)[0].astype(np.int32)
            if len(items) > 0:
                txns = np.full(len(items), i, dtype=np.int64)
                miner.add_chunk(txns, items)
    elif isinstance(df, pl.DataFrame):
        # Similar logic for Polars
        for i, row in enumerate(df.iter_rows()):
            items = np.where(row)[0].astype(np.int32)
            if len(items) > 0:
                txns = np.full(len(items), i, dtype=np.int64)
                miner.add_chunk(txns, items)
    else:
        # Numpy array
        for i, row in enumerate(df):
            items = np.where(row)[0].astype(np.int32)
            if len(items) > 0:
                txns = np.full(len(items), i, dtype=np.int64)
                miner.add_chunk(txns, items)

    return miner.mine(
        min_support=min_support,
        max_len=max_len,
        use_colnames=use_colnames,
        column_names=col_names,
        verbose=verbose,
    )


def _run_dense(
    df: pd.DataFrame,
    min_count: int,
    max_len: int | None,
    method: Method,
    verbose: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np

    t0 = 0.0
    if verbose:
        print(f"[{time.strftime('%X')}] Converting dense DataFrame to C-contiguous uint8 array...")
        t0 = time.perf_counter()
    data = np.ascontiguousarray(df.values, dtype=np.uint8)
    if verbose:
        t1 = time.perf_counter()
        print(f"[{time.strftime('%X')}] Done in {t1 - t0:.2f}s. Calling Rust backend ({method})...")
        t0 = t1
    raw = _RUST_DENSE[method](data, min_count, max_len)
    if verbose:
        print(f"[{time.strftime('%X')}] Rust mining completed in {time.perf_counter() - t0:.2f}s.")
    return raw


def _run_sparse(
    df: pd.DataFrame,
    n_cols: int,
    min_count: int,
    max_len: int | None,
    method: Method,
    verbose: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np

    t0 = 0.0
    if verbose:
        print(f"[{time.strftime('%X')}] Converting Pandas Sparse DataFrame to CSR array...")
        t0 = time.perf_counter()
    csr = df.sparse.to_coo().tocsr()
    csr.eliminate_zeros()
    indptr = np.asarray(csr.indptr, dtype=np.int32)
    indices = np.asarray(csr.indices, dtype=np.int32)
    if verbose:
        t1 = time.perf_counter()
        print(f"[{time.strftime('%X')}] Done in {t1 - t0:.2f}s. Calling Rust backend ({method})...")
        t0 = t1
    raw = _RUST_CSR[method](indptr, indices, n_cols, min_count, max_len)
    if verbose:
        print(f"[{time.strftime('%X')}] Rust mining completed in {time.perf_counter() - t0:.2f}s.")
    return raw


def _run_polars(
    df: pl.DataFrame,
    min_support: float,
    use_colnames: bool,
    max_len: int | None,
    method: Method,
    verbose: int = 0,
) -> pd.DataFrame:
    import numpy as np
    import polars as pl

    t0 = 0.0
    if verbose:
        print(f"[{time.strftime('%X')}] Converting Polars DataFrame to C-contiguous uint8 array...")
        t0 = time.perf_counter()
    n_rows, n_cols = df.shape
    min_count = math.ceil(min_support * n_rows)
    arr = np.ascontiguousarray(df.to_numpy(), dtype=np.uint8)
    if verbose:
        t1 = time.perf_counter()
        print(f"[{time.strftime('%X')}] Done in {t1 - t0:.2f}s. Calling Rust backend ({method})...")
        t0 = t1

    if method == "auto":
        nnz = df.select(pl.all().cast(pl.Boolean).sum()).sum_horizontal().item()
        density = nnz / (n_rows * n_cols) if n_rows * n_cols > 0 else 0.0

        # Memory check
        required_mem = _estimate_dense_memory(df.shape)
        available_mem = _get_available_memory()
        if required_mem > available_mem * 0.7:  # 70% threshold
            return _run_fpminer_fallback(df, min_support, max_len, use_colnames, list(df.columns), verbose)

        method = _select_method("auto", density, verbose, "polars")

    raw = _RUST_DENSE[method](arr, min_count, max_len)
    if verbose:
        print(
            f"[{time.strftime('%X')}] Rust mining completed in {time.perf_counter() - t0:.2f}s. Assembling DataFrame..."
        )
    return _build_result(raw, n_rows, min_support, df.columns, use_colnames)


def dispatch(
    df: DataFrame | Any,
    min_support: float,
    null_values: bool,
    use_colnames: bool,
    max_len: int | None,
    method: Method,
    column_names: list[str] | None = None,
    verbose: int = 0,
) -> pd.DataFrame:
    import numpy as np

    t0 = 0.0
    if verbose:
        print(f"[{time.strftime('%X')}] Analyzing input data type...")

    df = to_dataframe(df)

    t = type(df).__name__

    if t == "DataFrame" and getattr(df, "__module__", "").startswith("polars"):
        return _run_polars(
            typing.cast("pl.DataFrame", df),
            min_support,
            use_colnames,
            max_len,
            method,
            verbose,
        )

    if t in ("csr_matrix", "csr_array"):
        csr: Any = df
        n_rows, n_cols = csr.shape  # type: ignore[misc]
        min_count = math.ceil(min_support * n_rows)
        if verbose:
            print(f"[{time.strftime('%X')}] Extracting CSR arrays from SciPy matrix...")
            t0 = time.perf_counter()
        csr.eliminate_zeros()
        indptr = np.asarray(csr.indptr, dtype=np.int32)  # type: ignore[misc]
        indices = np.asarray(csr.indices, dtype=np.int32)  # type: ignore[misc]
        if verbose:
            t1 = time.perf_counter()
            print(f"[{time.strftime('%X')}] Done in {t1 - t0:.2f}s. Calling Rust backend ({method})...")
            t0 = t1
        density = csr.nnz / (n_rows * n_cols) if n_rows * n_cols > 0 else 0.0

        # Memory check for dense intermediate if we were to use fpgrowth/eclat
        # Note: CSR is already space-efficient, but Rust algorithms might expand it.
        # For CSR, we mostly care about the final itemsets size, but we'll stick to dense estimate
        # for a safe fallback if the dense version would be too big.
        if method == "auto":
            required_mem = _estimate_dense_memory(csr.shape)
            available_mem = _get_available_memory()
            if required_mem > available_mem * 0.7:
                col_names = column_names or [str(i) for i in range(n_cols)]
                return _run_fpminer_fallback(csr.toarray(), min_support, max_len, use_colnames, col_names, verbose)

        method = _select_method(method, density, verbose, "csr")
        raw = _RUST_CSR[method](indptr, indices, n_cols, min_count, max_len)
        if verbose:
            print(
                f"[{time.strftime('%X')}] Rust mining completed in {time.perf_counter() - t0:.2f}s. Assembling DataFrame..."
            )
        col_names = column_names or [str(i) for i in range(n_cols)]
        return _build_result(raw, n_rows, min_support, col_names, use_colnames)

    if t == "ndarray":
        df_nd = typing.cast("np.ndarray", df)
        n_rows, n_cols = df_nd.shape
        min_count = math.ceil(min_support * n_rows)
        col_names = [str(i) for i in range(n_cols)]
        if verbose:
            print(f"[{time.strftime('%X')}] Ensuring Numpy array is C-contiguous uint8...")
            t0 = time.perf_counter()
        data = np.ascontiguousarray(df_nd, dtype=np.uint8)
        if verbose:
            t1 = time.perf_counter()
            print(f"[{time.strftime('%X')}] Done in {t1 - t0:.2f}s. Calling Rust backend ({method})...")
            t0 = t1
        density = np.count_nonzero(df_nd) / (n_rows * n_cols) if n_rows * n_cols > 0 else 0.0

        # Memory check
        if method == "auto":
            required_mem = _estimate_dense_memory(df_nd.shape)
            available_mem = _get_available_memory()
            if required_mem > available_mem * 0.7:
                return _run_fpminer_fallback(df_nd, min_support, max_len, use_colnames, col_names, verbose)

        method = _select_method(method, density, verbose, "ndarray")
        raw = _RUST_DENSE[method](data, min_count, max_len)
        if verbose:
            print(
                f"[{time.strftime('%X')}] Rust mining completed in {time.perf_counter() - t0:.2f}s. Assembling DataFrame..."
            )
        return _build_result(raw, n_rows, min_support, col_names, use_colnames)

    df_pd = typing.cast("pd.DataFrame", df)
    n_rows, n_cols = df_pd.shape
    min_count = math.ceil(min_support * n_rows)
    col_names = list(df_pd.columns)

    import pandas as pd

    from ._validation import valid_input_check

    # Validate first so invalid values (e.g. 2) are caught before we coerce
    valid_input_check(df_pd, null_values)

    # Coerce integer 0/1 DataFrames to bool to avoid DeprecationWarning on next use
    if not hasattr(df_pd, "sparse") and not bool(df_pd.dtypes.apply(pd.api.types.is_bool_dtype).all()):
        df_pd = df_pd.astype(bool)

    if hasattr(df_pd, "sparse"):
        nnz = getattr(df_pd.sparse, "density", 0.0) * (n_rows * n_cols)
    else:
        nnz = df_pd.sum().sum()
    density = nnz / (n_rows * n_cols) if n_rows * n_cols > 0 else 0.0

    # Memory check
    if method == "auto":
        required_mem = _estimate_dense_memory(df_pd.shape)
        available_mem = _get_available_memory()
        if required_mem > available_mem * 0.7:
            return _run_fpminer_fallback(df_pd, min_support, max_len, use_colnames, col_names, verbose)

    method = _select_method(method, density, verbose, "pandas")

    if hasattr(df_pd, "sparse"):
        raw = _run_sparse(df_pd, n_cols, min_count, max_len, method, verbose)
    else:
        raw = _run_dense(df_pd, min_count, max_len, method, verbose)

    if verbose:
        print(f"[{time.strftime('%X')}] Assembling result DataFrame...")
    return _build_result(raw, n_rows, min_support, col_names, use_colnames)
