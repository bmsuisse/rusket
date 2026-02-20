from __future__ import annotations

import time


import math
import typing
from typing import TYPE_CHECKING, Any, Literal

from . import _rusket as _rust  # type: ignore

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import polars as pl

Method = Literal["fpgrowth", "eclat"]

_RUST_DENSE = {"fpgrowth": _rust.fpgrowth_from_dense, "eclat": _rust.eclat_from_dense}
_RUST_CSR = {"fpgrowth": _rust.fpgrowth_from_csr, "eclat": _rust.eclat_from_csr}


def _build_result(
    t0: float = 0.0, 
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
        return pd.DataFrame(columns=["support", "itemsets"])

    supports = supports_arr / n_rows

    if use_colnames:
        col_array = pa.array(col_names)
        items_pa = pa.DictionaryArray.from_arrays(
            pa.array(items_arr, type=pa.int32()), col_array
        )
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
    return typing.cast("pd.DataFrame", filtered_df)


def _run_dense(
    df: pd.DataFrame,
    min_count: int,
    max_len: int | None,
    method: Method,
    verbose: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np

    if verbose:
        print(
            f"[{time.strftime('%X')}] Converting dense DataFrame to C-contiguous uint8 array..."
        )
        t0 = time.perf_counter()
    data = np.ascontiguousarray(df.values, dtype=np.uint8)
    if verbose:
        t1 = time.perf_counter()
        print(
            f"[{time.strftime('%X')}] Done in {t1 - t0:.2f}s. Calling Rust backend ({method})..."
        )
        t0 = t1
    raw = _RUST_DENSE[method](data, min_count, max_len)
    if verbose:
        print(
            f"[{time.strftime('%X')}] Rust mining completed in {time.perf_counter() - t0:.2f}s."
        )
    return raw

    data = np.ascontiguousarray(df.values, dtype=np.uint8)
    return _RUST_DENSE[method](data, min_count, max_len)


def _run_sparse(
    df: pd.DataFrame,
    n_cols: int,
    min_count: int,
    max_len: int | None,
    method: Method,
    verbose: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np

    if verbose:
        print(
            f"[{time.strftime('%X')}] Converting Pandas Sparse DataFrame to CSR array..."
        )
        t0 = time.perf_counter()
    csr = df.sparse.to_coo().tocsr()
    csr.eliminate_zeros()
    indptr = np.asarray(csr.indptr, dtype=np.int32)
    indices = np.asarray(csr.indices, dtype=np.int32)
    if verbose:
        t1 = time.perf_counter()
        print(
            f"[{time.strftime('%X')}] Done in {t1 - t0:.2f}s. Calling Rust backend ({method})..."
        )
        t0 = t1
    raw = _RUST_CSR[method](indptr, indices, n_cols, min_count, max_len)
    if verbose:
        print(
            f"[{time.strftime('%X')}] Rust mining completed in {time.perf_counter() - t0:.2f}s."
        )
    return raw

    csr = df.sparse.to_coo().tocsr()
    csr.eliminate_zeros()
    indptr = np.asarray(csr.indptr, dtype=np.int32)
    indices = np.asarray(csr.indices, dtype=np.int32)
    return _RUST_CSR[method](indptr, indices, n_cols, min_count, max_len)


def _run_polars(
    df: pl.DataFrame,
    min_support: float,
    use_colnames: bool,
    max_len: int | None,
    method: Method,
    verbose: int = 0,
) -> pd.DataFrame:
    import numpy as np

    if verbose:
        print(
            f"[{time.strftime('%X')}] Converting Polars DataFrame to C-contiguous uint8 array..."
        )
        t0 = time.perf_counter()
    n_rows, n_cols = df.shape
    min_count = math.ceil(min_support * n_rows)
    arr = np.ascontiguousarray(df.to_numpy(), dtype=np.uint8)
    if verbose:
        t1 = time.perf_counter()
        print(
            f"[{time.strftime('%X')}] Done in {t1 - t0:.2f}s. Calling Rust backend ({method})..."
        )
        t0 = t1
    raw = _RUST_DENSE[method](arr, min_count, max_len)
    if verbose:
        print(
            f"[{time.strftime('%X')}] Rust mining completed in {time.perf_counter() - t0:.2f}s. Assembling DataFrame..."
        )
    return _build_result(raw, n_rows, min_support, df.columns, use_colnames)

    n_rows, n_cols = df.shape
    min_count = math.ceil(min_support * n_rows)
    arr = np.ascontiguousarray(df.to_numpy(), dtype=np.uint8)
    raw = _RUST_DENSE[method](arr, min_count, max_len)
    return _build_result(raw, n_rows, min_support, df.columns, use_colnames)


def dispatch(
    df: pd.DataFrame | pl.DataFrame | np.ndarray | Any,
    min_support: float,
    null_values: bool,
    use_colnames: bool,
    max_len: int | None,
    method: Method,
    column_names: list[str] | None = None,
    verbose: int = 0,
) -> pd.DataFrame:
    import math
    import numpy as np

    if verbose:
        print(f"[{time.strftime('%X')}] Analyzing input data type...")

    t = type(df).__name__

    if t == "DataFrame" and getattr(df, "__module__", "").startswith("pyspark"):
        df = typing.cast(Any, df).toPandas()
        t = "DataFrame"

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
            print(
                f"[{time.strftime('%X')}] Done in {t1 - t0:.2f}s. Calling Rust backend ({method})..."
            )
            t0 = t1
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
            print(
                f"[{time.strftime('%X')}] Ensuring Numpy array is C-contiguous uint8..."
            )
            t0 = time.perf_counter()
        data = np.ascontiguousarray(df_nd, dtype=np.uint8)
        if verbose:
            t1 = time.perf_counter()
            print(
                f"[{time.strftime('%X')}] Done in {t1 - t0:.2f}s. Calling Rust backend ({method})..."
            )
            t0 = t1
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

    from ._validation import valid_input_check

    valid_input_check(df_pd, null_values)

    if hasattr(df_pd, "sparse"):
        raw = _run_sparse(df_pd, n_cols, min_count, max_len, method, verbose)
    else:
        raw = _run_dense(df_pd, min_count, max_len, method, verbose)

    if verbose:
        print(f"[{time.strftime('%X')}] Assembling result DataFrame...")
    return _build_result(raw, n_rows, min_support, col_names, use_colnames)
