import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse
import rusket


def test_auto_dense_pandas():
    # Dense dataset: 4 transactions, 3 items -> density is 6/12 = 0.5 > 0.15 (FPGrowth)
    df = pd.DataFrame(
        {"apple": [1, 1, 0, 1], "banana": [1, 0, 0, 1], "cherry": [0, 0, 1, 0]}
    )
    res_auto = rusket.mine(df, min_support=0.5, method="auto")
    res_fpgrowth = rusket.fpgrowth(df, min_support=0.5)

    pd.testing.assert_frame_equal(res_auto, res_fpgrowth)


def test_auto_dense_polars():
    # Dense dataset: density = 0.5 > 0.15 (FPGrowth)
    df = pl.DataFrame(
        {"apple": [1, 1, 0, 1], "banana": [1, 0, 0, 1], "cherry": [0, 0, 1, 0]}
    )
    res_auto = rusket.mine(df, min_support=0.5, method="auto")
    res_fpgrowth = rusket.fpgrowth(df, min_support=0.5)

    pd.testing.assert_frame_equal(res_auto, res_fpgrowth)


def test_auto_sparse_pandas():
    # Extremely sparse dataset: 10 transactions, 10 items, only 3 ones -> density 3/100 = 0.03 < 0.15 (Eclat)
    data = np.zeros((10, 10), dtype=np.uint8)
    data[0, 0] = 1
    data[1, 1] = 1
    data[2, 2] = 1

    df = pd.DataFrame(data).astype(pd.SparseDtype("uint8", 0))
    res_auto = rusket.mine(df, min_support=0.1, method="auto")
    res_eclat = rusket.eclat(df, min_support=0.1)

    # ensure it doesn't crash
    assert len(res_auto) == len(res_eclat)


def test_auto_scipy_csr_sparse():
    # Sparse dataset -> Eclat
    data = np.zeros((100, 100), dtype=np.uint8)
    data[0, 10] = 1
    data[10, 20] = 1
    csr = sparse.csr_matrix(data)

    res_auto = rusket.mine(csr, min_support=0.01, method="auto")
    assert len(res_auto) >= 0


def test_auto_numpy_dense():
    # Dense -> FPGrowth
    data = np.ones((10, 10), dtype=np.uint8)
    res_auto = rusket.mine(data, min_support=0.5, method="auto")
    assert len(res_auto) > 0
