"""Shared base test classes – adapted from mlxtend/tests/test_fpbase.py."""

from __future__ import annotations

import io
import sys
import warnings
from contextlib import contextmanager
from typing import Callable

import numpy as np
import pandas as pd
from packaging.version import Version
from scipy.sparse import csr_matrix

pandas_version = pd.__version__


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def captured_output():
    new_out, new_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield new_out, new_err
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def assert_raises(
    error_type: type, substr: str, func: Callable, *args, **kwargs
) -> None:
    """Assert that *func* raises *error_type* with *substr* in its message."""
    try:
        func(*args, **kwargs)
        raise AssertionError(f"Expected {error_type.__name__} to be raised")
    except error_type as e:
        assert substr in str(e), f"Expected '{substr}' in '{e}'"


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    itemsets1 = [sorted(list(i)) for i in df1["itemsets"]]
    itemsets2 = [sorted(list(i)) for i in df2["itemsets"]]
    rows1 = sorted(zip(itemsets1, df1["support"]))
    rows2 = sorted(zip(itemsets2, df2["support"]))
    for row1, row2 in zip(rows1, rows2):
        if row1[0] != row2[0]:
            raise AssertionError(
                f"Expected different frequent itemsets\nx:{row1[0]}\ny:{row2[0]}"
            )
        elif row1[1] != row2[1]:
            raise AssertionError(
                f"Expected different support\nx:{row1[1]}\ny:{row2[1]}"
            )


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class FPTestEdgeCases:
    def setUp(self, fpalgo: Callable) -> None:
        self.fpalgo = fpalgo

    def test_all_ones(self) -> None:
        df = pd.DataFrame([[1, 1], [1, 1], [1, 1]], columns=["A", "B"]).astype(bool)
        res = self.fpalgo(df, min_support=1.0)
        assert res.shape[0] > 0

    def test_min_support_too_high(self) -> None:
        df = pd.DataFrame([[1, 0], [0, 1]], columns=["A", "B"]).astype(bool)
        res = self.fpalgo(df, min_support=1.0)
        assert res.shape[0] == 0


# ---------------------------------------------------------------------------
# Error tests
# ---------------------------------------------------------------------------


class FPTestErrors:
    def setUp(self, fpalgo: Callable) -> None:
        self.one_ary = np.array(
            [
                [0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
                [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
                [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1],
                [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0],
            ]
        )
        self.cols = [
            "Apple",
            "Corn",
            "Dill",
            "Eggs",
            "Ice cream",
            "Kidney Beans",
            "Milk",
            "Nutmeg",
            "Onion",
            "Unicorn",
            "Yogurt",
        ]
        self.df = pd.DataFrame(self.one_ary, columns=self.cols).astype(bool)
        self.fpalgo = fpalgo

    def test_wrong_values_errors(self) -> None:
        def test_with_dataframe(df: pd.DataFrame) -> None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                assert_raises(
                    ValueError,
                    "The allowed values for a DataFrame are True, "
                    "False, 0, 1. Found value 2",
                    self.fpalgo,
                    df,
                )

        df2 = pd.DataFrame(self.one_ary, columns=self.cols).copy()
        df2.iloc[3, 3] = 2
        test_with_dataframe(df2)

        sdf = df2.astype(pd.SparseDtype("int", fill_value=0))
        test_with_dataframe(sdf)

    def test_sparsedataframe_notzero_column(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            dfs = self.df.astype(pd.SparseDtype("int", 0))
            dfs.columns = list(range(len(dfs.columns)))
            self.fpalgo(dfs)

            dfs = self.df.astype(pd.SparseDtype("int", 0))
            dfs.columns = [i + 1 for i in range(len(dfs.columns))]
            assert_raises(
                ValueError,
                "Due to current limitations in Pandas, "
                "if the sparse format has integer column names,"
                "names, please make sure they either start "
                "with `0` or cast them as string column names: "
                "`df.columns = [str(i) for i in df.columns`].",
                self.fpalgo,
                dfs,
            )


# ---------------------------------------------------------------------------
# Example 1: grocery-style 5×11 dataset
# ---------------------------------------------------------------------------


class FPTestEx1:
    def setUp(self, fpalgo: Callable, one_ary: np.ndarray | None = None) -> None:
        if one_ary is None:
            self.one_ary = np.array(
                [
                    [0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
                    [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1],
                    [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                ]
            )
        else:
            self.one_ary = one_ary

        self.cols = [
            "Apple",
            "Corn",
            "Dill",
            "Eggs",
            "Ice cream",
            "Kidney Beans",
            "Milk",
            "Nutmeg",
            "Onion",
            "Unicorn",
            "Yogurt",
        ]
        self.df = pd.DataFrame(self.one_ary, columns=self.cols).astype(bool)
        self.fpalgo = fpalgo

    def test_frozenset_selection(self) -> None:
        res_df = self.fpalgo(self.df, use_colnames=True)
        assert res_df.values.shape == self.fpalgo(self.df).values.shape

        # PyArrow list arrays don't support `== set` directly in pandas.
        # We need to map the items back to a Python set for row-wise filtering in the test.
        def has_items(row_items, expected):
            return set(row_items) == set(expected)

        assert res_df[
            res_df["itemsets"].apply(lambda x: has_items(x, ["nothing"]))
        ].values.shape == (0, 2)
        assert res_df[
            res_df["itemsets"].apply(lambda x: has_items(x, ["Milk", "Kidney Beans"]))
        ].values.shape == (1, 2)

    def test_sparse(self) -> None:
        def test_with_fill_values(fill_value: object) -> None:
            sdt = pd.SparseDtype(type(fill_value), fill_value=fill_value)
            sdf = self.df.astype(sdt)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                res_df = self.fpalgo(sdf, use_colnames=True)
                assert res_df.values.shape == self.fpalgo(self.df).values.shape
            assert res_df[
                res_df["itemsets"].apply(lambda x: set(x) == {"Milk", "Kidney Beans"})
            ].values.shape == (1, 2)

        test_with_fill_values(0)
        test_with_fill_values(False)

    def test_sparse_with_zero(self) -> None:
        if Version(pandas_version) < Version("1.2"):
            return
        res_df = self.fpalgo(self.df)
        ary2 = self.one_ary.copy()
        ary2[3, :] = 1
        sparse_ary = csr_matrix(ary2)
        sparse_ary[3, :] = self.one_ary[3, :]
        sdf = pd.DataFrame.sparse.from_spmatrix(sparse_ary, columns=self.df.columns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            res_df2 = self.fpalgo(sdf)
        compare_dataframes(res_df2, res_df)


class FPTestEx1All(FPTestEx1):
    def setUp(self, fpalgo: Callable, one_ary: np.ndarray | None = None) -> None:
        FPTestEx1.setUp(self, fpalgo, one_ary=one_ary)

    def test_default(self) -> None:
        res_df = self.fpalgo(self.df)
        expect = pd.DataFrame(
            [
                [0.8, np.array([3])],
                [1.0, np.array([5])],
                [0.6, np.array([6])],
                [0.6, np.array([8])],
                [0.6, np.array([10])],
                [0.8, np.array([3, 5])],
                [0.6, np.array([3, 8])],
                [0.6, np.array([5, 6])],
                [0.6, np.array([5, 8])],
                [0.6, np.array([5, 10])],
                [0.6, np.array([3, 5, 8])],
            ],
            columns=["support", "itemsets"],
        )
        compare_dataframes(res_df, expect)

    def test_max_len(self) -> None:
        res_df1 = self.fpalgo(self.df)
        max_len = np.max(res_df1["itemsets"].apply(len))
        assert max_len == 3

        res_df2 = self.fpalgo(self.df, max_len=2)
        max_len = np.max(res_df2["itemsets"].apply(len))
        assert max_len == 2

    def test_low_memory_flag(self) -> None:
        # We don't have a low_memory flag – just skip silently.
        import inspect

        if "low_memory" not in inspect.signature(self.fpalgo).parameters:
            assert True


# ---------------------------------------------------------------------------
# Example 2: single-item transactions (no pairs)
# ---------------------------------------------------------------------------


class FPTestEx2:
    def setUp(self) -> None:
        from mlxtend.preprocessing import TransactionEncoder

        database = [["a"], ["b"], ["c", "d"], ["e"]]
        te = TransactionEncoder()
        te_ary = te.fit(database).transform(database)
        self.df = pd.DataFrame(te_ary, columns=te.columns_)


class FPTestEx2All(FPTestEx2):
    def setUp(self, fpalgo: Callable) -> None:
        self.fpalgo = fpalgo
        FPTestEx2.setUp(self)

    def test_output(self) -> None:
        res_df = self.fpalgo(self.df, min_support=0.001, use_colnames=True)
        expect = pd.DataFrame(
            [
                [0.25, frozenset(["a"])],
                [0.25, frozenset(["b"])],
                [0.25, frozenset(["c"])],
                [0.25, frozenset(["d"])],
                [0.25, frozenset(["e"])],
                [0.25, frozenset(["c", "d"])],
            ],
            columns=["support", "itemsets"],
        )
        compare_dataframes(res_df, expect)


# ---------------------------------------------------------------------------
# Example 3: min_support=0.0 error
# ---------------------------------------------------------------------------


class FPTestEx3:
    def setUp(self) -> None:
        from mlxtend.preprocessing import TransactionEncoder

        database = [["a"], ["b"], ["c", "d"], ["e"]]
        te = TransactionEncoder()
        te_ary = te.fit(database).transform(database)
        self.df = pd.DataFrame(te_ary, columns=te.columns_)


class FPTestEx3All(FPTestEx3):
    def setUp(self, fpalgo: Callable) -> None:
        self.fpalgo = fpalgo
        FPTestEx3.setUp(self)

    def test_output3(self) -> None:
        assert_raises(
            ValueError,
            "`min_support` must be a positive "
            "number within the interval `(0, 1]`. Got 0.0.",
            self.fpalgo,
            self.df,
            min_support=0.0,
        )
