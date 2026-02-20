"""Tests for rusket.from_transactions."""

from __future__ import annotations

import pandas as pd
import pytest

import rusket
from rusket import from_transactions


# ---------------------------------------------------------------------------
# List-of-lists input
# ---------------------------------------------------------------------------


class TestFromList:
    def test_basic(self) -> None:
        transactions = [[3, 4, 5], [3, 5], [8]]
        result = from_transactions(transactions)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 4)  # 3 transactions, 4 unique items
        assert result.dtypes.apply(pd.api.types.is_bool_dtype).all()

        # Transaction 0 should have items 3, 4, 5
        assert result.iloc[0].sum() == 3
        # Transaction 2 should have only item 8
        assert result.iloc[2].sum() == 1

    def test_string_items(self) -> None:
        transactions = [["bread", "milk"], ["bread", "eggs"], ["milk"]]
        result = from_transactions(transactions)

        assert set(result.columns) == {"bread", "milk", "eggs"}
        assert result.shape == (3, 3)
        assert result.iloc[0]["bread"] == True  # noqa: E712
        assert result.iloc[0]["eggs"] == False  # noqa: E712

    def test_empty_list(self) -> None:
        result = from_transactions([])
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (0, 0)

    def test_single_transaction(self) -> None:
        result = from_transactions([[1, 2, 3]])
        assert result.shape == (1, 3)
        assert result.iloc[0].all()

    def test_single_item_per_transaction(self) -> None:
        result = from_transactions([[1], [2], [3]])
        assert result.shape == (3, 3)
        assert result.values.sum() == 3  # exactly one True per row


# ---------------------------------------------------------------------------
# Pandas DataFrame input
# ---------------------------------------------------------------------------


class TestFromPandas:
    def test_basic(self) -> None:
        df = pd.DataFrame(
            {"order_id": [1, 1, 1, 2, 2, 3], "item": [3, 4, 5, 3, 5, 8]}
        )
        result = from_transactions(df)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 4)  # 3 orders, 4 unique items
        assert result.dtypes.apply(pd.api.types.is_bool_dtype).all()

    def test_custom_columns(self) -> None:
        df = pd.DataFrame(
            {"product": ["a", "b", "a", "c"], "basket": [1, 1, 2, 2]}
        )
        result = from_transactions(df, transaction_col="basket", item_col="product")
        assert result.shape == (2, 3)

    def test_string_items(self) -> None:
        df = pd.DataFrame(
            {
                "txn": [1, 1, 2, 2, 3],
                "item": ["bread", "milk", "bread", "eggs", "milk"],
            }
        )
        result = from_transactions(df)
        assert set(result.columns) == {"bread", "milk", "eggs"}
        assert result.shape == (3, 3)

    def test_too_few_columns(self) -> None:
        df = pd.DataFrame({"only_one": [1, 2, 3]})
        with pytest.raises(ValueError, match="at least 2 columns"):
            from_transactions(df)

    def test_missing_column(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError, match="not found"):
            from_transactions(df, transaction_col="missing")


# ---------------------------------------------------------------------------
# Polars DataFrame input
# ---------------------------------------------------------------------------


class TestFromPolars:
    def test_basic(self) -> None:
        pl = pytest.importorskip("polars")
        df = pl.DataFrame(
            {"order_id": [1, 1, 1, 2, 2, 3], "item": [3, 4, 5, 3, 5, 8]}
        )
        result = from_transactions(df)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 4)
        assert result.dtypes.apply(pd.api.types.is_bool_dtype).all()


# ---------------------------------------------------------------------------
# End-to-end: from_transactions → fpgrowth / eclat
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_fpgrowth(self) -> None:
        transactions = [
            ["bread", "milk", "butter"],
            ["bread", "milk"],
            ["bread", "eggs"],
            ["milk", "eggs"],
            ["bread", "milk", "eggs"],
        ]
        ohe = from_transactions(transactions)
        freq = rusket.fpgrowth(ohe, min_support=0.4, use_colnames=True)
        assert len(freq) > 0
        assert "support" in freq.columns
        assert "itemsets" in freq.columns

    def test_eclat(self) -> None:
        transactions = [
            ["bread", "milk", "butter"],
            ["bread", "milk"],
            ["bread", "eggs"],
            ["milk", "eggs"],
            ["bread", "milk", "eggs"],
        ]
        ohe = from_transactions(transactions)
        freq = rusket.eclat(ohe, min_support=0.4, use_colnames=True)
        assert len(freq) > 0

    def test_pandas_e2e(self) -> None:
        df = pd.DataFrame(
            {
                "order_id": [1, 1, 1, 2, 2, 3],
                "item": [3, 4, 5, 3, 5, 8],
            }
        )
        ohe = from_transactions(df)
        freq = rusket.fpgrowth(ohe, min_support=0.3, use_colnames=True)
        assert len(freq) > 0

    def test_type_error(self) -> None:
        with pytest.raises(TypeError, match="Expected"):
            from_transactions(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Explicit from_pandas / from_polars / from_spark wrappers
# ---------------------------------------------------------------------------


class TestExplicitHelpers:
    def test_from_pandas(self) -> None:
        from rusket import from_pandas

        df = pd.DataFrame({"txn": [1, 1, 2, 2], "item": ["a", "b", "a", "c"]})
        result = from_pandas(df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 3)
        assert result.dtypes.apply(pd.api.types.is_bool_dtype).all()

    def test_from_polars(self) -> None:
        pl = pytest.importorskip("polars")
        from rusket import from_polars

        df = pl.DataFrame({"txn": [1, 1, 2, 2], "item": ["a", "b", "a", "c"]})
        result = from_polars(df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 3)

    def test_from_spark_alias(self) -> None:
        """from_spark is just an alias that delegates to from_transactions."""
        from rusket import from_spark

        # We can't test with a real Spark DF without PySpark, but we can
        # verify it works with a Pandas DF (which goes through the same path).
        df = pd.DataFrame({"txn": [1, 1, 2], "item": ["x", "y", "x"]})
        result = from_spark(df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)
