"""Tests for rusket.from_transactions — type-preserving round-trips."""

from __future__ import annotations

import pandas as pd
import pytest

import rusket
from rusket import from_transactions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TXN_LONG = {
    "order_id": [1, 1, 1, 2, 2, 3],
    "item": [3, 4, 5, 3, 5, 8],
}

_TXN_STR = {
    "txn": [1, 1, 2, 2, 3],
    "item": ["bread", "milk", "bread", "eggs", "milk"],
}


# ---------------------------------------------------------------------------
# List-of-lists input → always pd.DataFrame
# ---------------------------------------------------------------------------


class TestFromList:
    def test_basic(self) -> None:
        transactions = [[3, 4, 5], [3, 5], [8]]
        result = from_transactions(transactions)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 4)
        assert result.dtypes.apply(pd.api.types.is_bool_dtype).all()

        assert result.iloc[0].sum() == 3
        assert result.iloc[2].sum() == 1

    def test_string_items(self) -> None:
        transactions = [["bread", "milk"], ["bread", "eggs"], ["milk"]]
        result = from_transactions(transactions)

        assert set(result.columns) == {"bread", "milk", "eggs"}
        assert result.shape == (3, 3)
        assert result.iloc[0]["bread"] == True  # noqa: E712

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
        assert result.values.sum() == 3


# ---------------------------------------------------------------------------
# Pandas → Pandas
# ---------------------------------------------------------------------------


class TestFromPandas:
    def test_returns_pandas(self) -> None:
        df = pd.DataFrame(_TXN_LONG)
        result = from_transactions(df)
        assert isinstance(result, pd.DataFrame)

    def test_shape(self) -> None:
        df = pd.DataFrame(_TXN_LONG)
        result = from_transactions(df)
        assert result.shape == (3, 4)

    def test_bool_dtype(self) -> None:
        df = pd.DataFrame(_TXN_LONG)
        result = from_transactions(df)
        assert result.dtypes.apply(pd.api.types.is_bool_dtype).all()

    def test_values_correct(self) -> None:
        df = pd.DataFrame({"txn": [1, 1, 2], "item": ["a", "b", "a"]})
        result = from_transactions(df)
        # txn1 has a,b — txn2 has a only
        assert sorted(result.columns) == ["a", "b"]
        row1 = result.iloc[0]
        row2 = result.iloc[1]
        assert bool(row1["a"]) and bool(row1["b"])
        assert bool(row2["a"]) and not bool(row2["b"])

    def test_custom_columns(self) -> None:
        df = pd.DataFrame({"product": ["a", "b", "a", "c"], "basket": [1, 1, 2, 2]})
        result = from_transactions(df, transaction_col="basket", item_col="product")
        assert result.shape == (2, 3)

    def test_string_items(self) -> None:
        df = pd.DataFrame(_TXN_STR)
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
# Polars → Polars
# ---------------------------------------------------------------------------


class TestFromPolars:
    def test_returns_polars(self) -> None:
        pl = pytest.importorskip("polars")
        df = pl.DataFrame(_TXN_LONG)
        result = from_transactions(df)
        assert isinstance(result, pl.DataFrame)

    def test_shape(self) -> None:
        pl = pytest.importorskip("polars")
        df = pl.DataFrame(_TXN_LONG)
        result = from_transactions(df)
        assert result.shape == (3, 4)

    def test_bool_dtype(self) -> None:
        pl = pytest.importorskip("polars")
        df = pl.DataFrame(_TXN_LONG)
        result = from_transactions(df)
        assert all(result[col].dtype == pl.Boolean for col in result.columns)

    def test_values_correct(self) -> None:
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({"txn": [1, 1, 2], "item": ["a", "b", "a"]})
        result = from_transactions(df)
        assert set(result.columns) == {"a", "b"}
        # Each row should have exactly the right booleans
        pd_equiv = result.to_pandas()
        assert bool(pd_equiv.iloc[0]["a"]) and bool(pd_equiv.iloc[0]["b"])
        assert bool(pd_equiv.iloc[1]["a"]) and not bool(pd_equiv.iloc[1]["b"])

    def test_custom_columns(self) -> None:
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({"product": ["a", "b", "a", "c"], "basket": [1, 1, 2, 2]})
        result = from_transactions(df, transaction_col="basket", item_col="product")
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (2, 3)

    def test_matches_pandas_result(self) -> None:
        """Polars and Pandas paths must produce equivalent boolean matrices."""
        pl = pytest.importorskip("polars")
        data = {"txn": [1, 1, 2, 2, 3], "item": ["bread", "milk", "bread", "eggs", "milk"]}
        pd_result = from_transactions(pd.DataFrame(data))
        pl_result = from_transactions(pl.DataFrame(data)).to_pandas()

        # Sort columns so comparison is stable
        pd_result = pd_result[sorted(pd_result.columns)].reset_index(drop=True)
        pl_result = pl_result[sorted(pl_result.columns)].reset_index(drop=True)

        pd.testing.assert_frame_equal(pd_result.astype(bool), pl_result.astype(bool), check_dtype=False)

    def test_from_polars_helper(self) -> None:
        pl = pytest.importorskip("polars")
        from rusket import from_polars

        df = pl.DataFrame({"txn": [1, 1, 2, 2], "item": ["a", "b", "a", "c"]})
        result = from_polars(df)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (2, 3)


# ---------------------------------------------------------------------------
# Spark → Spark
# ---------------------------------------------------------------------------


class TestFromSpark:
    @pytest.fixture(scope="class")
    def spark(self):  # type: ignore[no-untyped-def]
        pytest.importorskip("pyspark")
        from pyspark.sql import SparkSession

        session = (
            SparkSession.builder.master("local[1]")
            .appName("rusket-test-transactions")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        yield session
        session.stop()

    def test_returns_spark_dataframe(self, spark) -> None:  # type: ignore[no-untyped-def]
        df = spark.createDataFrame([(1, "x"), (1, "y"), (2, "x"), (3, "z")], ["txn", "item"])
        result = from_transactions(df)
        assert type(result).__name__ == "DataFrame"
        assert getattr(type(result), "__module__", "").startswith("pyspark")

    def test_columns(self, spark) -> None:  # type: ignore[no-untyped-def]
        df = spark.createDataFrame([(1, "x"), (1, "y"), (2, "x"), (3, "z")], ["txn", "item"])
        result = from_transactions(df)
        assert set(result.columns) == {"x", "y", "z"}

    def test_bool_dtype(self, spark) -> None:  # type: ignore[no-untyped-def]
        from pyspark.sql import types as T

        df = spark.createDataFrame([(1, "x"), (1, "y"), (2, "x")], ["txn", "item"])
        result = from_transactions(df)
        for field in result.schema.fields:
            assert isinstance(field.dataType, T.BooleanType), (
                f"Column {field.name!r} has type {field.dataType}, expected BooleanType"
            )

    def test_values_correct(self, spark) -> None:  # type: ignore[no-untyped-def]
        df = spark.createDataFrame([(1, "a"), (1, "b"), (2, "a")], ["txn", "item"])
        result = from_transactions(df)
        pd_result = result.toPandas().sort_values(list(result.columns)).reset_index(drop=True)
        # Check all values are booleans
        for col in pd_result.columns:
            assert pd_result[col].dtype == bool or pd.api.types.is_bool_dtype(pd_result[col])

    def test_from_spark_helper(self, spark) -> None:  # type: ignore[no-untyped-def]
        from rusket import from_spark

        df = spark.createDataFrame([(1, "a"), (2, "b")], ["txn", "item"])
        result = from_spark(df)
        assert type(result).__name__ == "DataFrame"
        assert getattr(type(result), "__module__", "").startswith("pyspark")
        assert set(result.columns) == {"a", "b"}

    def test_matches_pandas_values(self, spark) -> None:  # type: ignore[no-untyped-def]
        """Spark and Pandas paths must produce equivalent boolean matrices."""
        data = [(1, "bread"), (1, "milk"), (2, "bread"), (2, "eggs"), (3, "milk")]
        spark_df = spark.createDataFrame(data, ["txn", "item"])
        pd_df = pd.DataFrame(data, columns=["txn", "item"])

        spark_result = from_transactions(spark_df).toPandas()
        pd_result = from_transactions(pd_df)

        common_cols = sorted(set(spark_result.columns) & set(pd_result.columns))
        pd.testing.assert_frame_equal(
            spark_result[common_cols].astype(bool).sort_values(common_cols).reset_index(drop=True),
            pd_result[common_cols].astype(bool).sort_values(common_cols).reset_index(drop=True),
            check_dtype=False,
        )


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
        df = pd.DataFrame(_TXN_LONG)
        ohe = from_transactions(df)
        freq = rusket.fpgrowth(ohe, min_support=0.3, use_colnames=True)
        assert len(freq) > 0

    def test_polars_e2e(self) -> None:
        pl = pytest.importorskip("polars")
        df = pl.DataFrame(_TXN_LONG)
        ohe = from_transactions(df)
        # fpgrowth/eclat accept polars too — or convert
        pd_ohe = ohe.to_pandas()
        freq = rusket.fpgrowth(pd_ohe, min_support=0.3, use_colnames=True)
        assert len(freq) > 0

    def test_type_error(self) -> None:
        with pytest.raises(TypeError, match="Expected"):
            from_transactions(42)  # type: ignore[arg-type]
