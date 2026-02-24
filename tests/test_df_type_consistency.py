"""Type-roundtrip consistency tests for all rusket miners.

Contract: every public DataFrame-returning method must echo the input type.
  - pandas.DataFrame  → pandas.DataFrame
  - polars.DataFrame  → polars.DataFrame
  - pyspark.sql.DataFrame → pyspark.sql.DataFrame

Covers FPGrowth, Eclat, FIN, LCM, and AutoMiner for:
  - mine()
  - association_rules()
  - mine_grouped() + fit_grouped()
"""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from rusket.eclat import Eclat
from rusket.fin import FIN
from rusket.fpgrowth import FPGrowth
from rusket.lcm import LCM
from rusket.mine import AutoMiner

# ---------------------------------------------------------------------------
# Long-format transactional data (used for mine / association_rules)
# ---------------------------------------------------------------------------

_ROWS: list[tuple[str, str]] = [
    ("t1", "bread"), ("t1", "milk"), ("t1", "butter"),
    ("t2", "bread"), ("t2", "butter"), ("t2", "milk"),
    ("t3", "milk"), ("t3", "butter"),
    ("t4", "bread"), ("t4", "milk"),
    ("t5", "bread"), ("t5", "butter"),
    ("t6", "bread"), ("t6", "milk"), ("t6", "butter"),
    ("t7", "milk"), ("t7", "butter"),
    ("t8", "bread"), ("t8", "milk"),
]
_PANDAS_DF = pd.DataFrame(_ROWS, columns=["txn_id", "item"])
_POLARS_DF = pl.DataFrame({"txn_id": [t for t, _ in _ROWS], "item": [i for _, i in _ROWS]})

# ---------------------------------------------------------------------------
# One-hot grouped data (used for mine_grouped / fit_grouped)
# Each row is a transaction; store_id is the group column.
# ---------------------------------------------------------------------------

_ONEHOT_PD = pd.DataFrame(
    {
        "store_id": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "bread":    [  1,   1,   0,   1,   1,   1,   1,   0],
        "butter":   [  1,   0,   1,   1,   1,   1,   0,   0],
        "milk":     [  1,   1,   1,   0,   1,   1,   1,   1],
    }
)
_ONEHOT_PL = pl.from_pandas(_ONEHOT_PD)

_MINERS = [FPGrowth, Eclat, FIN, LCM, AutoMiner]
_MINER_IDS = [cls.__name__ for cls in _MINERS]

_MIN_SUPPORT = 0.4
_MIN_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_from_transactions(MinerCls: type, data: object) -> object:
    """Build a miner from long-format transactional data."""
    return MinerCls.from_transactions(
        data,
        transaction_col="txn_id",
        item_col="item",
        min_support=_MIN_SUPPORT,
    )


def _build_from_onehot(MinerCls: type, data: object) -> object:
    """Build a miner directly from a one-hot grouped DataFrame."""
    return MinerCls(data, min_support=_MIN_SUPPORT)  # type: ignore[call-arg]


# ===========================================================================
# mine() — Pandas → Pandas
# ===========================================================================


@pytest.mark.parametrize("MinerCls", _MINERS, ids=_MINER_IDS)
def test_mine_pandas_returns_pandas(MinerCls: type) -> None:
    result = _build_from_transactions(MinerCls, _PANDAS_DF).mine()  # type: ignore[union-attr]
    assert isinstance(result, pd.DataFrame), (
        f"{MinerCls.__name__}.mine() → {type(result).__name__}, expected pandas.DataFrame"
    )
    assert {"support", "itemsets"} <= set(result.columns)
    assert len(result) > 0


# ===========================================================================
# mine() — Polars → Polars
# ===========================================================================


@pytest.mark.parametrize("MinerCls", _MINERS, ids=_MINER_IDS)
def test_mine_polars_returns_polars(MinerCls: type) -> None:
    result = _build_from_transactions(MinerCls, _POLARS_DF).mine()  # type: ignore[union-attr]
    assert isinstance(result, pl.DataFrame), (
        f"{MinerCls.__name__}.mine() → {type(result).__name__}, expected polars.DataFrame"
    )
    assert {"support", "itemsets"} <= set(result.columns)
    assert result.height > 0


# ===========================================================================
# association_rules() — Pandas → Pandas
# ===========================================================================


@pytest.mark.parametrize("MinerCls", _MINERS, ids=_MINER_IDS)
def test_association_rules_pandas_returns_pandas(MinerCls: type) -> None:
    result = _build_from_transactions(MinerCls, _PANDAS_DF).association_rules(  # type: ignore[union-attr]
        metric="confidence", min_threshold=_MIN_THRESHOLD
    )
    assert isinstance(result, pd.DataFrame), (
        f"{MinerCls.__name__}.association_rules() → {type(result).__name__}, expected pandas.DataFrame"
    )
    assert {"antecedents", "consequents", "confidence"} <= set(result.columns)


# ===========================================================================
# association_rules() — Polars → Polars
# ===========================================================================


@pytest.mark.parametrize("MinerCls", _MINERS, ids=_MINER_IDS)
def test_association_rules_polars_returns_polars(MinerCls: type) -> None:
    result = _build_from_transactions(MinerCls, _POLARS_DF).association_rules(  # type: ignore[union-attr]
        metric="confidence", min_threshold=_MIN_THRESHOLD
    )
    assert isinstance(result, pl.DataFrame), (
        f"{MinerCls.__name__}.association_rules() → {type(result).__name__}, expected polars.DataFrame"
    )
    assert {"antecedents", "consequents", "confidence"} <= set(result.columns)


# ===========================================================================
# mine_grouped() — Pandas → Pandas
# mine_grouped reads self.data (the one-hot matrix with group_col).
# ===========================================================================


@pytest.mark.parametrize("MinerCls", _MINERS, ids=_MINER_IDS)
def test_mine_grouped_pandas_returns_pandas(MinerCls: type) -> None:
    model = _build_from_onehot(MinerCls, _ONEHOT_PD)
    result = model.mine_grouped("store_id", min_support=0.5)  # type: ignore[union-attr]
    assert isinstance(result, pd.DataFrame), (
        f"{MinerCls.__name__}.mine_grouped() → {type(result).__name__}, expected pandas.DataFrame"
    )
    assert {"store_id", "support", "itemsets"} <= set(result.columns)
    assert len(result) > 0


# ===========================================================================
# mine_grouped() — Polars → Polars
# ===========================================================================


@pytest.mark.parametrize("MinerCls", _MINERS, ids=_MINER_IDS)
def test_mine_grouped_polars_returns_polars(MinerCls: type) -> None:
    model = _build_from_onehot(MinerCls, _ONEHOT_PL)
    result = model.mine_grouped("store_id", min_support=0.5)  # type: ignore[union-attr]
    assert isinstance(result, pl.DataFrame), (
        f"{MinerCls.__name__}.mine_grouped() → {type(result).__name__}, expected polars.DataFrame"
    )
    assert {"store_id", "support", "itemsets"} <= set(result.columns)
    assert result.height > 0


# ===========================================================================
# fit_grouped() — Pandas → caches pandas result, returns self
# ===========================================================================


@pytest.mark.parametrize("MinerCls", _MINERS, ids=_MINER_IDS)
def test_fit_grouped_pandas_caches_pandas(MinerCls: type) -> None:
    model = _build_from_onehot(MinerCls, _ONEHOT_PD)
    returned = model.fit_grouped("store_id", min_support=0.5)  # type: ignore[union-attr]
    assert returned is model, "fit_grouped() must return self for chaining"
    cached = model._grouped_result  # type: ignore[union-attr]
    assert isinstance(cached, pd.DataFrame), (
        f"{MinerCls.__name__}.fit_grouped() cached {type(cached).__name__}, expected pandas.DataFrame"
    )
    assert {"store_id", "support", "itemsets"} <= set(cached.columns)
    assert len(cached) > 0


# ===========================================================================
# fit_grouped() — Polars → caches polars result, returns self
# ===========================================================================


@pytest.mark.parametrize("MinerCls", _MINERS, ids=_MINER_IDS)
def test_fit_grouped_polars_caches_polars(MinerCls: type) -> None:
    model = _build_from_onehot(MinerCls, _ONEHOT_PL)
    returned = model.fit_grouped("store_id", min_support=0.5)  # type: ignore[union-attr]
    assert returned is model, "fit_grouped() must return self for chaining"
    cached = model._grouped_result  # type: ignore[union-attr]
    assert isinstance(cached, pl.DataFrame), (
        f"{MinerCls.__name__}.fit_grouped() cached {type(cached).__name__}, expected polars.DataFrame"
    )
    assert {"store_id", "support", "itemsets"} <= set(cached.columns)
    assert cached.height > 0


# ===========================================================================
# Spark tests — skipped unless PySpark + Java are available
# ===========================================================================

pyspark = pytest.importorskip("pyspark", reason="PySpark not installed")


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    try:
        session = (
            SparkSession.builder.appName("rusket-type-consistency-test")
            .master("local[1]")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
            .getOrCreate()
        )
        yield session
        session.stop()
    except Exception as e:
        if "JAVA_GATEWAY_EXITED" in str(e) or "Java gateway process exited" in str(e):
            pytest.skip("Java runtime not available; PySpark requires Java.")
        else:
            raise


# mine() — Spark → Spark
@pytest.mark.parametrize("MinerCls", _MINERS, ids=_MINER_IDS)
def test_mine_spark_returns_spark(spark: object, MinerCls: type) -> None:
    import pyspark.sql

    spark_df = spark.createDataFrame(_PANDAS_DF)  # type: ignore[union-attr]
    result = _build_from_transactions(MinerCls, spark_df).mine()  # type: ignore[union-attr]

    assert isinstance(result, pyspark.sql.DataFrame), (
        f"{MinerCls.__name__}.mine() → {type(result).__name__}, expected pyspark.sql.DataFrame"
    )
    pd_result = result.toPandas()
    assert {"support", "itemsets"} <= set(pd_result.columns)
    assert len(pd_result) > 0


# association_rules() — Spark → Spark
@pytest.mark.parametrize("MinerCls", _MINERS, ids=_MINER_IDS)
def test_association_rules_spark_returns_spark(spark: object, MinerCls: type) -> None:
    import pyspark.sql

    spark_df = spark.createDataFrame(_PANDAS_DF)  # type: ignore[union-attr]
    result = _build_from_transactions(MinerCls, spark_df).association_rules(  # type: ignore[union-attr]
        metric="confidence", min_threshold=_MIN_THRESHOLD
    )
    assert isinstance(result, pyspark.sql.DataFrame), (
        f"{MinerCls.__name__}.association_rules() → {type(result).__name__}, expected pyspark.sql.DataFrame"
    )
    pd_result = result.toPandas()
    assert {"antecedents", "consequents", "confidence"} <= set(pd_result.columns)


# mine_grouped() — Spark → Spark
def test_mine_grouped_spark_returns_spark(spark: object) -> None:
    """mine_grouped() on a Spark one-hot DataFrame must return Spark."""
    import pyspark.sql

    spark_onehot = spark.createDataFrame(_ONEHOT_PD)  # type: ignore[union-attr]
    model = _build_from_onehot(FPGrowth, spark_onehot)
    result = model.mine_grouped("store_id", min_support=0.5)  # type: ignore[union-attr]

    assert isinstance(result, pyspark.sql.DataFrame), (
        f"mine_grouped() → {type(result).__name__}, expected pyspark.sql.DataFrame"
    )
    pd_result = result.toPandas()
    assert {"store_id", "support", "itemsets"} <= set(pd_result.columns)
    assert len(pd_result) > 0


# fit_grouped() — Spark → Spark, caches result and returns self
def test_fit_grouped_spark_caches_spark(spark: object) -> None:
    import pyspark.sql

    spark_onehot = spark.createDataFrame(_ONEHOT_PD)  # type: ignore[union-attr]
    model = _build_from_onehot(FPGrowth, spark_onehot)
    returned = model.fit_grouped("store_id", min_support=0.5)  # type: ignore[union-attr]

    assert returned is model, "fit_grouped() must return self for chaining"
    cached = model._grouped_result  # type: ignore[union-attr]
    assert isinstance(cached, pyspark.sql.DataFrame), (
        f"fit_grouped() cached {type(cached).__name__}, expected pyspark.sql.DataFrame"
    )
    pd_result = cached.toPandas()
    assert {"store_id", "support", "itemsets"} <= set(pd_result.columns)
