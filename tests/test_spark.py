import pandas as pd
import pytest

from rusket.spark import mine_grouped, to_spark

pyspark = pytest.importorskip("pyspark")


@pytest.fixture(scope="module")
def spark_session():
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName("rusket-test")
        .master("local[2]")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    yield spark
    spark.stop()


def test_to_spark(spark_session) -> None:
    df = pd.DataFrame({"store_id": [1, 1], "bread": [1, 0]})
    # Test valid conversion from pandas to Spark
    spark_df = to_spark(spark_session, df)

    # Let's count row instances
    assert spark_df.count() == 2


def test_mine_grouped(spark_session) -> None:
    df = pd.DataFrame(
        {
            "store_id": ["A", "A", "A", "B", "B"],
            "bread": [1, 1, 0, 1, 1],
            "butter": [1, 0, 1, 1, 0],
            "milk": [1, 1, 1, 0, 1],
        }
    )

    spark_df = to_spark(spark_session, df)

    result = mine_grouped(
        spark_df,
        group_col="store_id",
        min_support=0.5,
    )

    # Should get a PySpark DataFrame representing Association Rules across clusters
    assert isinstance(result, pyspark.sql.DataFrame)

    pd_result = result.toPandas()

    assert "store_id" in pd_result.columns
    assert "support" in pd_result.columns
    assert "itemsets" in pd_result.columns

    # Store A and B should both have results
    groups = pd_result["store_id"].unique()
    assert "A" in groups
    assert "B" in groups
