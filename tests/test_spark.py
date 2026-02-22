import numpy as np
import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")
from rusket.spark import mine_grouped, to_spark  # noqa: E402


@pytest.fixture(scope="module")
def spark_session():
    from pyspark.sql import SparkSession

    try:
        spark = (
            SparkSession.builder.appName("rusket-test")
            .master("local[2]")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            # Force Arrow Fallback in PySpark 3.4+ so our new RDD map works reliably
            .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
            .getOrCreate()
        )
        yield spark
        spark.stop()
    except Exception as e:
        if "JAVA_GATEWAY_EXITED" in str(e) or "Java gateway process exited" in str(e):
            pytest.skip("Java runtime is not working. PySpark requires Java to run.", allow_module_level=True)
        else:
            raise e


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


def test_spark_als(spark_session) -> None:
    from rusket.als import ALS

    df = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 2],
            "item_id": [10, 20, 10, 30, 20],
            "rating": [5.0, 3.0, 4.0, 5.0, 1.0],
        }
    )
    spark_df = to_spark(spark_session, df)

    model = ALS(factors=4, iterations=3, seed=42)
    # The spark_df will be natively ingested via arrow buffers
    model.fit_transactions(spark_df, user_col="user_id", item_col="item_id", rating_col="rating")

    assert model.fitted
    assert model.user_factors.shape[0] == 3
    assert model.item_factors.shape[0] == 3

    recs, scores = model.recommend_items(user_id=0, n=2)
    assert len(recs) > 0


def test_spark_bpr(spark_session) -> None:
    from rusket.bpr import BPR

    df = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 2],
            "item_id": [10, 20, 10, 30, 20],
        }
    )
    spark_df = to_spark(spark_session, df)

    model = BPR(factors=4, iterations=3, seed=42)
    # The spark_df will be natively ingested via arrow buffers
    model.fit_transactions(spark_df, user_col="user_id", item_col="item_id")

    assert model.fitted
    assert model.user_factors.shape[0] == 3
    assert model.item_factors.shape[0] == 3

    recs, scores = model.recommend_items(user_id=1, n=2)
    assert len(recs) > 0


def test_spark_prefixspan(spark_session) -> None:
    from rusket.prefixspan import prefixspan, sequences_from_event_log

    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 3, 3],
            "time": [10, 20, 30, 10, 20, 10, 20],
            "item": ["A", "B", "A", "B", "A", "A", "C"],
        }
    )

    spark_df = to_spark(spark_session, df)

    # Use our Spark Arrow converter natively
    seqs, mapping_dict = sequences_from_event_log(df=spark_df, user_col="user_id", time_col="time", item_col="item")

    # Seq 1: A, B, A (0, 1, 0)
    # Seq 2: B, A (1, 0)
    # Seq 3: A, C (0, 2)
    assert len(seqs) == 3

    # Run mining
    freq = prefixspan(seqs, min_support=2, max_len=2)

    # Ensure decoding dictionary maps integer IDs back
    # The longest frequent itemset with support >= 2 is [A] (3), [B] (2), [B, A] (2)
    def decode_seq(s):
        return [mapping_dict[i] for i in s]

    freq["sequence_str"] = freq["sequence"].apply(decode_seq)

    # Check [B, A] support
    ba = freq[freq["sequence_str"].apply(lambda x: x == ["B", "A"])].iloc[0]
    assert ba["support"] == 2


def test_prefixspan_grouped(spark_session) -> None:
    from rusket.spark import prefixspan_grouped

    df = pd.DataFrame(
        {
            "store_id": ["A", "A", "A", "B", "B", "B"],
            "user_id": [1, 1, 1, 2, 2, 2],
            "time": [10, 20, 30, 10, 20, 30],
            "item": ["X", "Y", "X", "Y", "Z", "Y"],
        }
    )

    spark_df = to_spark(spark_session, df)

    result = prefixspan_grouped(
        spark_df,
        group_col="store_id",
        user_col="user_id",
        time_col="time",
        item_col="item",
        min_support=1,
    )

    import pyspark

    assert isinstance(result, pyspark.sql.DataFrame)

    pd_result = result.toPandas()

    assert "store_id" in pd_result.columns
    assert "support" in pd_result.columns
    assert "sequence" in pd_result.columns

    # Store A should have X -> Y -> X
    a_res = pd_result[pd_result["store_id"] == "A"]
    assert len(a_res) > 0

    # Store B should have Y -> Z -> Y
    b_res = pd_result[pd_result["store_id"] == "B"]
    assert len(b_res) > 0


def test_hupm_grouped(spark_session) -> None:
    from rusket.spark import hupm_grouped

    df = pd.DataFrame(
        {
            "store_id": ["A", "A", "A", "A", "B", "B", "B"],
            "transaction_id": [1, 1, 2, 2, 3, 3, 4],
            "item_id": [10, 20, 10, 30, 10, 20, 30],
            "utility": [5.0, 10.0, 5.0, 15.0, 50.0, 2.0, 15.0],
        }
    )

    spark_df = to_spark(spark_session, df)

    result = hupm_grouped(
        spark_df,
        group_col="store_id",
        transaction_col="transaction_id",
        item_col="item_id",
        utility_col="utility",
        min_utility=15.0,
    )

    import pyspark

    assert isinstance(result, pyspark.sql.DataFrame)

    pd_result = result.toPandas()

    assert "store_id" in pd_result.columns
    assert "utility" in pd_result.columns
    assert "itemset" in pd_result.columns

    a_res = pd_result[pd_result["store_id"] == "A"]
    b_res = pd_result[pd_result["store_id"] == "B"]

    assert len(a_res) > 0
    assert len(b_res) > 0


def test_rules_grouped(spark_session) -> None:
    from rusket.spark import mine_grouped, rules_grouped

    df = pd.DataFrame(
        {
            "store_id": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "bread": [1, 1, 0, 1, 1, 1, 1, 0],
            "butter": [1, 0, 1, 1, 1, 1, 0, 0],
            "milk": [1, 1, 1, 0, 1, 1, 1, 1],
        }
    )

    spark_df = to_spark(spark_session, df)

    freq_df = mine_grouped(
        spark_df,
        group_col="store_id",
        min_support=0.5,
    )

    rules_df = rules_grouped(
        freq_df,
        group_col="store_id",
        num_itemsets={"A": 4, "B": 4},
        metric="confidence",
        min_threshold=0.5,
    )

    import pyspark

    assert isinstance(rules_df, pyspark.sql.DataFrame)

    pd_rules = rules_df.toPandas()

    assert "store_id" in pd_rules.columns
    assert "antecedents" in pd_rules.columns
    assert "consequents" in pd_rules.columns
    assert "confidence" in pd_rules.columns

    # Check that there are some rules
    assert len(pd_rules) > 0

    # Store A and B should both have rules mapping Bread -> Milk or similar
    a_rules = pd_rules[pd_rules["store_id"] == "A"]
    b_rules = pd_rules[pd_rules["store_id"] == "B"]

    assert len(a_rules) > 0
    assert len(b_rules) > 0


def test_recommend_batches(spark_session) -> None:
    from rusket.als import ALS
    from rusket.recommend import Recommender
    from rusket.spark import recommend_batches

    # Train a quick ALS model locally
    train_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "item_id": [10, 20, 10, 30, 20],
            "rating": [5.0, 3.0, 4.0, 5.0, 1.0],
        }
    )

    model = ALS(factors=4, iterations=3, seed=42)
    # Turn off the ALS warning simply for hygiene
    import warnings

    warnings.simplefilter("ignore", DeprecationWarning)

    model.fit_transactions(train_df, user_col="user_id", item_col="item_id", rating_col="rating")
    recommender = Recommender(als_model=model)

    # Now pretend we have a Spark DF of user histories to batch score
    batch_df = pd.DataFrame(
        {
            "user_id": ["0", "1", "2"],
        }
    )
    spark_batch = to_spark(spark_session, batch_df)

    # We can pass the `recommender` directly
    result = recommend_batches(
        spark_batch,
        model=recommender,
        user_col="user_id",
        k=2,
    )

    import pyspark

    assert isinstance(result, pyspark.sql.DataFrame)

    pd_result = result.toPandas()

    assert "user_id" in pd_result.columns
    assert "recommended_items" in pd_result.columns

    # Total 3 users processed
    assert len(pd_result) == 3

    # Ensure it parsed the lists correctly
    user_1_recs = pd_result[pd_result["user_id"] == "1"].iloc[0]["recommended_items"]
    assert isinstance(user_1_recs, np.ndarray) or isinstance(user_1_recs, list)
    assert len(user_1_recs) <= 2
