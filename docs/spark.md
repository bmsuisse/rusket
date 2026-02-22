# PySpark Integration

Distributed execution of rusket algorithms across Databricks or on-prem Hadoop clusters via zero-copy Apache Arrow transfers.

`rusket` integrates with PySpark clusters via zero-copy Apache Arrow transfers, enabling distributed execution of all its core algorithms without manual serialisation.

All distributed functions live in `rusket.spark` and use `applyInArrow` (Spark 3.4+) with `applyInPandas` as a fallback for older versions.

---

## Setup

```python
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
```

---

## `mine_grouped` — Distributed Market Basket Analysis per Store / Region

```python
import rusket.spark

spark_df = spark.table("gold.baskets_ohe")

freq_df = rusket.spark.mine_grouped(
    df=spark_df,
    group_col="store_id",
    min_support=0.05,
    method="auto",
    use_colnames=True,
    max_len=3,
)
# Output: store_id (string) | support (double) | itemsets (array<string>)
```

---

## `rules_grouped` — Distributed Association Rules per Segment

```python
rules_df = rusket.spark.rules_grouped(
    df=freq_df,
    group_col="store_id",
    num_itemsets={"store_A": 45_000, "store_B": 12_300},
    metric="confidence",
    min_threshold=0.6,
)
# Output: store_id | antecedents | consequents | confidence | lift | ...
```

**Full end-to-end regional pipeline:**

```python
freq_df  = rusket.spark.mine_grouped(spark_df, group_col="store_id", min_support=0.05)
rules_df = rusket.spark.rules_grouped(freq_df, group_col="store_id", num_itemsets=20_000)
rules_df.write.mode("overwrite").saveAsTable("gold.per_store_rules")
```

---

## `prefixspan_grouped` — Distributed Customer Journey Analysis

```python
seq_df = rusket.spark.prefixspan_grouped(
    df=spark_df,
    group_col="region",
    user_col="customer_id",
    time_col="event_ts",
    item_col="product_id",
    min_support=100,
    max_len=4,
)
# Output: region | support (long) | sequence (array<string>)
# Example:
# EMEA | 432 | [broadband, mobile, tv_bundle, cancel]
```

---

## `hupm_grouped` — Distributed High-Profit Bundle Discovery

```python
hupm_df = rusket.spark.hupm_grouped(
    df=spark_df,
    group_col="region",
    transaction_col="receipt_id",
    item_col="product_id",
    utility_col="margin",
    min_utility=500.0,
    max_len=3,
)
# Output: region | utility (double) | itemset (array<long>)
```

---

## `recommend_batches` — Overnight Batch Personalisation at Scale

```python
from rusket import ALS

als = ALS(factors=64, iterations=15).fit(user_item_csr)

rec_df = rusket.spark.recommend_batches(
    df=spark.table("silver.user_sessions"),
    model=als,
    user_col="customer_id",
    k=10,
)
# Output: customer_id (string) | recommended_items (array<int>)

rec_df.write.mode("overwrite").saveAsTable("gold.daily_recommendations")
```

---

## `mine_spark` — Global Mining via FPMiner Streaming

```python
from rusket.streaming import mine_spark

freq_df = mine_spark(
    spark_df=spark.table("silver.order_lines"),
    n_items=200_000,
    txn_col="order_id",
    item_col="sku_index",
    min_support=0.001,
    max_len=3,
)
```

---

## `to_spark`

Convert a Pandas or Polars DataFrame to a PySpark DataFrame:

```python
spark_df = rusket.spark.to_spark(spark_session, df)
```
