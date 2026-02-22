# PySpark Integration

`rusket` provides seamless integration with massive datasets hosted on PySpark clusters, leveraging zero-copy Apache Arrow transfers between Spark workers and the Rust core.

All distributed functions live in `rusket.spark` and use `applyInArrow` (Spark 3.4+) with `applyInPandas` as a fallback for older versions.

---

## Setup

```python
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
```

---

## `mine_grouped` — Distributed Market Basket Analysis

Group a PySpark DataFrame by a key column and run `rusket.mine` concurrently on each partition across the cluster. Input must be a One-Hot Encoded (wide) boolean matrix per partition.

```python
import rusket.spark

freq_df = rusket.spark.mine_grouped(
    df=spark_df,       # pyspark.sql.DataFrame (wide OHE format)
    group_col="store_id",
    min_support=0.05,
    method="auto",     # "auto" | "fpgrowth" | "eclat"
    use_colnames=True, # must be True for schema safety
    max_len=None,
)
# Output columns: store_id (string) | support (double) | itemsets (array<string>)
```

---

## `rules_grouped` — Distributed Association Rules

Takes the frequent itemsets output of `mine_grouped` and applies `association_rules` per group.

```python
rules_df = rusket.spark.rules_grouped(
    df=freq_spark_df,
    group_col="store_id",
    num_itemsets={"store_A": 10_000, "store_B": 5_000},  # or a single int
    metric="confidence",
    min_threshold=0.8,
)
# Output columns: store_id | antecedents | consequents | confidence | lift | ...
# (all 11 metrics are always included)
```

**Full end-to-end grouped pipeline:**

```python
import rusket.spark

freq_df  = rusket.spark.mine_grouped(spark_df, group_col="store_id", min_support=0.05)
rules_df = rusket.spark.rules_grouped(freq_df, group_col="store_id", num_itemsets=10_000)
```

---

## `prefixspan_grouped` — Distributed Sequential Pattern Mining

Applies PrefixSpan to a user event log grouped by a partition key (e.g. per store or per tenant).

```python
seq_df = rusket.spark.prefixspan_grouped(
    df=spark_df,
    group_col="store_id",
    user_col="user_id",      # sequence identifier within each group
    time_col="timestamp",    # ordering column
    item_col="item_id",
    min_support=10,          # absolute minimum (number of sequences)
    max_len=None,
)
# Output columns: store_id | support (long) | sequence (array<string>)
```

---

## `hupm_grouped` — Distributed High-Utility Pattern Mining

Applies the EFIM High-Utility Pattern Mining algorithm per partition.

```python
hupm_df = rusket.spark.hupm_grouped(
    df=spark_df,
    group_col="store_id",
    transaction_col="txn_id",
    item_col="item_id",
    utility_col="profit",
    min_utility=50.0,
    max_len=None,
)
# Output columns: store_id | utility (double) | itemset (array<long>)
```

---

## `recommend_batches` — Distributed Batch Recommendations

Distribute batch user recommendations across PySpark workers using a pre-trained `ALS` or `Recommender` model. The model is broadcast to each worker via `mapInArrow`.

```python
rec_df = rusket.spark.recommend_batches(
    df=user_history_spark_df,   # must contain user_col
    model=als,                  # ALS, BPR, or Recommender instance
    user_col="user_id",
    k=5,
)
# Output columns: user_id (string) | recommended_items (array<int>)
```

---

## `mine_spark` — Streaming Global Mining via FPMiner

When you need **global** patterns but the raw event log is too large to `.toPandas()` in one shot, stream integer `(txn_id, item_id)` chunks from Spark workers into the `FPMiner` streaming accumulator on the driver via zero-copy Arrow transfers.

```python
from rusket.streaming import mine_spark

freq_df = mine_spark(
    spark_df=events_df,     # pyspark.sql.DataFrame with txn_col + item_col
    n_items=50_000,         # total number of unique items
    txn_col="session_id",   # transaction grouping ID (integer-typed)
    item_col="item_id",     # item index column (integer-typed)
    min_support=0.01,
    max_len=3,
)
# Returns a standard Pandas DataFrame on the driver
print(freq_df.head())
```

---

## `to_spark`

Convert a Pandas or Polars DataFrame to a PySpark DataFrame.

```python
spark_df = rusket.spark.to_spark(spark_session, df)
```
