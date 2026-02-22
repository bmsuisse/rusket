# PySpark Integration

`rusket` integrates with PySpark clusters via zero-copy Apache Arrow transfers, enabling distributed execution of all its core algorithms across a Databricks or on-prem Hadoop cluster without manual serialisation.

All distributed functions live in `rusket.spark` and use `applyInArrow` (Spark 3.4+) with `applyInPandas` as a fallback for older versions.

---

## Setup

```python
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
```

---

## `mine_grouped` — Distributed Market Basket Analysis per Store / Region

A retail chain with 50 stores wants per-store "Frequently Bought Together" rules — different stores, different assortments, different shopping habits. `mine_grouped` distributes `rusket.mine` across a Spark cluster, running each store's rules in parallel on separate executor nodes.

```python
import rusket.spark

# Delta table: receipt_id | store_id | <one column per SKU (boolean)>
spark_df = spark.table("gold.baskets_ohe")

# Each store gets its own frequent itemsets — runs fully in parallel on the cluster
freq_df = rusket.spark.mine_grouped(
    df=spark_df,
    group_col="store_id",
    min_support=0.05,
    method="auto",      # "auto" | "fpgrowth" | "eclat"
    use_colnames=True,  # must be True for schema safety
    max_len=3,          # pairs and triples only — avoids combinatorial explosion
)
# Output: store_id (string) | support (double) | itemsets (array<string>)
```

---

## `rules_grouped` — Distributed Association Rules per Segment

Takes the output of `mine_grouped` and generates association rules per group. Ideal for feeding per-store rules into a recommendation API or a merchandising dashboard.

```python
rules_df = rusket.spark.rules_grouped(
    df=freq_df,
    group_col="store_id",
    num_itemsets={"store_A": 45_000, "store_B": 12_300},  # receipts per store
    metric="confidence",
    min_threshold=0.6,
)
# Output: store_id | antecedents | consequents | confidence | lift | ...
# (all 11 metrics are always included)
```

**Full end-to-end regional pipeline:**

```python
# Step 1 — mine per store
freq_df  = rusket.spark.mine_grouped(spark_df, group_col="store_id", min_support=0.05)

# Step 2 — generate "Frequently Bought Together" rules per store
rules_df = rusket.spark.rules_grouped(freq_df, group_col="store_id", num_itemsets=20_000)

# Step 3 — write to Delta for downstream serving
rules_df.write.mode("overwrite").saveAsTable("gold.per_store_rules")
```

---

## `prefixspan_grouped` — Distributed Customer Journey Analysis

A telco or media company wants to discover how customers navigate their product portfolio over time — which services do customers typically subscribe to before churning to a competitor?

PrefixSpan mines sequential patterns (ordered events) across users, grouped by region or segment.

```python
seq_df = rusket.spark.prefixspan_grouped(
    df=spark_df,               # event_log: customer_id | region | timestamp | product_id
    group_col="region",
    user_col="customer_id",    # sequence identifier
    time_col="event_ts",       # ordering column (timestamp)
    item_col="product_id",
    min_support=100,           # at least 100 customers must follow this path
    max_len=4,                 # journeys up to 4 steps long
)
# Output: region | support (long) | sequence (array<string>)
# Example:
# EMEA | 432 | [broadband, mobile, tv_bundle, cancel]
# APAC | 218 | [mobile, broadband, upgrade_premium]
```

---

## `hupm_grouped` — Distributed High-Profit Bundle Discovery

A specialty foods retailer wants to discover which product combinations generate the most revenue per region, even if they aren't bought very frequently — a classic use case for High-Utility Pattern Mining.

```python
hupm_df = rusket.spark.hupm_grouped(
    df=spark_df,               # receipt_id | region | product_id | margin
    group_col="region",
    transaction_col="receipt_id",
    item_col="product_id",
    utility_col="margin",      # gross margin per line item
    min_utility=500.0,         # only bundles generating ≥ €500 total margin
    max_len=3,
)
# Output: region | utility (double) | itemset (array<long>)
# Example:
# NORTH | 1840.0 | [aged_cheese, wine_flight, charcuterie]
```

---

## `recommend_batches` — Overnight Batch Personalisation at Scale

Score millions of users overnight with a pre-trained ALS model, then write personalised recommendations to your CRM or marketing automation platform.

```python
# 1. Train ALS on driver once (or load from a checkpoint)
from rusket import ALS
als = ALS(factors=64, iterations=15).fit(user_item_csr)

# 2. Broadcast to every Spark worker and score all users in parallel
rec_df = rusket.spark.recommend_batches(
    df=spark.table("silver.user_sessions"),  # must contain user_col
    model=als,
    user_col="customer_id",
    k=10,   # top-10 personalised SKU recommendations per customer
)
# Output: customer_id (string) | recommended_items (array<int>)

# 3. Write to CRM / email platform
rec_df.write.mode("overwrite").saveAsTable("gold.daily_recommendations")
```

---

## `mine_spark` — Global Mining via FPMiner Streaming

When you need **global** patterns (not per-group) but the raw event log is too large for `.toPandas()`, stream integer `(txn_id, item_id)` chunks from Spark workers into the `FPMiner` accumulator on the driver via zero-copy Arrow transfers.

Use case: a marketplace with 200M+ order lines needs one global set of "Customers also buy" rules.

```python
from rusket.streaming import mine_spark

freq_df = mine_spark(
    spark_df=spark.table("silver.order_lines"),
    n_items=200_000,          # distinct SKU count
    txn_col="order_id",       # integer-typed order identifier
    item_col="sku_index",     # 0-based integer SKU index
    min_support=0.001,        # 0.1% of all orders
    max_len=3,
)
# Returns a standard Pandas DataFrame on the driver
# ready for association_rules() or to be written to Delta
print(freq_df.head())
```

---

## `to_spark`

Convert a Pandas or Polars DataFrame to a PySpark DataFrame:

```python
spark_df = rusket.spark.to_spark(spark_session, df)
```
