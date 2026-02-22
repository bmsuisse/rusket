# PySpark Integration

`rusket` provides seamless integration with massive datasets hosted on PySpark clusters.

While standard Pandas DataFrames might struggle with memory constraints when handling highly granular transaction logs (e.g., millions of POS receipts or user web sessions), PySpark enables distributed data preparation. Rather than running a custom JVM algorithm, `rusket` can perform distributed Market Basket Analysis by partitioning the data logically and running its native Rust extensions directly on the PySpark worker nodes.

## Distributing Workloads (Grouped Mining)

A common use case in enterprise environments is to run association rule mining *per store*, *per region*, or *per customer segment*, rather than globally across the entire business.

The `rusket.mine_grouped` function handles this natively by leveraging PySpark's `applyInArrow` (or `applyInPandas` for older Spark versions) capabilities. The dataset is grouped by a partition column, and the fast Rust-powered `rusket` engine is executed concurrently across your cluster.

### Fast Grouped Mining

Here is how you can process a multi-tenant or multi-store dataset:

```python
from pyspark.sql import SparkSession
import rusket.spark

spark = SparkSession.builder.getOrCreate()

# Ensure Arrow-based optimizations are enabled in Spark
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# Example: 10 Million row Spark DataFrame 
# columns: store_id, milk, bread, butter (One-Hot Encoded)
spark_df = spark.read.parquet("s3://data/one_hot_receipts.parquet")

# Mine frequent itemsets PER store, entirely distributed!
freq_df = rusket.spark.mine_grouped(
    df=spark_df,
    group_col="store_id",
    min_support=0.05,        # 5% minimum support per store
    method="auto",
    use_colnames=True
)

# freq_df is a new PySpark DataFrame containing:
# store_id (String) | support (Double) | itemsets (Array[String])
display(freq_df)
```

## Collecting Large Summaries via Arrow (Streaming)

Sometimes you want to analyze global patterns but the raw transactional event log is too large to `.toPandas()` directly to the driver memory.

`rusket.streaming.mine_spark` allows you to stream integer `(transaction_id, item_id)` chunks directly from the PySpark workers into the `rusket.FPMiner` single-node streaming accumulator running on the driver, leveraging zero-copy Apache Arrow transfers.

```python
from rusket.streaming import mine_spark

# Assuming an event log Spark DataFrame
# columns: session_id (Long), item_id (Int)
events_df = spark.read.parquet("s3://data/clickstream_events.parquet")

# Process the entire distributed DataFrame via a memory-safe stream
freq_df = mine_spark(
    spark_df=events_df,
    n_items=50_000,       # Total number of unique items
    txn_col="session_id", # The grouping ID
    item_col="item_id",   # The integer ID of the item
    min_support=0.01,     # 1% support globally
    max_len=3             # Max basket length
)

# freq_df is a standard Pandas DataFrame on the driver
print(freq_df.head())
```
