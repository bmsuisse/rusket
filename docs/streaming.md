# Streaming & Big Data

Handle billions of rows of transaction data without running out of memory using rusket's `FPMiner` streaming accumulator.

When dealing with extremely large transactional datasets — such as billions of rows of e-commerce clickstreams or year-long retail logs — loading the entire one-hot encoded matrix into RAM (even as a sparse matrix) may exhaust your system's memory.

To solve this, `rusket` includes `FPMiner`, a highly optimized streaming accumulator written completely in Rust.

## The Streaming Concept

Instead of converting a "long-format" event log `(user_id, item_id)` into a massive `N × M` sparse matrix, the `FPMiner` accepts small, memory-safe chunks of raw integers.

Rust accumulates these `(transaction_id, item_id)` pairs internally using a highly efficient `HashMap<i64, Vec<i32>>`. Because this happens incrementally:

1. **Python memory** overhead is strictly limited to the size of a single chunk (e.g., 10 million rows).
2. **Matrix pivoting** (Group-By operations) is avoided entirely.

```mermaid
graph LR
    A["Parquet File<br/>(1 Billion Rows)"] --> B["Python Chunk<br/>(10M Rows)"]
    B -->|Stream| C["Rust FPMiner<br/>(Accumulates inside Heap)"]
    C -->|Next Chunk| A
    C -->|mine()| D["Frequent Itemsets<br/>pd.DataFrame"]
```

## Reading from Disk (Parquet / CSV)

```python
import pandas as pd
from rusket import FPMiner

miner = FPMiner(n_items=100_000)

for chunk in pd.read_parquet("massive_event_log.parquet", chunksize=10_000_000):
    txn_ids = chunk["user_session"].to_numpy(dtype="int64")
    item_ids = chunk["product_id"].to_numpy(dtype="int32")
    miner.add_chunk(txn_ids, item_ids)

freq_itemsets = miner.mine(
    min_support=0.005,
    max_len=4,
    method="auto"
)
```

## Arrow and DuckDB Integrations

For even higher performance, bypass Pandas entirely using `pyarrow` underneath a DuckDB query engine:

```python
import duckdb
from rusket.streaming import mine_duckdb

con = duckdb.connect("my_analytics_db.duckdb")

freq = mine_duckdb(
    con=con,
    query="SELECT session_id, product_id FROM sales WHERE region = 'EMEA'",
    n_items=50_000,
    txn_col="session_id",
    item_col="product_id",
    min_support=0.01,
    chunk_size=5_000_000
)
```
