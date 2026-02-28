# API Reference

> This file is **auto-generated** by `scripts/gen_api_reference.py`.  Do not edit by hand — update the Python docstrings instead.

## Functional API

Convenience module-level functions.  For most use-cases these are the only entry points you need.

### `mine`

Mine frequent itemsets using the optimal algorithm.

This module-level function relies on the Object-Oriented APIs.
Automatically selects between FP-Growth and Eclat based on density,
or falls back to FPMiner (streaming) if memory is low.

```python
from rusket.mine import mine

mine(df: 'pd.DataFrame | pl.DataFrame | np.ndarray | Any', min_support: 'float' = 0.5, null_values: 'bool' = False, use_colnames: 'bool' = True, max_len: 'int | None' = None, method: 'str' = 'auto', verbose: 'int' = 0, column_names: 'list[str] | None' = None) -> 'pd.DataFrame'
```

---

### `fpgrowth`

Find frequent itemsets using the optimal algorithm (Eclat or FP-growth).

This module-level function relies on the Object-Oriented APIs.

```python
from rusket.fpgrowth import fpgrowth

fpgrowth(df: 'pd.DataFrame | pl.DataFrame | np.ndarray | Any', min_support: 'float' = 0.5, null_values: 'bool' = False, use_colnames: 'bool' = True, max_len: 'int | None' = None, method: 'str' = 'auto', verbose: 'int' = 0, column_names: 'list[str] | None' = None) -> 'pd.DataFrame'
```

---

### `eclat`

Find frequent itemsets using the Eclat algorithm.

This module-level function relies on the Object-Oriented APIs.

```python
from rusket.eclat import eclat

eclat(df: 'pd.DataFrame | pl.DataFrame | np.ndarray | Any', min_support: 'float' = 0.5, null_values: 'bool' = False, use_colnames: 'bool' = True, max_len: 'int | None' = None, verbose: 'int' = 0, column_names: 'list[str] | None' = None) -> 'pd.DataFrame'
```

---

### `association_rules`

```python
from rusket.association_rules import association_rules

association_rules(df: 'pd.DataFrame | Any', num_itemsets: 'int | None' = None, df_orig: 'pd.DataFrame | None' = None, null_values: 'bool' = False, metric: 'str' = 'confidence', min_threshold: 'float' = 0.8, support_only: 'bool' = False, return_metrics: 'list[str]' = ['antecedent support', 'consequent support', 'support', 'confidence', 'lift', 'representativity', 'leverage', 'conviction', 'zhangs_metric', 'jaccard', 'certainty', 'kulczynski']) -> 'pd.DataFrame'
```

---

### `prefixspan`

Mine sequential patterns using the PrefixSpan algorithm.

This function discovers frequent sequences of items across multiple users/sessions.
Currently, this assumes sequences where each event consists of a single item
(e.g., a sequence of page views or a sequence of individual products bought over time).

```python
from rusket.prefixspan import prefixspan

prefixspan(sequences: 'list[list[int]]', min_support: 'int | float', max_len: 'int | None' = None) -> 'pd.DataFrame'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| sequences | list of list of int | A list of sequences, where each sequence is a list of integers representing items. Example: `[[1, 2, 3], [1, 3], [2, 3]]`. |
| min_support | int \| float | The minimum absolute support (number of sequences a pattern must appear in), or float percent. |
| max_len | int, optional | The maximum length of the sequential patterns to mine. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| pd.DataFrame |  | A DataFrame containing 'support' and 'sequence' columns. |

---

### `hupm`

Mine high-utility itemsets.

This function discovers combinations of items that generate a high total utility
(e.g., profit) across all transactions, even if they aren't the most frequent.

```python
from rusket.hupm import hupm

hupm(transactions: 'list[list[int]]', utilities: 'list[list[float]]', min_utility: 'float', max_len: 'int | None' = None) -> 'pd.DataFrame'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| transactions | list of list of int | A list of transactions, where each transaction is a list of item IDs. |
| utilities | list of list of float | A list of identical structure to `transactions`, but containing the numeric utility (e.g., profit) of that item in that specific transaction. |
| min_utility | float | The minimum total utility required to consider a pattern "high-utility". |
| max_len | int, optional | The maximum length of the itemsets to mine. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| pd.DataFrame |  | A DataFrame containing 'utility' and 'itemset' columns. |

---

### `sequences_from_event_log`

Convert an event log DataFrame into the sequence format required by PrefixSpan.

Accepts Pandas, Polars, or PySpark DataFrames. Data is grouped by `user_col`,
ordered by `time_col`, and `item_col` values are collected into sequences.

```python
from rusket.prefixspan import sequences_from_event_log

sequences_from_event_log(df: 'Any', user_col: 'str', time_col: 'str', item_col: 'str') -> 'tuple[list[list[int]], dict[int, Any]]'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| df | pd.DataFrame \| pl.DataFrame \| pyspark.sql.DataFrame | Event log containing users, timestamps, and items. |
| user_col | str | Column name identifying the sequence (e.g., user_id or session_id). |
| time_col | str | Column name for ordering events. |
| item_col | str | Column name for the items. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| tuple of (indptr, indices, item_mapping) |  | - indptr: CSR-style index pointer list. - indices: Flattened item index list. - item_mapping: A dictionary mapping the integer IDs back to the original item labels. |

---

### `mine_hupm`

Mine high-utility itemsets from a long-format DataFrame.

Converts a Pandas or Polars DataFrame into the required list-of-lists format
and runs the High-Utility Pattern Mining (HUPM) algorithm.

```python
from rusket.hupm import mine_hupm

mine_hupm(data: 'Any', transaction_col: 'str', item_col: 'str', utility_col: 'str', min_utility: 'float', max_len: 'int | None' = None) -> 'pd.DataFrame'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| data | pd.DataFrame or pl.DataFrame | A long-format DataFrame where each row represents an item in a transaction. |
| transaction_col | str | Column name identifying the transaction ID. |
| item_col | str | Column name identifying the item ID (must be numeric integers). |
| utility_col | str | Column name identifying the numeric utility (e.g. price, profit) of the item. |
| min_utility | float | The minimum total utility required to consider a pattern "high-utility". |
| max_len | int, optional | Maximum length of the itemsets to mine. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| pd.DataFrame |  | A DataFrame containing 'utility' and 'itemset' columns. |

---

### `mine_duckdb`

Stream directly from a DuckDB query via Arrow RecordBatches.

This is extremely memory efficient, bypassing Pandas entirely.

```python
from rusket.streaming import mine_duckdb

mine_duckdb(con: 'Any', query: 'str', n_items: 'int', txn_col: 'str', item_col: 'str', min_support: 'float' = 0.5, max_len: 'int | None' = None, chunk_size: 'int' = 1000000) -> 'pd.DataFrame'
```

---

### `mine_spark`

Stream natively from a PySpark DataFrame on Databricks via Arrow.

Uses `toLocalIterator()` to fetch Arrow chunks incrementally directly
to the driver node, avoiding massive memory spikes.

```python
from rusket.streaming import mine_spark

mine_spark(spark_df: 'Any', n_items: 'int', txn_col: 'str', item_col: 'str', min_support: 'float' = 0.5, max_len: 'int | None' = None) -> 'pd.DataFrame'
```

---

### `from_transactions`

Convert long-format transactional data to a one-hot boolean matrix.

The return type mirrors the input type:

- **Polars** ``DataFrame`` → **Polars** ``DataFrame``
- **Pandas** ``DataFrame`` → **Pandas** ``DataFrame``
- **Spark** ``DataFrame``  → **Spark**  ``DataFrame``
- ``list[list[...]]``      → **Pandas** ``DataFrame``

```python
from rusket.transactions import from_transactions

from_transactions(data: 'DataFrame | Sequence[Sequence[str | int]] | Any', transaction_col: 'str | None' = None, item_col: 'str | None' = None, min_item_count: 'int' = 1, verbose: 'int' = 0) -> 'Any'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| data |  | One of:  - **Pandas / Polars / Spark DataFrame** with (at least) two columns: one for the transaction identifier and one for the item. - **List of lists** where each inner list contains the items of a single transaction, e.g. ``[["bread", "milk"], ["bread", "eggs"]]``. |
| transaction_col |  | Name of the column that identifies transactions.  If ``None`` the first column is used.  Ignored for list-of-lists input. |
| item_col |  | Name of the column that contains item values.  If ``None`` the second column is used.  Ignored for list-of-lists input. |
| min_item_count |  | Minimum number of times an item must appear to be included in the resulting one-hot-encoded matrix. Default is 1. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| DataFrame |  | A boolean DataFrame (same type as input) ready for :func:`rusket.fpgrowth` or :func:`rusket.eclat`. Column names correspond to the unique items. |

**Examples**

```python
>>> import rusket
>>> import pandas as pd
>>> df = pd.DataFrame({
...     "order_id": [1, 1, 1, 2, 2, 3],
...     "item": [3, 4, 5, 3, 5, 8],
... })
>>> ohe = rusket.from_transactions(df)
>>> freq = rusket.fpgrowth(ohe, min_support=0.5, use_colnames=True)
```

---

### `from_transactions_csr`

Convert long-format transactional data to a CSR matrix + column names.

Unlike :func:`from_transactions`, this returns a raw
``scipy.sparse.csr_matrix`` that can be passed directly to
:func:`rusket.fpgrowth` or :func:`rusket.eclat` — **no pandas overhead**.

For billion-row datasets, this processes data in chunks of ``chunk_size``
rows, keeping peak memory to one chunk + the running CSR.

```python
from rusket.transactions import from_transactions_csr

from_transactions_csr(data: 'DataFrame | str | Any', transaction_col: 'str | None' = None, item_col: 'str | None' = None, chunk_size: 'int' = 10000000) -> 'tuple[Any, list[str]]'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| data |  | One of:  - **Pandas DataFrame** with (at least) two columns. - **Polars DataFrame** or **Spark DataFrame** (converted internally). - **File path** (str / Path) to a Parquet file — read in chunks. |
| transaction_col |  | Name of the transaction-id column. Defaults to the first column. |
| item_col |  | Name of the item column. Defaults to the second column. |
| chunk_size |  | Number of rows per chunk. Lower values use less memory. Default: 10 million rows. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| tuple[scipy.sparse.csr_matrix, list[str]] |  | A CSR matrix and the list of column (item) names.  Pass directly::  csr, names = from_transactions_csr(df) freq = fpgrowth(csr, min_support=0.001, use_colnames=True, column_names=names) |

**Examples**

```python
>>> import rusket
>>> csr, names = rusket.from_transactions_csr("orders.parquet")
>>> freq = rusket.fpgrowth(csr, min_support=0.001,
...                        use_colnames=True, column_names=names)
```

---

### `from_pandas`

Shorthand for ``from_transactions(df, transaction_col, item_col)``.

```python
from rusket.transactions import from_pandas

from_pandas(df: 'pd.DataFrame', transaction_col: 'str | None' = None, item_col: 'str | None' = None, min_item_count: 'int' = 1, verbose: 'int' = 0) -> 'pd.DataFrame'
```

---

### `from_polars`

Shorthand for ``from_transactions(df, transaction_col, item_col)``.

```python
from rusket.transactions import from_polars

from_polars(df: 'pl.DataFrame', transaction_col: 'str | None' = None, item_col: 'str | None' = None, min_item_count: 'int' = 1, verbose: 'int' = 0) -> 'pl.DataFrame'
```

---

### `from_spark`

Shorthand for ``from_transactions(df, transaction_col, item_col)``.

```python
from rusket.transactions import from_spark

from_spark(df: 'SparkDataFrame', transaction_col: 'str | None' = None, item_col: 'str | None' = None, min_item_count: 'int' = 1, verbose: 'int' = 0) -> 'SparkDataFrame'
```

---

### `from_arrow`

Convert a PyArrow Table in long format to a one-hot boolean PyArrow Table.

This is a zero-copy-friendly shorthand for ``from_transactions(table, ...)``.  The
input table must have at least two columns: one for the transaction identifier and
one for the item.  The returned table has boolean columns (one per unique item).

```python
from rusket.transactions import from_arrow

from_arrow(table: 'pa.Table', transaction_col: 'str | None' = None, item_col: 'str | None' = None, min_item_count: 'int' = 1, verbose: 'int' = 0) -> 'pa.Table'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| table |  | A ``pyarrow.Table`` with at least two columns (transaction id + item). |
| transaction_col |  | Name of the transaction-id column. Defaults to the first column. |
| item_col |  | Name of the item column. Defaults to the second column. |
| min_item_count |  | Minimum occurrences for an item to be included. Default is 1. |
| verbose |  | Verbosity level. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| pyarrow.Table |  | A boolean Table ready for :func:`rusket.fpgrowth` / :func:`rusket.eclat`. |

---

### `evaluate`

Evaluate a trained recommendation model on a test set.

Compute metrics like NDCG@k, Hit Rate@k, Precision@k, and Recall@k using
fast natively-backed Rust evaluation loops.

When a model has ``_user_labels`` / ``_item_labels`` (set by
``from_transactions()``), the test IDs are automatically mapped to
internal 0-based indices so that ``recommend_items()`` receives valid
indices and the recommended item indices can be compared with the
ground truth.

```python
from rusket.evaluation import evaluate

evaluate(model: 'Any', test_interactions: 'Any', k: 'int' = 10, metrics: 'list[MetricName] | None' = None) -> 'dict[str, float]'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| model | Any | A trained recommendation model supporting ``recommend_items(user_id, k, exclude_seen)``. |
| test_interactions | np.ndarray or pd.DataFrame | Ground truth test interactions. Must either have columns "user" and "item", or be a 2D array format. |
| k | int, default=10 | The cutoff rank for evaluation. |
| metrics | list of str, optional | Metrics to compute. Default: ["ndcg", "hr", "precision", "recall"]. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| dict[str, float] |  | Dictionary of averaged metric values. |

---

### `train_test_split`

Split interactions into random train and test sets.

```python
from rusket.model_selection import train_test_split

train_test_split(df, user_col: 'str', item_col: 'str', test_size: 'float' = 0.2, random_state: 'int | None' = None)
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| df | pd.DataFrame | The interaction dataframe. |
| user_col | str | Name of the user column. |
| item_col | str | Name of the item column. |
| test_size | float, default=0.2 | Percentage of data to put in the test set. |
| random_state | int, optional | Set random seed (currently not used by Rust backend, but reserved for future). |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| tuple[pd.DataFrame, pd.DataFrame] |  | train_df, test_df |

---

### `leave_one_out_split`

Leave exactly one interaction per user for the test set.

If a timestamp column is provided, the latest interaction is left out.
If no timestamp is provided, a random interaction is chosen.

```python
from rusket.model_selection import leave_one_out_split

leave_one_out_split(df, user_col: 'str', item_col: 'str', timestamp_col: 'str | None' = None)
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| df | pd.DataFrame | The interaction dataframe. |
| user_col | str | Name of the user column (must be numeric encoded to i32 ideally, or pandas int). |
| item_col | str | Name of the item column. |
| timestamp_col | str, optional | Name of the timestamp or ordering column. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| tuple[pd.DataFrame, pd.DataFrame] |  | train_df, test_df |

---

### `pca`

Project data into `n_components` dimensions using PCA.

```python
from rusket.pca import pca

pca(x: 'npt.NDArray[Any]', n_components: 'int' = 2, svd_solver: 'str' = 'auto') -> 'ProjectedSpace'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| x | array-like of shape (n_samples, n_features) |  |
| n_components | int, default=2 |  |
| svd_solver | {"auto", "exact", "randomized"}, default="auto" |  |

---

### `pca2`

Project data into exactly 2 dimensions using PCA.

```python
from rusket.pca import pca2

pca2(x: 'npt.NDArray[Any]', svd_solver: 'str' = 'auto') -> 'ProjectedSpace'
```

---

### `pca3`

Project data into exactly 3 dimensions using PCA.

```python
from rusket.pca import pca3

pca3(x: 'npt.NDArray[Any]', svd_solver: 'str' = 'auto') -> 'ProjectedSpace'
```

---

## OOP Mining API

All mining classes share a common `Miner.from_transactions()` / `.mine()` interface. `FPGrowth`, `Eclat`, `AutoMiner`, and `HUPM` also inherit `RuleMinerMixin` which adds `.association_rules()` and `.recommend_items()` helpers.

### `FPGrowth`

FP-Growth frequent itemset miner.

This class wraps the fast, core Rust FP-Growth implementation.

```python
from rusket.fpgrowth import FPGrowth

FPGrowth(data: 'pd.DataFrame | pl.DataFrame | np.ndarray | Any', item_names: 'list[str] | None' = None, min_support: 'float' = 0.5, null_values: 'bool' = False, use_colnames: 'bool' = True, max_len: 'int | None' = None, verbose: 'int' = 0, **kwargs: 'Any')
```

#### `FPGrowth.mine`

Execute the FP-growth algorithm on the stored data.

```python
from rusket.fpgrowth import FPGrowth.mine

FPGrowth.mine(**kwargs: 'Any') -> 'pd.DataFrame'
```

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| pandas.DataFrame |  | DataFrame with two columns: - `support`: the support score. - `itemsets`: list of items (indices or column names). |

---

---

### `Eclat`

Eclat frequent itemset miner.

Eclat is typically faster than FP-growth on dense datasets due to
efficient vertical bitset intersection logic.

```python
from rusket.eclat import Eclat

Eclat(data: 'pd.DataFrame | pl.DataFrame | np.ndarray | Any', item_names: 'list[str] | None' = None, min_support: 'float' = 0.5, null_values: 'bool' = False, use_colnames: 'bool' = True, max_len: 'int | None' = None, verbose: 'int' = 0, **kwargs: 'Any')
```

#### `Eclat.mine`

Execute the Eclat algorithm on the stored data.

```python
from rusket.eclat import Eclat.mine

Eclat.mine(**kwargs: 'Any') -> 'pd.DataFrame'
```

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| pandas.DataFrame |  | DataFrame with two columns: - `support`: the support score. - `itemsets`: list of items (indices or column names). |

---

---

### `AutoMiner`

Automatic frequent itemset miner.

Selects the optimal miner (FP-Growth or Eclat) based on matrix density.
Automatically falls back to streaming (FPMiner) if the dataset exceeds
available memory.

```python
from rusket.mine import AutoMiner

AutoMiner(data: 'pd.DataFrame | pl.DataFrame | np.ndarray | Any', item_names: 'list[str] | None' = None, min_support: 'float' = 0.5, null_values: 'bool' = False, use_colnames: 'bool' = True, max_len: 'int | None' = None, verbose: 'int' = 0, **kwargs: 'Any')
```

#### `AutoMiner.mine`

Execute the optimal algorithm on the stored data.

```python
from rusket.mine import AutoMiner.mine

AutoMiner.mine(**kwargs: 'Any') -> 'pd.DataFrame'
```

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| pandas.DataFrame |  | DataFrame with two columns: - `support`: the support score. - `itemsets`: list of items (indices or column names). |

---

---

### `PrefixSpan`

Sequential Pattern Mining (PrefixSpan) model.

This class discovers frequent sequences of items across multiple users/sessions.

```python
from rusket.prefixspan import PrefixSpan

PrefixSpan(data: 'list[list[int]]', min_support: 'int | float', max_len: 'int | None' = None, item_mapping: 'dict[int, Any] | None' = None)
```

#### `PrefixSpan.mine`

Mine sequential patterns using PrefixSpan.

```python
from rusket.prefixspan import PrefixSpan.mine

PrefixSpan.mine(**kwargs: 'Any') -> 'pd.DataFrame'
```

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| pd.DataFrame |  | A DataFrame containing 'support' and 'sequence' columns. Sequences are mapped back to original item names if `from_transactions` was used. |

---

---

### `HUPM`

High-Utility Pattern Mining (HUPM) model.

This class discovers combinations of items that generate a high total utility
(e.g., profit) across all transactions, even if they aren't the most frequent.

```python
from rusket.hupm import HUPM

HUPM(transactions: 'list[list[int]]', utilities: 'list[list[float]]', min_utility: 'float', max_len: 'int | None' = None)
```

#### `HUPM.mine`

Mine high-utility itemsets.

```python
from rusket.hupm import HUPM.mine

HUPM.mine(**kwargs: 'Any') -> 'pd.DataFrame'
```

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| pd.DataFrame |  | A DataFrame containing 'utility' and 'itemset' columns. |

---

---

### `FPMiner`

Streaming FP-Growth / Eclat accumulator for billion-row datasets.

Feeds (transaction_id, item_id) integer arrays to Rust one chunk at a
time.  Rust accumulates per-transaction item lists in a
``HashMap<i64, Vec<i32>>``.  Peak **Python** memory = one chunk.

```python
from rusket.streaming import FPMiner

FPMiner(n_items: 'int', max_ram_mb: 'int | None' = -1, hint_n_transactions: 'int | None' = None) -> 'None'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| n_items | int | Number of distinct items (column count).  All item IDs fed via :meth:`add_chunk` must be in ``[0, n_items)``. |

**Examples**

```python
Process a Parquet file 10 M rows at a time:

>>> import pandas as pd
>>> import numpy as np
>>> from rusket import FPMiner
>>> miner = FPMiner(n_items=500_000)
>>> for chunk in pd.read_parquet("orders.parquet", chunksize=10_000_000):
...     txn = chunk["txn_id"].to_numpy(dtype="int64")
...     item = chunk["item_idx"].to_numpy(dtype="int32")
...     miner.add_chunk(txn, item)
>>> freq = miner.mine(min_support=0.001, max_len=3, use_colnames=True)
```

#### `FPMiner.add_arrow_batch`

Feed a PyArrow RecordBatch directly into the miner.
Zero-copy extraction is used if types match (Int64/Int32).

```python
from rusket.streaming import FPMiner.add_arrow_batch

FPMiner.add_arrow_batch(batch: 'Any', txn_col: 'str', item_col: 'str') -> 'FPMiner'
```

---

#### `FPMiner.add_chunk`

Feed a chunk of (transaction_id, item_id) pairs.

```python
from rusket.streaming import FPMiner.add_chunk

FPMiner.add_chunk(txn_ids: 'np.ndarray', item_ids: 'np.ndarray') -> 'FPMiner'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| txn_ids | np.ndarray[int64] | 1-D array of transaction identifiers (arbitrary 64-bit integers). |
| item_ids | np.ndarray[int32] | 1-D array of item column indices (0-based). |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| self  (for chaining) |  |  |

---

#### `FPMiner.fit`

Sklearn-compatible alias for ``mine()``. Runs the mining algorithm.

```python
from rusket.streaming import FPMiner.fit

FPMiner.fit(**kwargs: 'Any') -> 'FPMiner'
```

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| self |  |  |

---

#### `FPMiner.mine`

Mine frequent itemsets from all accumulated transactions.

```python
from rusket.streaming import FPMiner.mine

FPMiner.mine(min_support: 'float' = 0.5, max_len: 'int | None' = None, use_colnames: 'bool' = True, column_names: 'list[str] | None' = None, method: "typing.Literal['fpgrowth', 'eclat', 'auto']" = 'auto', verbose: 'int' = 0) -> 'pd.DataFrame'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| min_support | float | Minimum support threshold in ``(0, 1]``. |
| max_len | int \| None | Maximum itemset length. |
| use_colnames | bool | If ``True``, itemsets contain column names instead of indices. |
| column_names | list[str] \| None | Column names to use when ``use_colnames=True``. |
| method | "fpgrowth" \| "eclat" \| "auto" | Mining algorithm to use.  ``"auto"`` (default) picks the best algorithm automatically based on data density after pre-filtering rare items (Borgelt 2003 heuristic: density < 15% → Eclat, else FPGrowth). |
| verbose | int | Level of verbosity: >0 prints progress logs and times. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| pd.DataFrame |  | Columns ``support`` and ``itemsets``. |

---

#### `FPMiner.predict`

Return the last mined result, or run ``fit()`` first.

```python
from rusket.streaming import FPMiner.predict

FPMiner.predict(**kwargs: 'Any') -> 'pd.DataFrame'
```

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| pd.DataFrame |  | The frequent itemsets. |

---

#### `FPMiner.reset`

Free all accumulated data.

```python
from rusket.streaming import FPMiner.reset

FPMiner.reset() -> 'None'
```

---

---

### `FIN`

FIN (Fast Itemset per Nodeset) frequent itemset miner.

This class wraps the fast core Rust FIN implementation.

```python
from rusket.fin import FIN

FIN(data: 'pd.DataFrame | pl.DataFrame | np.ndarray | Any', item_names: 'list[str] | None' = None, min_support: 'float' = 0.5, null_values: 'bool' = False, use_colnames: 'bool' = True, max_len: 'int | None' = None, verbose: 'int' = 0, **kwargs: 'Any')
```

#### `FIN.mine`

Execute the FIN algorithm on the stored data.

```python
from rusket.fin import FIN.mine

FIN.mine(**kwargs: 'Any') -> 'pd.DataFrame'
```

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| pandas.DataFrame |  | DataFrame with two columns: - `support`: the support score. - `itemsets`: list of items (indices or column names). |

---

---

### `LCM`

LCM (Linear Closed Itemset Miner) frequent itemset miner.

This class wraps the fast core Rust LCM implementation using Prefix-Preserving Closure Extension.
It produces only *closed* frequent itemsets, offering massive memory savings
and faster execution out-of-the-box compared to classic algorithms on dense datasets.

```python
from rusket.lcm import LCM

LCM(data: 'pd.DataFrame | pl.DataFrame | np.ndarray | Any', item_names: 'list[str] | None' = None, min_support: 'float' = 0.5, null_values: 'bool' = False, use_colnames: 'bool' = True, max_len: 'int | None' = None, verbose: 'int' = 0, **kwargs: 'Any')
```

#### `LCM.mine`

Execute the LCM algorithm on the stored data to find closed itemsets.

```python
from rusket.lcm import LCM.mine

LCM.mine(**kwargs: 'Any') -> 'pd.DataFrame'
```

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| pandas.DataFrame |  | DataFrame with two columns: - `support`: the support score. - `itemsets`: list of items (indices or column names). |

---

---

## `RuleMinerMixin` — Shared Miner Interface

`FPGrowth`, `Eclat`, `AutoMiner`, and `HUPM` all inherit these methods from `RuleMinerMixin`.  You do not construct `RuleMinerMixin` directly.

### `RuleMinerMixin.association_rules`

Generate association rules from the mined frequent itemsets.

```python
from rusket.model import RuleMinerMixin.association_rules

RuleMinerMixin.association_rules(metric: 'str' = 'confidence', min_threshold: 'float' = 0.8, return_metrics: 'list[str] | None' = None) -> 'pd.DataFrame'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| metric | str, default='confidence' | The metric to evaluate if a rule is of interest. |
| min_threshold | float, default=0.8 | The minimum threshold for the evaluation metric. |
| return_metrics | list[str] \| None, default=None | List of metrics to include in the resulting DataFrame. Defaults to all available metrics. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| pd.DataFrame |  | DataFrame of strong association rules. |

---

### `RuleMinerMixin.recommend_items`

Deprecated: use :meth:`recommend_for_cart` instead.

```python
from rusket.model import RuleMinerMixin.recommend_items

RuleMinerMixin.recommend_items(items: 'list[Any]', n: 'int' = 5) -> 'list[Any]'
```

---

### `RuleMinerMixin._invalidate_rules_cache`

Clear the cached association rules (call after re-mining).

```python
from rusket.model import RuleMinerMixin._invalidate_rules_cache

RuleMinerMixin._invalidate_rules_cache() -> 'None'
```

---

## Recommenders

### `ALS`

Implicit ALS collaborative filtering model.

```python
from rusket.als import ALS

ALS(factors: 'int' = 64, regularization: 'float' = 0.01, alpha: 'float' = 40.0, iterations: 'int' = 15, seed: 'int' = 42, verbose: 'int' = 0, cg_iters: 'int' = 10, use_cholesky: 'bool' = False, anderson_m: 'int' = 0, **kwargs: 'Any') -> 'None'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| factors | int | Number of latent factors. |
| regularization | float | L2 regularisation weight. |
| alpha | float | Confidence scaling: ``confidence = 1 + alpha * r``. |
| iterations | int | Number of ALS outer iterations. |
| seed | int | Random seed. |
| cg_iters | int | Conjugate Gradient iterations per user/item solve (ignored when ``use_cholesky=True``).  Reduce to 3 for very large datasets. |
| use_cholesky | bool | Use a direct Cholesky solve instead of iterative CG. Exact solution; faster when users have many interactions relative to ``factors``. |
| anderson_m | int | History window for **Anderson Acceleration** of the outer ALS loop (default 0 = disabled).  Recommended value: **5**.  ALS is a fixed-point iteration ``(U,V) → F(U,V)``.  Anderson mixing extrapolates over the last ``m`` residuals to reach the fixed point faster, typically reducing the number of outer iterations by 30–50 % at identical recommendation quality::  # Baseline: 15 iterations model = ALS(iterations=15, cg_iters=3)  # Anderson-accelerated: 10 iterations, ~2.5× faster, same quality model = ALS(iterations=10, cg_iters=3, anderson_m=5)  Memory overhead: ``m`` copies of the full ``(U ∥ V)`` matrix (~57 MB per copy at 25M ratings, k=64). |

**Examples**

```python
Fold in a new user without retraining the entire model matrix:

>>> import rusket
>>> import numpy as np
>>> from scipy.sparse import csr_matrix
>>> # Fit model on some data
>>> model = rusket.ALS(factors=8).fit(csr_matrix(np.random.randint(0, 2, size=(10, 20))))
>>> # New user interacts with items 3, 5, and 12
>>> latent_factors = model.recalculate_user([3, 5, 12])
>>> # `latent_factors` is a 1D array of length `factors=8`
```

#### `ALS.batch_recommend`

Top-N items for all users efficiently computed in parallel.

```python
from rusket.als import ALS.batch_recommend

ALS.batch_recommend(n: 'int' = 10, exclude_seen: 'bool' = True, format: 'str' = 'polars') -> 'Any'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| n | int, default=10 | The number of items to recommend per user. |
| exclude_seen | bool, default=True | Whether to exclude items the user has already interacted with. |
| format | str, default="polars" | The DataFrame format to return. One of "pandas", "polars", or "spark". |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| DataFrame |  | A DataFrame with columns `user_id`, `item_id`, and `score`. |

---

#### `ALS.fit`

Fit the model to the user-item interaction matrix.

```python
from rusket.als import ALS.fit

ALS.fit(interactions: 'Any' = None) -> 'ALS'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| interactions | sparse matrix or numpy array, optional | If None, uses the matrix prepared by ``from_transactions()``. |

**Raises**

| Exception | Condition |
| --- | --- |
| RuntimeError |  | If the model is already fitted. Create a new instance to refit. |
| TypeError |  | If the input matrix is not a recognizable sparse matrix or numpy array. |

---

#### `ALS.recalculate_user`

Calculate the latent factors for a new or existing user given their interacted items.

```python
from rusket.als import ALS.recalculate_user

ALS.recalculate_user(user_items: 'Any') -> 'np.ndarray'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| user_items | list of int or 1D array-like | The item indices the user has interacted with. If the model was fitted using a DataFrame with item names, these should be the mapped item indices from 0 to n_items - 1.  Note: Confidence values for interactions are currently treated as 1. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| ndarray |  | A 1D numpy array of shape (factors,) containing the user's latent factors. |

**Raises**

| Exception | Condition |
| --- | --- |
| RuntimeError |  | If the model is not fitted. |
| ValueError |  | If any item index is out of bounds. |

---

#### `ALS.recommend_items`

Top-N items for a user. Set exclude_seen=False to include already-seen items.

```python
from rusket.als import ALS.recommend_items

ALS.recommend_items(user_id: 'int', n: 'int' = 10, exclude_seen: 'bool' = True) -> 'tuple[Any, Any]'
```

---

#### `ALS.recommend_users`

Top-N users for an item.

```python
from rusket.als import ALS.recommend_users

ALS.recommend_users(item_id: 'int', n: 'int' = 10) -> 'tuple[Any, Any]'
```

---

---

### `BPR`

Bayesian Personalized Ranking (BPR) model for implicit feedback.

BPR optimizes for ranking rather than reconstruction error (like ALS).
It works by drawing positive items the user interacted with, and negative items
they haven't, and adjusting latent factors to ensure the positive item scores higher.

```python
from rusket.bpr import BPR

BPR(factors: 'int' = 64, learning_rate: 'float' = 0.05, regularization: 'float' = 0.01, iterations: 'int' = 150, seed: 'int' = 42, verbose: 'int' = 0, **kwargs: 'Any') -> 'None'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| factors | int | Number of latent factors (default: 64). |
| learning_rate | float | SGD learning rate (default: 0.05). |
| regularization | float | L2 regularization weight (default: 0.01). |
| iterations | int | Number of passes over the entire interaction dataset (default: 150). |
| seed | int | Random seed for Hogwild! SGD sampling (default: 42). |

#### `BPR.fit`

Fit the BPR model to the user-item interaction matrix.

```python
from rusket.bpr import BPR.fit

BPR.fit(interactions: 'Any' = None) -> 'BPR'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| interactions | sparse matrix or numpy array, optional | If None, uses the matrix prepared by ``from_transactions()``. |

**Raises**

| Exception | Condition |
| --- | --- |
| RuntimeError |  | If the model is already fitted. |
| TypeError |  | If the input matrix is not a recognizable sparse matrix or numpy array. |

---

#### `BPR.recommend_items`

Top-N items for a user.

```python
from rusket.bpr import BPR.recommend_items

BPR.recommend_items(user_id: 'int', n: 'int' = 10, exclude_seen: 'bool' = True) -> 'tuple[Any, Any]'
```

---

---

### `FM`

Factorization Machines (FM) context-aware model for predictive tasks (e.g. CTR).

This model supports binary classification tasks using Log Loss (Binary Cross Entropy).
Inputs should be formatted as a scipy sparse CSR matrix where features are binary (0/1).
Each row is a sample consisting of User, Item, and Context features.

```python
from rusket.fm import FM

FM(factors: 'int' = 8, learning_rate: 'float' = 0.05, regularization: 'float' = 0.01, iterations: 'int' = 100, seed: 'int' = 42, verbose: 'int' = 0, **kwargs: 'Any') -> 'None'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| factors | int | Number of latent factors for the cross terms (default: 8). |
| learning_rate | float | SGD learning rate (default: 0.05). |
| regularization | float | L2 regularization weight (default: 0.01). |
| iterations | int | Number of training epochs (default: 100). |
| seed | int | Random seed for SGD sampling (default: 42). |
| verbose | bool | Whether to print training progress (default: False). |

#### `FM.fit`

Fit the FM model to Context-aware Data.

```python
from rusket.fm import FM.fit

FM.fit(X: 'Any', y: 'Any') -> 'FM'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| X | scipy.sparse.csr_matrix or numpy array | Sparse binary feature matrix of shape (n_samples, n_features). Each row represents a single interaction with all its context features. |
| y | numpy.ndarray | Binary target labels (0.0 or 1.0) of shape (n_samples,). |

---

#### `FM.predict`

Alias for :meth:`predict_proba`.

```python
from rusket.fm import FM.predict

FM.predict(X: 'Any') -> 'Any'
```

---

#### `FM.predict_proba`

Predict the probability (CTR) of interactions.

```python
from rusket.fm import FM.predict_proba

FM.predict_proba(X: 'Any') -> 'Any'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| X | scipy.sparse.csr_matrix or numpy array | Sparse binary feature matrix of shape (n_samples, n_features). |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| numpy.ndarray |  | Predicted probabilities of shape (n_samples,). |

---

---

### `Recommender`

Hybrid recommender combining ALS collaborative filtering, semantic similarities, and association rules.

```python
from rusket.recommend import Recommender

Recommender(model: 'Any | None' = None, rules_df: 'pd.DataFrame | None' = None, item_embeddings: 'np.ndarray | None' = None)
```

#### `Recommender.predict_next_chunk`

Batch-rank the next best products for every user in *user_history_df*.

```python
from rusket.recommend import Recommender.predict_next_chunk

Recommender.predict_next_chunk(user_history_df: 'pd.DataFrame', user_col: 'str' = 'user_id', k: 'int' = 5) -> 'pd.DataFrame'
```

---

#### `Recommender.recommend_for_cart`

Suggest items to add to an active cart using association rules.

```python
from rusket.recommend import Recommender.recommend_for_cart

Recommender.recommend_for_cart(cart_items: 'list[int]', n: 'int' = 5) -> 'list[int]'
```

---

#### `Recommender.recommend_for_user`

Top-N recommendations for a user via Hybrid ALS + Semantic.

```python
from rusket.recommend import Recommender.recommend_for_user

Recommender.recommend_for_user(user_id: 'int', n: 'int' = 5, alpha: 'float' = 0.5, target_item_for_semantic: 'int | None' = None) -> 'tuple[np.ndarray, np.ndarray]'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| user_id | int | The user ID to generate recommendations for. |
| n | int, default=5 | Number of items to return. |
| alpha | float, default=0.5 | Weight blending CF vs Semantic. ``alpha=1.0`` is pure CF. ``alpha=0.0`` is pure semantic. |
| target_item_for_semantic | int \| None, default=None | If provided, semantic similarity is computed against this item. If None, and alpha < 1.0, it computes semantic similarity against the user's most recently interacted item (if history is available) or falls back to pure CF. |

---

---

### `NextBestAction`

Hybrid recommender combining ALS collaborative filtering, semantic similarities, and association rules.

```python
from rusket.recommend import NextBestAction

NextBestAction(model: 'Any | None' = None, rules_df: 'pd.DataFrame | None' = None, item_embeddings: 'np.ndarray | None' = None)
```

#### `NextBestAction.predict_next_chunk`

Batch-rank the next best products for every user in *user_history_df*.

```python
from rusket.recommend import NextBestAction.predict_next_chunk

NextBestAction.predict_next_chunk(user_history_df: 'pd.DataFrame', user_col: 'str' = 'user_id', k: 'int' = 5) -> 'pd.DataFrame'
```

---

#### `NextBestAction.recommend_for_cart`

Suggest items to add to an active cart using association rules.

```python
from rusket.recommend import NextBestAction.recommend_for_cart

NextBestAction.recommend_for_cart(cart_items: 'list[int]', n: 'int' = 5) -> 'list[int]'
```

---

#### `NextBestAction.recommend_for_user`

Top-N recommendations for a user via Hybrid ALS + Semantic.

```python
from rusket.recommend import NextBestAction.recommend_for_user

NextBestAction.recommend_for_user(user_id: 'int', n: 'int' = 5, alpha: 'float' = 0.5, target_item_for_semantic: 'int | None' = None) -> 'tuple[np.ndarray, np.ndarray]'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| user_id | int | The user ID to generate recommendations for. |
| n | int, default=5 | Number of items to return. |
| alpha | float, default=0.5 | Weight blending CF vs Semantic. ``alpha=1.0`` is pure CF. ``alpha=0.0`` is pure semantic. |
| target_item_for_semantic | int \| None, default=None | If provided, semantic similarity is computed against this item. If None, and alpha < 1.0, it computes semantic similarity against the user's most recently interacted item (if history is available) or falls back to pure CF. |

---

---

### `EASE`

Embarrassingly Shallow Autoencoders for Sparse Data (EASE).

An implicit collaborative filtering algorithm that computes a closed-form
item-item similarity matrix by solving a ridge regression problem. EASE
often achieves state-of-the-art recommendation quality and very fast
inference, particularly on datasets with strong item-item correlations.

```python
from rusket.ease import EASE

EASE(regularization: 'float' = 500.0, verbose: 'int' = 0, **kwargs: 'Any') -> 'None'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| regularization | float | L2 regularization weight (lambda). Higher values encourage smaller weights and reduce overfitting. Default is 500.0. |

#### `EASE.fit`

Fit the model to the user-item interaction matrix.

```python
from rusket.ease import EASE.fit

EASE.fit(interactions: 'Any' = None) -> 'EASE'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| interactions | sparse matrix or numpy array, optional | If None, uses the matrix prepared by ``from_transactions()``. |

---

#### `EASE.recommend_items`

Top-N items for a user. Set exclude_seen=False to include already-seen items.

```python
from rusket.ease import EASE.recommend_items

EASE.recommend_items(user_id: 'int', n: 'int' = 10, exclude_seen: 'bool' = True) -> 'tuple[Any, Any]'
```

---

---

### `ItemKNN`

Ultra-fast Sparse Item-Item K-Nearest Neighbors Recommender.

Computes an item-item similarity matrix and only retains the top-K neighbors
per item. Similarity methods include BM25, TF-IDF, Cosine, or unweighted Count.

```python
from rusket.item_knn import ItemKNN

ItemKNN(method: "Literal['bm25', 'tfidf', 'cosine', 'count']" = 'bm25', k: 'int' = 20, bm25_k1: 'float' = 1.2, bm25_b: 'float' = 0.75, verbose: 'int' = 0, **kwargs: 'Any')
```

#### `ItemKNN.fit`

Fit the ItemKNN model.

```python
from rusket.item_knn import ItemKNN.fit

ItemKNN.fit(interactions: 'Any' = None) -> 'ItemKNN'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| interactions | scipy.sparse.csr_matrix, optional | A sparse matrix of shape (n_users, n_items). If None, uses the matrix prepared by ``from_transactions()``. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| ItemKNN |  | The fitted model. |

---

#### `ItemKNN.recommend_items`

Top-N items for a user.

```python
from rusket.item_knn import ItemKNN.recommend_items

ItemKNN.recommend_items(user_id: 'int', n: 'int' = 10, exclude_seen: 'bool' = True) -> 'tuple[Any, Any]'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| user_id | int | The user ID to generate recommendations for. |
| n | int, default=10 | Number of items to return. |
| exclude_seen | bool, default=True | Whether to exclude items the user has already interacted with. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| tuple[np.ndarray, np.ndarray] |  | ``(item_ids, scores)`` sorted by descending score. |

---

---

### `FPMC`

Factorizing Personalized Markov Chains (FPMC) model for sequential recommendation.

FPMC combines Matrix Factorization (modeling user preferences) and Markov Chains
(modeling sequential transitions between items). It is highly effective for tasks
where both personal taste and sequential behavior matter (e.g., next-basket delivery).

```python
from rusket.fpmc import FPMC

FPMC(factors: 'int' = 64, learning_rate: 'float' = 0.05, regularization: 'float' = 0.01, iterations: 'int' = 150, seed: 'int' = 42, verbose: 'int' = 0, **kwargs: 'Any') -> 'None'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| factors | int | Number of latent factors (default: 64). |
| learning_rate | float | SGD learning rate (default: 0.05). |
| regularization | float | L2 regularization weight (default: 0.01). |
| iterations | int | Number of passes over the transitions (default: 150). |
| seed | int | Random seed for sampling (default: 42). |
| verbose | bool | Whether to print training progress (default: False). |

#### `FPMC.fit`

Fit the FPMC model to a list of sequential interactions.

```python
from rusket.fpmc import FPMC.fit

FPMC.fit(sequences: 'list[list[int]] | None' = None, n_items: 'int | None' = None) -> 'FPMC'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| sequences | list of list of int, optional | List of item sequences, where each sequence belongs to a unique user. Users are assigned IDs from 0 to len(sequences)-1. If None, uses data prepared by ``from_transactions()``. |
| n_items | int \| None | Maximum number of items. If None, it is inferred from data. |

---

#### `FPMC.recommend_items`

Top-N sequential items for a user.

```python
from rusket.fpmc import FPMC.recommend_items

FPMC.recommend_items(user_id: 'int', n: 'int' = 10, exclude_seen: 'bool' = True) -> 'tuple[Any, Any]'
```

---

---

### `SVD`

Funk SVD collaborative filtering model.

Biased matrix factorization trained with SGD:
    r̂_ui = μ + b_u + b_i + p_u · q_i

```python
from rusket.svd import SVD

SVD(factors: 'int' = 64, learning_rate: 'float' = 0.005, regularization: 'float' = 0.02, iterations: 'int' = 20, seed: 'int' = 42, verbose: 'int' = 0, **kwargs: 'Any') -> 'None'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| factors | int | Number of latent factors. |
| learning_rate | float | SGD learning rate. |
| regularization | float | L2 regularisation weight. |
| iterations | int | Number of SGD epochs. |
| seed | int | Random seed for reproducibility. |
| verbose | int | Verbosity level (0 = silent, 1+ = progress). |

#### `SVD.batch_recommend`

Top-N items for all users efficiently computed in parallel.

```python
from rusket.svd import SVD.batch_recommend

SVD.batch_recommend(n: 'int' = 10, exclude_seen: 'bool' = True, format: 'str' = 'polars') -> 'Any'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| n | int | Number of items per user. |
| exclude_seen | bool | Whether to filter already-seen items. |
| format | str | Output format: "polars" or "pandas". |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| DataFrame |  | A DataFrame with columns ``user_id``, ``item_id``, and ``score``. |

---

#### `SVD.fit`

Fit the model to the user-item interaction matrix.

```python
from rusket.svd import SVD.fit

SVD.fit(interactions: 'Any' = None) -> 'SVD'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| interactions | scipy.sparse matrix, np.ndarray, pd.DataFrame, or polars DataFrame, optional | User-item interaction matrix with explicit ratings. If None, uses the matrix prepared by ``from_transactions()``. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| self |  |  |

---

#### `SVD.predict`

Predict the rating for a user-item pair.

```python
from rusket.svd import SVD.predict

SVD.predict(user_id: 'int', item_id: 'int') -> 'float'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| user_id | int | User index. |
| item_id | int | Item index. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| float |  | Predicted rating. |

---

#### `SVD.recommend_items`

Top-N items for a user.

```python
from rusket.svd import SVD.recommend_items

SVD.recommend_items(user_id: 'int', n: 'int' = 10, exclude_seen: 'bool' = True) -> 'tuple[Any, Any]'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| user_id | int | User index. |
| n | int | Number of items to recommend. |
| exclude_seen | bool | Whether to filter already-seen items. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| tuple[np.ndarray, np.ndarray] |  | (item_ids, scores) |

---

#### `SVD.recommend_users`

Top-N users for an item.

```python
from rusket.svd import SVD.recommend_users

SVD.recommend_users(item_id: 'int', n: 'int' = 10) -> 'tuple[Any, Any]'
```

---

---

### `LightGCN`

LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.

A state-of-the-art collaborative filtering model that propagates embeddings
over the user–item bipartite graph without non-linear transformations.

Typical training time on ml-100k: < 0.5s/epoch.

```python
from rusket.lightgcn import LightGCN

LightGCN(factors: 'int' = 64, k_layers: 'int' = 3, learning_rate: 'float' = 0.001, lambda_: 'float' = 0.0001, iterations: 'int' = 20, seed: 'int | None' = None, verbose: 'int' = 0, **kwargs: 'Any') -> 'None'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| factors | int | Embedding dimensionality (latent factors). |
| k_layers | int | Number of graph-propagation layers (1–4). |
| learning_rate | float | Adam learning rate. |
| lambda\_ | float | L2 regularization coefficient. |
| iterations | int | Number of training epochs. |
| seed | int or None | Seed for reproducible training. |
| verbose | int | Print training progress. |

#### `LightGCN.fit`

Fit the model to a user-item interaction matrix.

```python
from rusket.lightgcn import LightGCN.fit

LightGCN.fit(interactions: 'Any' = None) -> 'LightGCN'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| interactions | scipy.sparse.csr_matrix or numpy.ndarray, optional | A sparse or dense user-item interaction matrix. If None, uses data prepared by ``from_transactions()``. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| LightGCN |  | The fitted model. |

---

#### `LightGCN.recommend_items`

Top-N items for a user.

```python
from rusket.lightgcn import LightGCN.recommend_items

LightGCN.recommend_items(user_id: 'int', n: 'int' = 10, exclude_seen: 'bool' = True) -> 'tuple[np.ndarray, np.ndarray]'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| user_id | int | Original user ID (before encoding). |
| n | int, default=10 | Number of recommendations. |
| exclude_seen | bool, default=True | Whether to exclude items the user has already interacted with. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| tuple[np.ndarray, np.ndarray] |  | ``(item_ids, scores)`` sorted by descending score. |

---

---

### `SASRec`

SASRec – Self-Attentive Sequential Recommendation.

Applies a causal Transformer to user interaction sequences to predict
the next item. Significantly outperforms Markov-chain methods like FPMC
on long sequences.

```python
from rusket.sasrec import SASRec

SASRec(factors: 'int' = 64, n_layers: 'int' = 2, max_seq: 'int' = 50, learning_rate: 'float' = 0.0005, lambda_: 'float' = 0.0001, iterations: 'int' = 20, seed: 'int | None' = None, verbose: 'int' = 0, **kwargs: 'Any') -> 'None'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| factors | int | Embedding dimensionality. |
| n_layers | int | Number of Transformer blocks. |
| max_seq | int | Maximum input sequence length (older items are dropped). |
| learning_rate | float | SGD learning rate (decays during training). |
| lambda\_ | float | L2 regularization. |
| iterations | int | Number of training epochs. |
| seed | int or None | Seed for reproducibility. |
| verbose | int | Print epoch progress. |

#### `SASRec.fit`

Train SASRec on integer-encoded sequences (0-indexed item IDs).

```python
from rusket.sasrec import SASRec.fit

SASRec.fit(sequences: 'list[list[int]] | None' = None) -> 'SASRec'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| sequences | list of list of int, optional | List of per-user interaction histories (item IDs). If None, uses data prepared by ``from_transactions()``. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| SASRec |  | The fitted model. |

---

#### `SASRec.recommend_items`

Top-N items for a user or an ad-hoc sequence.

```python
from rusket.sasrec import SASRec.recommend_items

SASRec.recommend_items(user_id: 'int | list[int]', n: 'int' = 10, exclude_seen: 'bool' = True) -> 'tuple[np.ndarray, np.ndarray]'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| user_id | int or list[int] | The ID of the user (implicitly 0 to len(sequences)-1 from fit), or a list of items representing an ad-hoc sequence. |
| n | int, default=10 | Number of recommendations. |
| exclude_seen | bool, default=True | Whether to exclude items the user has already interacted with. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| tuple[np.ndarray, np.ndarray] |  | ``(item_ids, scores)`` sorted by descending score. |

---

---

### `PopularityRecommender`

Recommend items by global popularity (interaction count).

A non-personalised baseline that ranks every item by the total number
of interactions it received.  Useful as a sanity-check baseline when
evaluating more sophisticated models.

```python
from rusket.popularity import PopularityRecommender

PopularityRecommender(verbose: 'int' = 0, **kwargs: 'Any') -> 'None'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| verbose | int, default=0 | Verbosity level. |

#### `PopularityRecommender.fit`

Fit the model by counting interactions per item.

```python
from rusket.popularity import PopularityRecommender.fit

PopularityRecommender.fit(interactions: 'Any' = None) -> 'PopularityRecommender'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| interactions | sparse matrix or numpy array, optional | User-item interaction matrix.  If *None*, uses the matrix prepared by ``from_transactions()``. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| PopularityRecommender |  | The fitted model. |

---

#### `PopularityRecommender.recommend_items`

Return the *n* most popular items for a user.

```python
from rusket.popularity import PopularityRecommender.recommend_items

PopularityRecommender.recommend_items(user_id: 'int', n: 'int' = 10, exclude_seen: 'bool' = True) -> 'tuple[np.ndarray, np.ndarray]'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| user_id | int | Internal user index. |
| n | int, default=10 | Number of items to return. |
| exclude_seen | bool, default=True | Whether to exclude items the user has already interacted with. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| tuple[np.ndarray, np.ndarray] |  | ``(item_ids, scores)`` sorted by descending popularity. |

---

---

### `ContentBased`

Content-based recommender using TF-IDF vectorization and cosine similarity.

Recommends items similar to a given item based on textual features
(descriptions, tags, genres, etc.).

```python
from rusket.content_based import ContentBased

ContentBased(max_features: 'int' = 5000, ngram_range: 'tuple[int, int]' = (1, 2), **kwargs: 'Any') -> 'None'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| max_features | int, default=5000 | Maximum number of TF-IDF features to extract. |
| ngram_range | tuple[int, int], default=(1, 2) | Range of n-grams for TF-IDF vectorisation. |

#### `ContentBased.fit`

Compute TF-IDF vectors and the pairwise cosine similarity matrix.

```python
from rusket.content_based import ContentBased.fit

ContentBased.fit() -> 'ContentBased'
```

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| ContentBased |  | The fitted model. |

---

#### `ContentBased.recommend_similar`

Find the *n* most similar items to a given item.

```python
from rusket.content_based import ContentBased.recommend_similar

ContentBased.recommend_similar(item: 'Any', n: 'int' = 10) -> 'tuple[np.ndarray, np.ndarray]'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| item | Any | Item ID (as it appeared in ``item_col`` of the source DataFrame). |
| n | int, default=10 | Number of similar items to return. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| tuple[np.ndarray, np.ndarray] |  | ``(item_ids, similarity_scores)`` sorted by descending similarity. |

---

---

### `HybridRecommender`

Weighted ensemble of multiple recommendation models.

Blends the output of several pre-fitted models by combining their
``recommend_items`` scores with configurable weights.

```python
from rusket.hybrid import HybridRecommender

HybridRecommender(models_and_weights: 'list[tuple[Any, float]]') -> 'None'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| models_and_weights | list[tuple[Any, float]] | List of ``(model, weight)`` pairs.  Each model must implement ``recommend_items(user_id, n, exclude_seen) -> (ids, scores)``. |

#### `HybridRecommender.fit`

No-op — constituent models must be pre-fitted.

```python
from rusket.hybrid import HybridRecommender.fit

HybridRecommender.fit() -> 'HybridRecommender'
```

---

#### `HybridRecommender.recommend_items`

Blend recommendations from all constituent models.

For each model, requests a large candidate set (``n * 3``), maps item
scores into a shared score vector, applies the weight, and returns the
top-*n* from the blended result.

```python
from rusket.hybrid import HybridRecommender.recommend_items

HybridRecommender.recommend_items(user_id: 'int', n: 'int' = 10, exclude_seen: 'bool' = True) -> 'tuple[np.ndarray, np.ndarray]'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| user_id | int | Internal user index. |
| n | int, default=10 | Number of items to return. |
| exclude_seen | bool, default=True | Whether to exclude items already seen. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| tuple[np.ndarray, np.ndarray] |  | ``(item_ids, blended_scores)`` sorted by descending score. |

---

---

### `NMF`

Non-negative Matrix Factorization for collaborative filtering.

Decomposes the user-item interaction matrix **R** into two non-negative
matrices **W** (users × factors) and **H** (factors × items) such that
``R ≈ W @ H``.  The multiplicative update rules guarantee non-negativity
without a projection step.

```python
from rusket.nmf import NMF

NMF(factors: 'int' = 64, iterations: 'int' = 100, regularization: 'float' = 0.01, seed: 'int' = 42, verbose: 'int' = 0, **kwargs: 'Any') -> 'None'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| factors | int, default=64 | Number of latent factors. |
| iterations | int, default=100 | Number of multiplicative update iterations. |
| regularization | float, default=0.01 | L2 regularisation penalty applied to both W and H. |
| seed | int, default=42 | Random seed for initialisation. |
| verbose | int, default=0 | Verbosity level. |

#### `NMF.fit`

Fit via multiplicative update rules.

```python
from rusket.nmf import NMF.fit

NMF.fit(interactions: 'Any' = None) -> 'NMF'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| interactions | sparse matrix or numpy array, optional | User-item interaction matrix.  If *None*, uses the matrix prepared by ``from_transactions()``. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| NMF |  | The fitted model. |

---

#### `NMF.recommend_items`

Top-N items for a user via W @ H^T.

```python
from rusket.nmf import NMF.recommend_items

NMF.recommend_items(user_id: 'int', n: 'int' = 10, exclude_seen: 'bool' = True) -> 'tuple[np.ndarray, np.ndarray]'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| user_id | int | Internal user index. |
| n | int, default=10 | Number of items to return. |
| exclude_seen | bool, default=True | Whether to exclude already-seen items. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| tuple[np.ndarray, np.ndarray] |  | ``(item_ids, scores)`` sorted by descending score. |

---

---

## Analytics & Utilities

### `score_potential`

Cross-selling potential scores — shape ``(n_users, n_items)`` or ``(n_users, len(target_categories))``.

Items the user has already interacted with are masked to ``-inf``.

```python
from rusket.recommend import score_potential

score_potential(user_history: 'list[list[int]]', model: 'Any', target_categories: 'list[int] | None' = None) -> 'np.ndarray'
```

---

### `similar_items`

Find the most similar items to a given item ID based on latent factors.

Computes cosine similarity between the specified item's latent vector
and all other item vectors in the ``item_factors`` matrix.

```python
from rusket.similarity import similar_items

similar_items(model: 'SupportsItemFactors', item_id: 'int', n: 'int' = 5) -> 'tuple[np.ndarray, np.ndarray]'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| model | SupportsItemFactors | A fitted model instance with an ``item_factors`` property. |
| item_id | int | The internal integer index of the target item. |
| n | int | Number of most similar items to return. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| tuple[np.ndarray, np.ndarray] |  | ``(item_ids, cosine_similarities)`` sorted in descending order. |

---

### `find_substitutes`

Substitute/cannibalizing products via negative association rules.

Items with high individual support but low co-occurrence (lift < 1.0)
likely cannibalize each other.

```python
from rusket.analytics import find_substitutes

find_substitutes(rules_df: 'pd.DataFrame', max_lift: 'float' = 0.8) -> 'pd.DataFrame'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| rules_df |  | DataFrame output from ``rusket.association_rules``. |
| max_lift |  | Upper bound for lift; lift < 1.0 implies negative correlation. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| pd.DataFrame sorted ascending by lift (most severe cannibalization first). |  |  |

---

### `customer_saturation`

Customer saturation by unique items/categories bought, split into deciles.

```python
from rusket.analytics import customer_saturation

customer_saturation(df: 'pd.DataFrame', user_col: 'str', category_col: 'str | None' = None, item_col: 'str | None' = None) -> 'pd.DataFrame'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| df |  | Interaction DataFrame. |
| user_col |  | Column identifying the user. |
| category_col |  | Category column (optional; at least one of category/item required). |
| item_col |  | Item column (optional). |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| pd.DataFrame with ``unique_count``, ``saturation_pct``, and ``decile`` columns. |  |  |

---

### `export_item_factors`

Exports latent item factors as a DataFrame for Vector DBs.

This format is ideal for ingesting into FAISS, Pinecone, or Qdrant for
Retrieval-Augmented Generation (RAG) and semantic search.

```python
from rusket.export import export_item_factors

export_item_factors(model: 'SupportsItemFactors', include_labels: 'bool' = True, normalize: 'bool' = False, format: 'str' = 'pandas') -> 'Any'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| model | SupportsItemFactors | A fitted model instance with an ``item_factors`` property. |
| include_labels | bool, default=True | Whether to include the string item labels (if available from the model's fitting method). |
| normalize | bool, default=False | Whether to L2-normalize the factors before export. |
| format | str, default="pandas" | The DataFrame format to return. One of "pandas", "polars", or "spark". |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| Any |  | A DataFrame where each row is an item with columns ``item_id``, optionally ``item_label``, and ``vector`` (a dense 1-D numpy array of the item's latent factors). |

**Examples**

```python
>>> model = rusket.ALS(factors=32).fit(interactions)
>>> df = rusket.export_item_factors(model)
>>> # Ingest into FAISS / Pinecone / Qdrant
>>> vectors = np.stack(df["vector"].values)
```

---

### `PCA`

Principal Component Analysis (PCA).

Linear dimensionality reduction using Singular Value Decomposition
of the centred data, computed entirely in Rust via the ``faer`` crate.

```python
from rusket.pca import PCA

PCA(n_components: 'int' = 2, svd_solver: 'str' = 'auto') -> 'None'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| n_components | int | Number of principal components to keep. |
| Attributes (available after ``fit()``) |  |  |
| ---------- |  |  |
| components_ | np.ndarray | Principal axes in feature space, shape ``(n_components, n_features)``. |
| explained_variance_ | np.ndarray | Variance explained per component (uses ``n - 1`` degrees of freedom). |
| explained_variance_ratio_ | np.ndarray | Fraction of total variance explained per component. |
| singular_values_ | np.ndarray | Singular values corresponding to each component. |
| mean_ | np.ndarray | Per-feature empirical mean estimated from the training data. |
| n_components_ | int | Number of components that were actually fitted (may be less than requested if ``n_components > min(n_samples, n_features)``). |

**Examples**

```python
>>> import numpy as np
>>> import rusket
>>> X = np.random.default_rng(42).standard_normal((100, 10)).astype(np.float32)
>>> pca = rusket.PCA(n_components=3)
>>> pca.fit(X)
PCA(n_components=3)
>>> pca.transform(X).shape
(100, 3)
>>> pca.explained_variance_ratio_.sum()  # close to fraction of total
0.4...
```

#### `PCA.fit`

Fit PCA on the data matrix ``X``.

```python
from rusket.pca import PCA.fit

PCA.fit(X: 'npt.NDArray[Any]') -> 'PCA'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| X | array-like of shape (n_samples, n_features) | Training data. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| self |  |  |

---

#### `PCA.fit_transform`

Fit the model with ``X`` and apply dimensionality reduction.

```python
from rusket.pca import PCA.fit_transform

PCA.fit_transform(X: 'npt.NDArray[Any]') -> 'npt.NDArray[np.float32]'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| X | array-like of shape (n_samples, n_features) |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| X_new | ndarray of shape (n_samples, n_components) |  |

---

#### `PCA.transform`

Apply dimensionality reduction to ``X``.

```python
from rusket.pca import PCA.transform

PCA.transform(X: 'npt.NDArray[Any]') -> 'npt.NDArray[np.float32]'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| X | array-like of shape (n_samples, n_features) |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| X_new | ndarray of shape (n_samples, n_components) |  |

---

---

### `Pipeline`

Multi-stage recommendation pipeline.

Composes multiple recommendation models into a **retrieve → rerank → filter**
funnel, following the architecture used by production recommendation systems
at Twitter/X, YouTube, and Spotify.

```python
from rusket.pipeline import Pipeline

Pipeline(retrieve: 'Any | list[Any] | None' = None, rerank: 'Any | None' = None, filter: 'Callable[[list[Any], list[float]], tuple[list[Any], list[float]]] | None' = None, merge_strategy: "Literal['max', 'mean', 'sum']" = 'max') -> 'None'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| retrieve | list or single model | One or more ``ImplicitRecommender`` instances used for candidate generation.  Each model's ``recommend_items()`` is called and results are merged. |
| rerank | model, optional | An ``ImplicitRecommender`` used to re-score the merged candidate set. Typically a heavier model (e.g. BPR or LightGCN) that produces higher-quality rankings on a smaller candidate pool. |
| filter | callable, optional | A function ``(item_ids, scores) -> (filtered_ids, filtered_scores)`` applied after re-ranking.  Use for block lists, category restrictions, recency filters, NSFW removal, etc. |
| merge_strategy | {'max', 'mean', 'sum'}, default='max' | How to combine scores when multiple retrievers return the same item. |

**Examples**

```python
>>> pipeline = Pipeline(
...     retrieve=[als, item_knn],
...     rerank=bpr,
...     filter=lambda ids, sc: (
...         [i for i in ids if i not in blocked_set],
...         [s for i, s in zip(ids, sc) if i not in blocked_set],
...     ),
... )
>>> items, scores = pipeline.recommend(user_id=42, n=10)
```

#### `Pipeline.recommend`

Run the full pipeline for a single user.

```python
from rusket.pipeline import Pipeline.recommend

Pipeline.recommend(user_id: 'int | Any', n: 'int' = 10, exclude_seen: 'bool' = True, retrieve_k: 'int | None' = None) -> 'tuple[np.ndarray, np.ndarray]'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| user_id | int or any | The user to generate recommendations for. |
| n | int, default=10 | Number of final items to return. |
| exclude_seen | bool, default=True | Whether to exclude items the user has already interacted with. |
| retrieve_k | int, optional | Number of candidates per retriever.  Defaults to ``n * 10`` to produce a wide candidate pool for re-ranking. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| tuple[np.ndarray, np.ndarray] |  | ``(item_ids, scores)`` arrays, sorted by descending score. |

---

#### `Pipeline.recommend_batch`

Batch recommendations for multiple users.

Uses the **Rust-accelerated** fast path when all models expose
``user_factors`` / ``item_factors`` and share the same user indexing.
Falls back to the Python per-user loop otherwise.

```python
from rusket.pipeline import Pipeline.recommend_batch

Pipeline.recommend_batch(user_ids: 'list[int | Any] | np.ndarray | None' = None, n: 'int' = 10, exclude_seen: 'bool' = True, retrieve_k: 'int | None' = None, format: 'str' = 'pandas') -> 'Any'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| user_ids | list or array, optional | Users to score.  If None, uses all users from the first retriever. |
| n | int, default=10 | Items per user. |
| exclude_seen | bool, default=True | Whether to exclude items users have already interacted with. |
| retrieve_k | int, optional | Candidates per retriever (default: ``n * 10``). |
| format | str, default='pandas' | Output format: ``'pandas'``, ``'polars'``, or ``'records'``. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| DataFrame or list of dicts |  | Columns: ``user_id``, ``item_ids``, ``scores``. |

---

---

## Visualization (`rusket.viz`)

Graph and visualization utilities.  Requires `networkx` (`pip install networkx`).

### `rusket.viz.to_networkx`

Convert a Rusket association rules DataFrame into a NetworkX Directed Graph.

Nodes represent individual items. Directed edges represent rules
(antecedent → consequent). Edge weights are set by the ``edge_attr``
parameter (typically lift or confidence).

This is extremely useful for running community detection algorithms
(e.g., Louvain, Girvan-Newman) to automatically discover **product clusters**,
or for visualising cross-selling patterns as a force-directed graph.

```python
from rusket.viz import rusket.viz.to_networkx

rusket.viz.to_networkx(rules_df: 'pd.DataFrame', source_col: 'str' = 'antecedents', target_col: 'str' = 'consequents', edge_attr: 'str' = 'lift') -> 'networkx.DiGraph'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| rules_df | pd.DataFrame | A Pandas DataFrame generated by ``rusket.association_rules()``. |
| source_col | str, default='antecedents' | Column name containing antecedents (graph edge sources). |
| target_col | str, default='consequents' | Column name containing consequents (graph edge targets). |
| edge_attr | str, default='lift' | The metric to use as edge weight/thickness. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| networkx.DiGraph |  | A directed graph of the association rules. If ``rules_df`` is empty, returns an empty ``DiGraph``. |

> **Notes**
> Requires the ``networkx`` package (``pip install networkx``).
When multiple rules produce the same directed edge, only the highest-weight
rule is retained.

**Examples**

```python
>>> import rusket
>>> G = rusket.viz.to_networkx(rules_df, edge_attr="lift")
>>> # Community detection with networkx
>>> import networkx.algorithms.community as nx_comm
>>> communities = nx_comm.greedy_modularity_communities(G.to_undirected())
```

---

## Distributed Spark API (`rusket.spark`)

All functions in `rusket.spark` distribute computation across PySpark partitions using Apache Arrow (zero-copy) for maximum throughput.

### `rusket.spark.mine_grouped`

Distribute Market Basket Analysis across PySpark partitions.

This function groups a PySpark DataFrame by `group_col` and applies
`rusket.mine` to each group concurrently across the cluster.

It assumes the input PySpark DataFrame is formatted like a dense
boolean matrix (One-Hot Encoded) per group, where rows are transactions.

```python
from rusket.spark import rusket.spark.mine_grouped

rusket.spark.mine_grouped(df: 'Any', group_col: 'str', min_support: 'float' = 0.5, max_len: 'int | None' = None, method: 'str' = 'auto', use_colnames: 'bool' = True) -> 'Any'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| df |  | The input `pyspark.sql.DataFrame`. |
| group_col |  | The column to group by (e.g. `store_id`). |
| min_support |  | Minimum support threshold. |
| max_len |  | Maximum itemset length. |
| method |  | Algorithm to use: 'auto', 'fpgrowth', or 'eclat'. |
| use_colnames |  | If True, returns item names instead of column indices. Must be True for PySpark `applyInArrow` schema consistency. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| pyspark.sql.DataFrame |  | A PySpark DataFrame containing: - `group_col` - `support` (float) - `itemsets` (array of strings) |

---

### `rusket.spark.rules_grouped`

Distribute Association Rule Mining across PySpark partitions.

This takes the frequent itemsets DataFrame (output of `mine_grouped`)
and applies `association_rules` uniformly across the groups.

```python
from rusket.spark import rusket.spark.rules_grouped

rusket.spark.rules_grouped(df: 'Any', group_col: 'str', num_itemsets: 'dict[Any, int] | int', metric: 'str' = 'confidence', min_threshold: 'float' = 0.8) -> 'Any'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| df |  | The PySpark `DataFrame` containing frequent itemsets. |
| group_col |  | The column to group by. |
| num_itemsets |  | A dictionary mapping group IDs to their total transaction count, or a single integer if all groups have the same number of transactions. |
| metric |  | The metric to filter by (e.g. "confidence", "lift"). |
| min_threshold |  | The minimal threshold for the evaluation metric. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| pyspark.sql.DataFrame |  | A DataFrame containing antecedents, consequents, and all rule metrics, prepended with the `group_col`. |

---

### `rusket.spark.prefixspan_grouped`

Distribute Sequential Pattern Mining (PrefixSpan) across PySpark partitions.

This function groups a PySpark DataFrame by `group_col` and applies
`PrefixSpan.from_transactions` to each group concurrently across the cluster.

```python
from rusket.spark import rusket.spark.prefixspan_grouped

rusket.spark.prefixspan_grouped(df: 'Any', group_col: 'str', user_col: 'str', time_col: 'str', item_col: 'str', min_support: 'int' = 1, max_len: 'int | None' = None) -> 'Any'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| df |  | The input `pyspark.sql.DataFrame`. |
| group_col |  | The column to group by (e.g. `store_id`). |
| user_col |  | The column identifying the sequence within each group (e.g., `user_id` or `session_id`). |
| time_col |  | The column used for ordering events within a sequence. |
| item_col |  | The column containing the items. |
| min_support |  | The minimum absolute support (number of sequences a pattern must appear in). |
| max_len |  | Maximum length of the sequential patterns to mine. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| pyspark.sql.DataFrame |  | A PySpark DataFrame containing: - `group_col` - `support` (long/int64) - `sequence` (array of strings) |

---

### `rusket.spark.hupm_grouped`

Distribute High-Utility Pattern Mining (HUPM) across PySpark partitions.

This function groups a PySpark DataFrame by `group_col` and applies
`HUPM.from_transactions` to each group concurrently across the cluster.

```python
from rusket.spark import rusket.spark.hupm_grouped

rusket.spark.hupm_grouped(df: 'Any', group_col: 'str', transaction_col: 'str', item_col: 'str', utility_col: 'str', min_utility: 'float', max_len: 'int | None' = None) -> 'Any'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| df |  | The input `pyspark.sql.DataFrame`. |
| group_col |  | The column to group by (e.g. `store_id`). |
| transaction_col |  | The column identifying the transaction within each group. |
| item_col |  | The column containing the numeric item IDs. |
| utility_col |  | The column containing the numeric utility (e.g., profit) of the item in the transaction. |
| min_utility |  | The minimum total utility required to consider a pattern "high-utility". |
| max_len |  | Maximum length of the itemsets to mine. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| pyspark.sql.DataFrame |  | A PySpark DataFrame containing: - `group_col` - `utility` (double/float64) - `itemset` (array of longs/int64) |

---

### `rusket.spark.recommend_batches`

Distribute Batch Recommendations across PySpark partitions.

This function uses `mapInArrow` to process partitions of users concurrently,
applying a pre-fitted `Recommender` (or `ALS`) to each chunk.

```python
from rusket.spark import rusket.spark.recommend_batches

rusket.spark.recommend_batches(df: 'Any', model: 'Any', user_col: 'str' = 'user_id', k: 'int' = 5) -> 'Any'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| df |  | The PySpark `DataFrame` containing user histories (must contain `user_col`). |
| model |  | The pre-trained `Recommender` or `ALS` model instance to use for scoring. |
| user_col |  | The column identifying the user. |
| k |  | The number of top recommendations to return per user. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| pyspark.sql.DataFrame |  | A DataFrame with two columns: - `user_col` - `recommended_items` (array of longs/int64) |

---

### `rusket.spark.to_spark`

Convert a Pandas or Polars DataFrame into a PySpark DataFrame.

```python
from rusket.spark import rusket.spark.to_spark

rusket.spark.to_spark(spark_session: 'Any', df: 'Any') -> 'Any'
```

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| spark_session |  | The active PySpark `SparkSession`. |
| df |  | The `pd.DataFrame` or `pl.DataFrame` to convert. |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| pyspark.sql.DataFrame |  | The resulting PySpark DataFrame. |

---

