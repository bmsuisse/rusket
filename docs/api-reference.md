# API Reference

## `mine` (Recommended)

```python
from rusket import mine

mine(
    df,
    min_support=0.5,
    null_values=False,
    use_colnames=False,
    max_len=None,
    method="auto",
    verbose=0,
) -> pd.DataFrame
```

Dynamically selects the optimal mining algorithm (`fpgrowth` or `eclat`) based on the dataset density heuristically. It's highly recommended to use this entry point instead of calling the algorithms directly.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pd.DataFrame \| pl.DataFrame` | — | One-hot encoded DataFrame. Rows are transactions, columns are items. Accepts bool or 0/1 integer values. Sparse pandas DataFrames are supported via the CSR path. |
| `min_support` | `float` | `0.5` | Minimum support threshold in `(0, 1]`. Items occurring in fewer than `ceil(min_support × n_rows)` transactions are excluded. |
| `null_values` | `bool` | `False` | Allow NaN values in `df` (pandas only). When `True`, NaNs are treated as zeros. |
| `use_colnames` | `bool` | `False` | If `True`, itemsets contain column names instead of integer column indices. |
| `max_len` | `int \| None` | `None` | Maximum itemset length. `None` means unlimited. |
| `method` | `"auto" \| "fpgrowth" \| "eclat"` | `"auto"` | Algorithm to use. "auto" selects Eclat for sparse datasets and FP-Growth for dense ones. |
| `verbose` | `int` | `0` | Verbosity level. |

### Returns

`pandas.DataFrame` with columns:

| Column | Type | Description |
|---|---|---|
| `support` | `float` | Support of the itemset (fraction of transactions). |
| `itemsets` | `frozenset` | Set of column indices (or names when `use_colnames=True`). |

### Raises

| Exception | Condition |
|---|---|
| `ValueError` | `min_support` ≤ 0 |
| `TypeError` | `df` is not a pandas or Polars DataFrame |

### Examples

=== "Dense pandas"

    ```python
    import pandas as pd
    from rusket import mine

    df = pd.DataFrame({"a": [1,1,0], "b": [1,0,1], "c": [0,1,1]})
    freq = mine(df, min_support=0.5, use_colnames=True)
    ```

=== "Sparse pandas"

    ```python
    import pandas as pd
    from pandas.arrays import SparseArray
    from rusket import mine

    df = pd.DataFrame.sparse.from_spmatrix(my_csr_matrix, columns=items)
    freq = mine(df, min_support=0.1, use_colnames=True)
    ```

=== "Polars"

    ```python
    import polars as pl
    from rusket import mine

    df = pl.DataFrame({"a": [1,1,0], "b": [1,0,1], "c": [0,1,1]})
    freq = mine(df, min_support=0.5, use_colnames=True)
    ```

---

## `fpgrowth`

```python
from rusket import fpgrowth

fpgrowth(
    df,
    min_support=0.5,
    null_values=False,
    use_colnames=False,
    max_len=None,
    verbose=0,
) -> pd.DataFrame
```

Find frequent itemsets in a one-hot encoded transaction DataFrame using the **FP-Growth** algorithm, implemented in Rust for maximum performance.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pd.DataFrame \| pl.DataFrame` | — | One-hot encoded DataFrame. Rows are transactions, columns are items. Accepts bool or 0/1 integer values. Sparse pandas DataFrames are supported via the CSR path. |
| `min_support` | `float` | `0.5` | Minimum support threshold in `(0, 1]`. Items occurring in fewer than `ceil(min_support × n_rows)` transactions are excluded. |
| `null_values` | `bool` | `False` | Allow NaN values in `df` (pandas only). When `True`, NaNs are treated as zeros. |
| `use_colnames` | `bool` | `False` | If `True`, itemsets contain column names instead of integer column indices. |
| `max_len` | `int \| None` | `None` | Maximum itemset length. `None` means unlimited. |
| `verbose` | `int` | `0` | Verbosity level. Currently unused; kept for API compatibility with mlxtend. |

### Returns

`pandas.DataFrame` with columns:

| Column | Type | Description |
|---|---|---|
| `support` | `float` | Support of the itemset (fraction of transactions). |
| `itemsets` | `frozenset` | Set of column indices (or names when `use_colnames=True`). |

### Raises

| Exception | Condition |
|---|---|
| `ValueError` | `min_support` ≤ 0 |
| `TypeError` | `df` is not a pandas or Polars DataFrame |

### Examples

=== "Dense pandas"

    ```python
    import pandas as pd
    from rusket import fpgrowth

    df = pd.DataFrame({"a": [1,1,0], "b": [1,0,1], "c": [0,1,1]})
    freq = fpgrowth(df, min_support=0.5, use_colnames=True)
    ```

=== "Sparse pandas"

    ```python
    import pandas as pd
    from pandas.arrays import SparseArray
    from rusket import fpgrowth

    df = pd.DataFrame.sparse.from_spmatrix(my_csr_matrix, columns=items)
    freq = fpgrowth(df, min_support=0.1, use_colnames=True)
    ```

=== "Polars"

    ```python
    import polars as pl
    from rusket import fpgrowth

    df = pl.DataFrame({"a": [1,1,0], "b": [1,0,1], "c": [0,1,1]})
    freq = fpgrowth(df, min_support=0.5, use_colnames=True)
    ```

---

## `eclat`

```python
from rusket import eclat

eclat(
    df,
    min_support=0.5,
    null_values=False,
    use_colnames=False,
    max_len=None,
    verbose=0,
) -> pd.DataFrame
```

Find frequent itemsets using the **Eclat** algorithm (vertical bitset representation with hardware `popcnt` for support counting). Same parameters and return value as `fpgrowth`.

### Parameters

Same as `fpgrowth` — see above.

### Returns

`pandas.DataFrame` with columns `['support', 'itemsets']` — identical format to `fpgrowth`.

### Examples

```python
import pandas as pd
from rusket import eclat

df = pd.DataFrame({"a": [True,True,False], "b": [True,False,True], "c": [False,True,True]})
freq = eclat(df, min_support=0.5, use_colnames=True)
```

## `association_rules`

```python
from rusket import association_rules

association_rules(
    df,
    num_itemsets,
    df_orig=None,
    null_values=False,
    metric="confidence",
    min_threshold=0.8,
    support_only=False,
    return_metrics=ALL_METRICS,
) -> pd.DataFrame
```

Generate association rules from a DataFrame of frequent itemsets. The rule-generation and metric computation is performed in Rust.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pd.DataFrame` | — | Output of `mine()` or `fpgrowth()` with columns `['support', 'itemsets']`. |
| `num_itemsets` | `int` | — | **Total number of transactions** in the original dataset (= `len(original_df)`). |
| `df_orig` | `pd.DataFrame \| None` | `None` | Original (non-binarised) DataFrame. Only needed when `null_values=True`. |
| `null_values` | `bool` | `False` | Apply null-value correction (not yet on the Rust path; falls back gracefully). |
| `metric` | `str` | `"confidence"` | Primary filter metric. See [Metrics](#metrics) below. |
| `min_threshold` | `float` | `0.8` | Minimum value of `metric` for a rule to be included in the result. |
| `support_only` | `bool` | `False` | If `True`, only compute support; fill all other metrics with `NaN`. |
| `return_metrics` | `list[str]` | all 12 | Metric columns to include in the result DataFrame. |

### Returns

`pandas.DataFrame` with columns:

| Column | Type | Description |
|---|---|---|
| `antecedents` | `frozenset` | Left-hand side (LHS) of the rule. |
| `consequents` | `frozenset` | Right-hand side (RHS) of the rule. |
| `antecedent support` | `float` | Support of the antecedent alone. |
| `consequent support` | `float` | Support of the consequent alone. |
| `support` | `float` | Support of the full rule (LHS ∪ RHS). |
| `confidence` | `float` | P(RHS \| LHS). |
| `lift` | `float` | Confidence / consequent support. |
| `representativity` | `float` | Fraction of transactions covered by the rule. |
| `leverage` | `float` | Support − antecedent_support × consequent_support. |
| `conviction` | `float` | (1 − consequent_support) / (1 − confidence). |
| `zhangs_metric` | `float` | Zhang's correlation metric. |
| `jaccard` | `float` | Jaccard similarity of antecedent and consequent. |
| `certainty` | `float` | Certainty factor. |
| `kulczynski` | `float` | Kulczynski measure. |

### Metrics

The `metric` parameter accepts any of:

`confidence` · `lift` · `support` · `leverage` · `conviction` ·
`zhangs_metric` · `jaccard` · `certainty` · `kulczynski` ·
`representativity` · `antecedent support` · `consequent support`

### Raises

| Exception | Condition |
|---|---|
| `ValueError` | `df` is missing `'support'` or `'itemsets'` columns |
| `ValueError` | `df` is empty |
| `ValueError` | Unknown `metric` value and `support_only=False` |

---

## `FPMiner`

```python
from rusket import FPMiner

miner = FPMiner(n_items=500_000)
```

**Streaming accumulator** for billion-scale datasets.  Accepts chunks of
`(transaction_id, item_id)` integer arrays one at a time — Rust accumulates
them in a `HashMap<i64, Vec<i32>>`.  Peak **Python** memory = one chunk.

### Constructor

| Parameter | Type | Description |
|---|---|---|
| `n_items` | `int` | Number of distinct items (column count). Item IDs must be in `[0, n_items)`. |

### Methods

#### `add_chunk(txn_ids, item_ids) → self`

Feed a chunk of integer pairs into the accumulator.

| Parameter | Type | Description |
|---|---|---|
| `txn_ids` | `np.ndarray[int64]` | Transaction IDs (arbitrary integers). |
| `item_ids` | `np.ndarray[int32]` | Item column indices `[0, n_items)`. |

#### `mine(min_support, max_len, use_colnames, column_names, method) → pd.DataFrame`

Mine frequent itemsets from all accumulated data.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `min_support` | `float` | `0.5` | Minimum support in `(0, 1]`. |
| `max_len` | `int \| None` | `None` | Maximum itemset length. |
| `use_colnames` | `bool` | `False` | Return column names instead of indices. |
| `column_names` | `list[str] \| None` | `None` | Names for columns when `use_colnames=True`. |
| `method` | `"fpgrowth" \| "eclat"` | `"fpgrowth"` | Mining algorithm. |

#### `reset()`

Free all accumulated data.

### Properties

| Property | Type | Description |
|---|---|---|
| `n_transactions` | `int` | Distinct transactions accumulated so far. |
| `n_items` | `int` | Column count (set at construction). |

### Example

```python
import numpy as np
from rusket import FPMiner

miner = FPMiner(n_items=500_000)

# Process a Parquet file in 10M-row chunks
for chunk in pd.read_parquet("orders.parquet", chunksize=10_000_000):
    txn = chunk["txn_id"].to_numpy(dtype="int64")
    item = chunk["item_idx"].to_numpy(dtype="int32")
    miner.add_chunk(txn, item)

freq = miner.mine(min_support=0.001, max_len=3, use_colnames=False)
```

---

## `from_transactions_csr`

```python
from rusket import from_transactions_csr

csr, column_names = from_transactions_csr(
    data,
    transaction_col=None,
    item_col=None,
    chunk_size=10_000_000,
)
```

Converts long-format transactional data to a raw `scipy.sparse.csr_matrix`
and a list of column names, for direct input into `fpgrowth()` or `eclat()`.

Accepts the **same input types** as `from_transactions`, plus a **file path**
to a Parquet file, which is read in chunks to avoid loading all data into
memory at once.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `pd.DataFrame \| str \| Path` | — | A pandas/Polars/Spark DataFrame, or a file path to a Parquet file. |
| `transaction_col` | `str \| None` | `None` | Transaction ID column name (defaults to first column). |
| `item_col` | `str \| None` | `None` | Item column name (defaults to second column). |
| `chunk_size` | `int` | `10_000_000` | Rows per chunk for large files. |

### Returns

`tuple[scipy.sparse.csr_matrix, list[str]]` — CSR matrix + column names.

### Example

```python
from rusket import from_transactions_csr, mine

# From Parquet — never loads entire file
csr, names = from_transactions_csr("orders.parquet", chunk_size=10_000_000)
freq = mine(csr, min_support=0.001, use_colnames=True, column_names=names)
```

---

## Recommendation & Analytics

### `rusket.recommend.Recommender`

High-level Hybrid Recommender that combines ALS and Association Rules.

```python
from rusket import Recommender

rec = Recommender(als_model=als, rules_df=rules_df)
rec.recommend_for_user(user_id=42, n=5)
rec.recommend_for_cart(cart_items=[14, 7], n=3)
```

### `rusket.score_potential`

Calculates cross-selling potential scores to identify "missed opportunities" for users who should have bought an item by now but haven't.

```python
from rusket import score_potential

scores = score_potential(user_history, als_model, target_categories=[101, 102])
```

### `rusket.similar_items`

Find the most similar items to a given item ID based on ALS/BPR latent factors using fast Cosine Similarity.

```python
from rusket import similar_items

similar_ids, scores = similar_items(als_model, item_id=99, n=5)
```

### `rusket.export_item_factors`

Exports ALS/BPR latent item factors as a Pandas DataFrame for Vector DBs (FAISS, Qdrant, Pinecone) for Retrieval-Augmented Generation (RAG).

```python
from rusket import export_item_factors

df_vectors = export_item_factors(als_model, include_labels=True)
```

### `rusket.viz.to_networkx`

Converts a `rusket` association rules DataFrame into a NetworkX Directed Graph. Useful for product clustering and visualization.

```python
from rusket.viz import to_networkx

G = to_networkx(rules_df, edge_attr="lift")
```
