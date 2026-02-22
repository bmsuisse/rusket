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

## Class-Based (OOP) Mining API

All mining algorithms (`FPGrowth`, `Eclat`, `AutoMiner`, `HUPM`, `PrefixSpan`) expose a uniform OOP interface via `BaseModel`. This is convenient when you want to go from raw transactional data to association rules without handling intermediate DataFrames.

### `Miner.from_transactions`

```python
model = FPGrowth.from_transactions(
    data,                        # pd.DataFrame | pl.DataFrame | list[list]
    transaction_col=None,        # defaults to first column
    item_col=None,               # defaults to second column
    verbose=0,
    # + any algorithm kwargs: min_support, max_len, ...
)
```

Also available as `.from_pandas(df, ...)` and `.from_polars(df, ...)`.

### `Miner.mine`

```python
freq = model.mine(
    min_support=0.5,
    use_colnames=False,
    max_len=None,
) -> pd.DataFrame
```

### `RuleMinerMixin.association_rules`

Available on all mining classes (`FPGrowth`, `Eclat`, `AutoMiner`, `HUPM`):

```python
rules = model.association_rules(
    metric="confidence",
    min_threshold=0.8,
    return_metrics=None,   # None = all metrics
) -> pd.DataFrame
```

`num_itemsets` is inferred automatically from the data passed to `from_transactions`.

### `RuleMinerMixin.recommend_items`

```python
suggestions = model.recommend_items(
    items=["bread", "milk"],  # current cart contents
    n=5,
) -> list[Any]
```

Generates association rules on-the-fly (lift ≥ 1.0) and returns the top `n` consequents ordered by lift and confidence.

### Example

```python
from rusket import FPGrowth, AutoMiner
import pandas as pd

df = pd.DataFrame({
    "order_id": [1, 1, 2, 2, 3, 3, 3],
    "item":     ["bread", "milk", "bread", "eggs", "milk", "eggs", "butter"],
})

# All three lines are equivalent:
model = FPGrowth.from_transactions(df, min_support=0.4)
# model = Eclat.from_transactions(df, min_support=0.4)
# model = AutoMiner.from_transactions(df, min_support=0.4)

freq  = model.mine(use_colnames=True)
rules = model.association_rules(metric="lift", min_threshold=1.0)
cart_suggestions = model.recommend_items(["bread"], n=3)
```

---

## `ALS`

**Alternating Least Squares** for implicit feedback collaborative filtering.

```python
from rusket import ALS

als = ALS(
    factors=64,
    regularization=0.01,
    alpha=40.0,
    iterations=15,
    seed=42,
    verbose=False,
    cg_iters=10,
    use_cholesky=False,
    anderson_m=0,
)
```

### Constructor Parameters

| Parameter | Default | Description |
|---|---|---|
| `factors` | `64` | Number of latent factors. |
| `regularization` | `0.01` | L2 regularisation weight. |
| `alpha` | `40.0` | Confidence scaling: `confidence = 1 + alpha × r`. |
| `iterations` | `15` | Number of ALS outer iterations. |
| `seed` | `42` | Random seed. |
| `verbose` | `False` | Print per-iteration loss. |
| `cg_iters` | `10` | Conjugate Gradient iterations per user/item solve. Use `3` for very large datasets. Ignored when `use_cholesky=True`. |
| `use_cholesky` | `False` | Use a direct Cholesky solve instead of iterative CG. Exact; faster when users have many interactions relative to `factors`. |
| `anderson_m` | `0` | Anderson Acceleration history window (0 = disabled). A value of `5` typically reduces ALS iterations by 30–50% at no quality cost. |

### Methods

#### `fit(interactions) → ALS`

Fit the model to a user-item interaction matrix.

| Parameter | Type | Description |
|---|---|---|
| `interactions` | `scipy.sparse.csr_matrix \| np.ndarray` | User × Item matrix with implicit feedback values. |

#### `from_transactions(data, user_col, item_col, rating_col, ...) → ALS`

Fit directly from a long-format event log DataFrame.

#### `recommend_items(user_id, n, exclude_seen) → tuple[np.ndarray, np.ndarray]`

Top-N items for a user.

| Parameter | Default | Description |
|---|---|---|
| `user_id` | — | Integer index of the user. |
| `n` | `10` | Number of items to return. |
| `exclude_seen` | `True` | Exclude items the user already interacted with. |

#### `recommend_users(item_id, n) → tuple[np.ndarray, np.ndarray]`

Top-N users for an item (reverse lookup).

### Properties

| Property | Description |
|---|---|
| `user_factors` | User factor matrix `(n_users, factors)`. |
| `item_factors` | Item factor matrix `(n_items, factors)`. |

---

## `BPR`

**Bayesian Personalized Ranking** for implicit feedback collaborative filtering. Optimises for ranking rather than reconstruction error.

```python
from rusket import BPR

bpr = BPR(
    factors=64,
    learning_rate=0.05,
    regularization=0.01,
    iterations=150,
    seed=42,
    verbose=False,
)
```

Same `fit()`, `from_transactions()`, `recommend_items()`, `user_factors`, and `item_factors` interface as `ALS`.

---

## Recommendation & Analytics

### `rusket.recommend.Recommender`

High-level Hybrid Recommender that combines ALS and Association Rules.

```python
from rusket import Recommender

rec = Recommender(als_model=als, rules_df=rules_df, item_embeddings=None)

# Personalized recommendations (ALS)
items, scores = rec.recommend_for_user(
    user_id=42,
    n=5,
    alpha=1.0,                      # 1.0 = pure CF, 0.0 = pure semantic
    target_item_for_semantic=None,  # anchor item for semantic blending
)

# Cart cross-sell (Association Rules)
rec.recommend_for_cart(cart_items=[14, 7], n=3)

# Batch recommendations for a user history DataFrame
batch_df = rec.predict_next_chunk(user_history_df, user_col="user_id", k=5)
```

### `rusket.score_potential`

Calculates cross-selling potential scores to identify "missed opportunities".

```python
from rusket import score_potential

scores = score_potential(user_history, als_model, target_categories=[101, 102])
```

### `rusket.similar_items`

Find the most similar items to a given item ID based on ALS/BPR latent factors.

```python
from rusket import similar_items

similar_ids, scores = similar_items(als_model, item_id=99, n=5)
```

### `rusket.export_item_factors`

Export ALS/BPR item factors as a Pandas DataFrame for Vector DBs.

```python
from rusket import export_item_factors

df_vectors = export_item_factors(als_model, include_labels=True)
```

### `rusket.find_substitutes`

Identify cannibalized / substitutable products via negative association rules (lift < 1.0).

```python
from rusket import find_substitutes

substitutes = find_substitutes(rules_df, max_lift=0.8)
# Returns a DataFrame sorted ascending by lift (most severe cannibalization first)
```

### `rusket.customer_saturation`

Segment users by their category/item purchase depth into deciles.

```python
from rusket import customer_saturation

saturation = customer_saturation(
    df,
    user_col="user_id",
    category_col="category_id",  # or item_col="item_id"
)
# Returns a DataFrame with unique_count, saturation_pct, and decile columns
```

### `rusket.viz.to_networkx`

Convert association rules into a NetworkX Directed Graph for community detection.

```python
from rusket.viz import to_networkx

G = to_networkx(rules_df, edge_attr="lift")
```

---

## Distributed Spark API (`rusket.spark`)

### `mine_grouped`

Distribute Market Basket Analysis across PySpark partitions (grouped by a key column).

```python
import rusket.spark

freq_df = rusket.spark.mine_grouped(
    df=spark_df,
    group_col="store_id",
    min_support=0.05,
    method="auto",   # "auto" | "fpgrowth" | "eclat"
    use_colnames=True,
    max_len=None,
) -> pyspark.sql.DataFrame  # store_id, support, itemsets (array<string>)
```

### `rules_grouped`

Distribute Association Rule Mining across PySpark partitions using the output of `mine_grouped`.

```python
rules_df = rusket.spark.rules_grouped(
    df=freq_spark_df,
    group_col="store_id",
    num_itemsets={"A": 10_000, "B": 5_000},  # or a single int for all groups
    metric="confidence",
    min_threshold=0.8,
) -> pyspark.sql.DataFrame  # store_id, antecedents, consequents, + 11 metrics
```

### `prefixspan_grouped`

Distribute Sequential Pattern Mining (PrefixSpan) across PySpark partitions.

```python
seq_df = rusket.spark.prefixspan_grouped(
    df=spark_df,
    group_col="store_id",
    user_col="user_id",
    time_col="timestamp",
    item_col="item_id",
    min_support=10,
    max_len=None,
) -> pyspark.sql.DataFrame  # store_id, support (long), sequence (array<string>)
```

### `hupm_grouped`

Distribute High-Utility Pattern Mining (HUPM) across PySpark partitions.

```python
hupm_df = rusket.spark.hupm_grouped(
    df=spark_df,
    group_col="store_id",
    transaction_col="txn_id",
    item_col="item_id",
    utility_col="profit",
    min_utility=50.0,
    max_len=None,
) -> pyspark.sql.DataFrame  # store_id, utility (double), itemset (array<long>)
```

### `recommend_batches`

Distribute batch recommendations across PySpark partitions using a pre-fitted model.

```python
rec_df = rusket.spark.recommend_batches(
    df=user_history_spark_df,
    model=als_or_recommender,
    user_col="user_id",
    k=5,
) -> pyspark.sql.DataFrame  # user_id, recommended_items (array<int>)
```

### `to_spark`

Convert a Pandas or Polars DataFrame to a PySpark DataFrame.

```python
spark_df = rusket.spark.to_spark(spark_session, df)
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
