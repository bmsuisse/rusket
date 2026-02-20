# API Reference

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
| `df` | `pd.DataFrame` | — | Output of `fpgrowth()` with columns `['support', 'itemsets']`. |
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
