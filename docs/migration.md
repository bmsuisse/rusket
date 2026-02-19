# Migration from mlxtend

rusket is designed as a **drop-in replacement** for `mlxtend.frequent_patterns`.  
In the vast majority of cases the only change you need is the import line.

## Import change

=== "Before (mlxtend)"

    ```python
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    ```

=== "After (rusket)"

    ```python
    from rusket import fpgrowth, association_rules
    ```

---

## API comparison

### `fpgrowth`

| Parameter | mlxtend | rusket | Notes |
|---|---|---|---|
| `df` | `pd.DataFrame` | `pd.DataFrame \| pl.DataFrame` | rusket also accepts Polars |
| `min_support` | `float` | `float` | identical |
| `use_colnames` | `bool` | `bool` | identical |
| `max_len` | `int\|None` | `int\|None` | identical |
| `verbose` | `int` | `int` | accepted but unused |
| `null_values` | `bool` | `bool` | pandas only |

### `association_rules`

| Parameter | mlxtend | rusket | Notes |
|---|---|---|---|
| `df` | `pd.DataFrame` | `pd.DataFrame` | output of `fpgrowth` |
| `num_itemsets` | `int` | `int` | identical |
| `metric` | `str` | `str` | identical (12 metrics) |
| `min_threshold` | `float` | `float` | identical |
| `support_only` | `bool` | `bool` | identical |
| `return_metrics` | `list[str]` | `list[str]` | identical |

---

## Return value

Both functions return **identical DataFrame structures**:

- `fpgrowth` → `pd.DataFrame` with `['support', 'itemsets']`
- `association_rules` → `pd.DataFrame` with `['antecedents', 'consequents', ...metrics]`

Itemsets are `frozenset` objects, exactly as in mlxtend.

---

## What's different?

!!! note "Behaviour differences"
    - **Performance**: rusket is significantly faster on medium/large datasets and uses far less memory.
    - **Polars input**: rusket accepts `polars.DataFrame` natively; mlxtend does not.
    - **Sparse DataFrames**: rusket uses the CSR path, which is more memory-efficient than mlxtend for sparse data.
    - **`null_values` / Rust path**: When `null_values=True`, rusket currently falls back gracefully (no error), but the Rust path is not yet used for null-containing DataFrames.

---

## Uninstalling mlxtend

Once you have validated that rusket produces the same results:

```bash
pip uninstall mlxtend
```

rusket has no runtime dependency on mlxtend.
