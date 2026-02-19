# Polars Support

rusket accepts **`polars.DataFrame`** natively alongside pandas, via the Arrow-backed zero-copy path.

## Installation

Install rusket with the Polars extra:

=== "pip"

    ```bash
    pip install "rusket[polars]"
    ```

=== "uv"

    ```bash
    uv add "rusket[polars]"
    ```

This pins `polars>=0.20`. If you already have Polars installed, you can also just `pip install rusket`.

---

## Usage

The `fpgrowth` function detects Polars DataFrames automatically — no extra parameters needed:

```python
import polars as pl
from rusket import fpgrowth, association_rules

# Build a Polars one-hot DataFrame
df = pl.DataFrame({
    "milk":  [True, True,  False, True],
    "bread": [True, False, True,  True],
    "eggs":  [False, True, True,  True],
})

# Mine frequent itemsets — identical call as with pandas
freq = fpgrowth(df, min_support=0.5, use_colnames=True)
print(freq)
#    support          itemsets
# 0     0.75          (milk,)
# 1     0.75         (bread,)
# ...

# association_rules always returns a pandas DataFrame
rules = association_rules(freq, num_itemsets=len(df), metric="lift", min_threshold=1.0)
print(rules)
```

!!! note "Return type"
    `fpgrowth` always returns a **pandas DataFrame**, regardless of input type.  
    `association_rules` also returns a **pandas DataFrame**.

---

## How it works

The Polars path uses `polars.DataFrame.to_numpy()` which returns an Arrow-backed NumPy buffer — **zero-copy for numeric dtypes**.

```
Polars DataFrame
    │
    ▼  df.to_numpy()  (zero-copy for bool/int dtypes)
numpy uint8 array
    │
    ▼  fpgrowth_from_dense()  (Rust, PyO3 ReadonlyArray2<u8>)
Rust FP-Tree mining
    │
    ▼
pandas DataFrame  [support, itemsets]
```

No intermediate Python object creation occurs between the Polars input and the Rust mining step.

---

## Supported dtypes

| Polars dtype | Supported |
|---|---|
| `Boolean` | ✅ |
| `Int8 / Int16 / Int32 / Int64` | ✅ (0/1 values) |
| `UInt8 / UInt16 / UInt32 / UInt64` | ✅ (0/1 values) |
| `Float32 / Float64` | ⚠️ (0.0/1.0 values, cast to uint8) |
| Categorical / String | ❌ (pre-encode with `get_dummies`) |

!!! tip "Lazy frames"
    Pass `.collect()` before calling `fpgrowth` if you have a `LazyFrame`:
    ```python
    freq = fpgrowth(lazy_df.collect(), min_support=0.3, use_colnames=True)
    ```
