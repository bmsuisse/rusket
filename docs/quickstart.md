# Quick Start

## Installation

=== "pip"

    ```bash
    pip install rusket
    ```

=== "uv"

    ```bash
    uv add rusket
    ```

=== "conda"

    ```bash
    pip install rusket  # rusket is not on conda-forge yet
    ```

To also enable **Polars** support:

=== "pip"

    ```bash
    pip install "rusket[polars]"
    ```

=== "uv"

    ```bash
    uv add "rusket[polars]"
    ```

!!! tip "Coming from mlxtend?"
    rusket is a **drop-in replacement**. In most cases you only need to change your import:
    ```python
    # Before
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    # After
    from rusket import fpgrowth, association_rules
    ```
    See the full [Migration Guide](migration.md) for details.

---

## Step 1 — Prepare your data

`fpgrowth` expects a **one-hot encoded** DataFrame of boolean or 0/1 integer values where rows are transactions and columns are items.

```python
import pandas as pd

dataset = [
    ["milk", "bread"],
    ["milk", "eggs"],
    ["bread", "eggs"],
    ["milk", "bread", "eggs"],
]

# Build a one-hot DataFrame
items = ["milk", "bread", "eggs"]
df = pd.DataFrame(
    [[item in tx for item in items] for tx in dataset],
    columns=items,
    dtype=bool,
)
print(df)
#    milk  bread   eggs
# 0  True   True  False
# 1  True  False   True
# 2 False   True   True
# 3  True   True   True
```

---

## Step 2 — Mine frequent itemsets

```python
from rusket import fpgrowth

freq = fpgrowth(df, min_support=0.5, use_colnames=True)
print(freq)
#    support          itemsets
# 0     0.75          (milk,)
# 1     0.75         (bread,)
# 2     0.75          (eggs,)
# 3     0.50   (milk, bread,)
# 4     0.50    (milk, eggs,)
# 5     0.50   (bread, eggs,)
# 6     0.25  (milk, bread, eggs,)
```

---

## Step 2b — Or use Eclat

`eclat` uses vertical bitset mining — same API, same results. It can be faster on certain data shapes.

```python
from rusket import eclat

freq = eclat(df, min_support=0.5, use_colnames=True)
print(freq)  # identical output to fpgrowth
```

!!! tip "When to use which?"
    - **FP-Growth** — good default for all data shapes, excels on dense data
    - **Eclat** — competitive alternative, especially on sparse retail basket data

---

## Step 3 — Generate association rules

```python
from rusket import association_rules

rules = association_rules(
    freq,
    num_itemsets=len(df),
    metric="confidence",
    min_threshold=0.6,
)
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
```

!!! note "num_itemsets"
    Pass the **total transaction count** (`len(df)`) so that support-based metrics are computed correctly.

---

## Billion-Scale Streaming

For datasets with hundreds of millions or billions of rows, use ``FPMiner`` to
stream data into Rust **one chunk at a time** — without ever holding the full
dataset in memory.

```python
from rusket import FPMiner

miner = FPMiner(n_items=500_000)  # total distinct items

# Feed chunks — e.g. from a Parquet file, Spark, or a database cursor
for chunk in pd.read_parquet("orders.parquet", chunksize=10_000_000):
    txn = chunk["txn_id"].to_numpy(dtype="int64")   # arbitrary int IDs
    item = chunk["item_idx"].to_numpy(dtype="int32") # 0-based item index
    miner.add_chunk(txn, item)

# Mine — all data in Rust, output is a normal pandas DataFrame
freq = miner.mine(min_support=0.001, max_len=3, use_colnames=False)
rules = association_rules(freq, num_itemsets=miner.n_transactions)
```

!!! tip "Peak memory"
    Peak Python memory = one chunk (typically 1–2 GB).
    Rust holds the per-transaction item lists (~5 GB for 200M transactions).
    The final mining step passes CSR arrays directly — zero copies.

### Direct CSR path

If you already have integer arrays (e.g. from a database query), skip
``from_transactions`` entirely:

```python
from scipy import sparse as sp
from rusket import fpgrowth

csr = sp.csr_matrix(
    (np.ones(len(txn_ids), dtype=np.int8), (txn_ids, item_ids)),
    shape=(n_transactions, n_items),
)
freq = fpgrowth(csr, min_support=0.001, column_names=item_names)
```

---

## What's Next?

- [Migration from mlxtend](migration.md) — side-by-side comparison
- [API Reference](api-reference.md) — all parameters and metrics explained
- [Polars Support](polars.md) — zero-copy Arrow path
- [Benchmarks](benchmarks.md) — performance vs mlxtend
