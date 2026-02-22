# Quick Start

Install rusket and run your first Market Basket Analysis in minutes.

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
    from rusket import mine, association_rules
    ```
    See the full [Migration Guide](migration.md) for details.

---

## Business Scenario — Supermarket Cross-Selling

## Step 1 — Prepare your data

`mine` expects a **one-hot encoded** DataFrame where rows are transactions and columns are products.

```python
import pandas as pd
from rusket import from_transactions

orders = pd.DataFrame({
    "receipt_id": [1001, 1001, 1001, 1002, 1002, 1003, 1003, 1004],
    "product":    ["milk", "bread", "butter",
                   "milk", "eggs",
                   "bread", "butter",
                   "milk", "bread", "eggs", "coffee"],
})

basket = from_transactions(orders, transaction_col="receipt_id", item_col="product")
```

---

## Step 2 — Mine frequent product combinations

```python
from rusket import mine

freq = mine(basket, min_support=0.4, use_colnames=True)
print(freq.sort_values("support", ascending=False))
```

!!! tip
    `mine(method="auto")` picks `eclat` for sparse data (density < 0.15) and `fpgrowth` for dense data.

---

## Step 3 — Generate "Frequently Bought Together" rules

```python
from rusket import association_rules

rules = association_rules(
    freq,
    num_itemsets=len(basket),
    metric="confidence",
    min_threshold=0.6,
)
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
```

!!! note
    Pass the **total transaction count** (`len(basket)`) as `num_itemsets` so that support-based metrics are computed correctly.

---

## OOP API — Fluent Pipeline

```python
from rusket import AutoMiner

model = AutoMiner.from_transactions(orders, transaction_col="receipt_id", item_col="product", min_support=0.4)
freq  = model.mine(use_colnames=True)
rules = model.association_rules(metric="lift", min_threshold=1.0)

basket_contents = ["milk", "bread"]
suggestions = model.recommend_items(basket_contents, n=3)
```

---

## Billion-Scale Streaming

```python
from rusket import FPMiner

miner = FPMiner(n_items=500_000)

for chunk in pd.read_parquet("sales_fact.parquet", chunksize=10_000_000):
    txn  = chunk["receipt_id"].to_numpy(dtype="int64")
    item = chunk["product_idx"].to_numpy(dtype="int32")
    miner.add_chunk(txn, item)

freq  = miner.mine(min_support=0.001, max_len=3)
rules = association_rules(freq, num_itemsets=miner.n_transactions)
```

!!! tip
    Peak Python memory = one chunk. Rust holds the per-transaction item lists. The final mining step passes CSR arrays directly — zero copies.

### Direct CSR path

```python
from scipy import sparse as sp
from rusket import mine

csr = sp.csr_matrix(
    (np.ones(len(receipt_ids), dtype=np.int8), (receipt_ids, sku_indices)),
    shape=(n_receipts, n_skus),
)
freq = mine(csr, min_support=0.001, column_names=sku_names)
```

---

## What's Next?

- [Migration from mlxtend](migration.md)
- [API Reference](api-reference.md)
- [Polars Support](polars.md)
- [Recommender Workflows](recommender.md)
