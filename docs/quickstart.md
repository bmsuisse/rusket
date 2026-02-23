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



---

## Business Scenario — Supermarket Cross-Selling

## Step 1 — Prepare your data

`AutoMiner` expects a **one-hot encoded** DataFrame where rows are transactions and columns are products.

```python
import pandas as pd
from rusket import AutoMiner

orders = pd.DataFrame({
    "receipt_id": [1001, 1001, 1001, 1002, 1002, 1003, 1003, 1004],
    "product":    ["milk", "bread", "butter",
                   "milk", "eggs",
                   "bread", "butter",
                   "milk", "bread", "eggs", "coffee"],
})

model = AutoMiner.from_transactions(orders, transaction_col="receipt_id", item_col="product", min_support=0.4)
```

---

## Step 2 — Mine frequent product combinations

```python
freq = model.mine(use_colnames=True)
print(freq.sort_values("support", ascending=False))
```

!!! tip
    `AutoMiner` picks `Eclat` for sparse data (density < 0.15) and `FPGrowth` for dense data.

---

## Step 3 — Generate "Frequently Bought Together" rules

```python
rules = model.association_rules(metric="confidence", min_threshold=0.6)
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
```

---

## Recommendations

```python
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
rules = miner.association_rules()
```

!!! tip
    Peak Python memory = one chunk. Rust holds the per-transaction item lists. The final mining step passes CSR arrays directly — zero copies.

### Direct CSR path

```python
from scipy import sparse as sp
from rusket import AutoMiner

csr = sp.csr_matrix(
    (np.ones(len(receipt_ids), dtype=np.int8), (receipt_ids, sku_indices)),
    shape=(n_receipts, n_skus),
)
freq = AutoMiner(csr).mine(min_support=0.001, column_names=sku_names)
```

---

## What's Next?

- [API Reference](api-reference.md)
- [Polars Support](polars.md)
- [Recommender Workflows](recommender.md)
