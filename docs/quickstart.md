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

`FPGrowth` expects a **one-hot encoded** DataFrame where rows are transactions and columns are products.

```python
import pandas as pd
from rusket import FPGrowth

orders = pd.DataFrame({
    "receipt_id": [1001, 1001, 1001, 1002, 1002, 1003, 1003, 1004],
    "product":    ["milk", "bread", "butter",
                   "milk", "eggs",
                   "bread", "butter",
                   "milk", "bread", "eggs", "coffee"],
})

model = FPGrowth.from_transactions(orders, transaction_col="receipt_id", item_col="product", min_support=0.4)
```

---

## Step 2 — Mine frequent product combinations

```python
freq = model.mine(use_colnames=True)
print(freq.sort_values("support", ascending=False))
```

!!! tip
    `FPGrowth` picks `Eclat` for sparse data (density < 0.15) and `FPGrowth` for dense data.

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
from rusket import FPGrowth

csr = sp.csr_matrix(
    (np.ones(len(receipt_ids), dtype=np.int8), (receipt_ids, sku_indices)),
    shape=(n_receipts, n_skus),
)
freq = FPGrowth(csr).mine(min_support=0.001, column_names=sku_names)
```

---

## What's Next?

- [API Reference](api-reference.md)
- [Polars Support](polars.md)
- [Recommender Workflows](recommender.md)
## Saving and Serving Models

`rusket` models use a unified `BaseModel` that provides `.save()` and `.load()` functionality. You can also export trained models to a Vector Database (like LanceDB, FAISS, or Qdrant) for fast, real-time serving in production.

```python
import rusket
from pathlib import Path

# 1. Train the model
model = rusket.ALS(factors=32).fit(interactions)

# 2. Save your trained model to disk
model.save("my_als_model.pkl")

# 3. Load it back using the generic loader
loaded_model = rusket.load_model("my_als_model.pkl")

# 4. Export the embeddings for a Vector Database
items_df = rusket.export_item_factors(
    loaded_model, 
    normalize=True,     # Best for Cosine Similarity search
    format="pandas"
)

# 5. Serve it in real-time (Example using LanceDB)
import lancedb

# Create a local vector database
db = lancedb.connect("./lancedb_store")
table = db.create_table("items", data=items_df)

# Query the table with a specific user's latent factors
user_emb = loaded_model.user_factors[0]

# Normalize user vector identically to items
user_emb = user_emb / max(np.linalg.norm(user_emb), 1e-9)

# Retrieve top 5 item recommendations for this user!
results = table.search(user_emb).limit(5).to_pandas()
print(results)
```
