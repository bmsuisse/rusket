# Market Basket Analysis Cookbook

Welcome to the `rusket` cookbook! This guide provides practical examples for performing market basket analysis — finding frequent itemsets and generating recommendation rules efficiently.

---

## Setup

```python
import numpy as np
import pandas as pd
import polars as pl
import plotly.express as px
import pyarrow.compute as pc
from rusket import fpgrowth, association_rules
```

---

## 1. Synthetic Dataset Generation

Let's generate a synthetic retail dataset — simulating a supermarket where customers buy various categories of items.

```python
np.random.seed(42)

items = [
    "Milk", "Bread", "Butter", "Eggs", "Cheese", "Yogurt",
    "Coffee", "Tea", "Sugar", "Apples", "Bananas", "Oranges",
    "Chicken", "Beef", "Fish", "Rice", "Pasta", "Tomato Sauce",
    "Onions", "Garlic",
]

n_transactions = 10_000
n_items = len(items)

# Simulate different purchase frequencies (power-law distribution)
probabilities = np.power(np.arange(1, n_items + 1, dtype=float), -0.7)
probabilities /= probabilities.max()
probabilities = np.clip(probabilities * 0.3, 0.01, 0.8)

data = np.random.rand(n_transactions, n_items) < probabilities
df = pd.DataFrame(data, columns=items)

print(f"Dataset shape: {df.shape}")
df.head()
```

---

## 2. Frequent Pattern Mining

Extract frequent itemsets using the blazing-fast `fpgrowth` algorithm.
`min_support=0.05` means an itemset must appear in at least 5% of all transactions.

```python
fi = fpgrowth(df, min_support=0.05, use_colnames=True)

print(f"Found {len(fi)} frequent itemsets.")
fi.sort_values(by="support", ascending=False).head(10)
```

### Visualizing Frequent Itemsets

Plot the top 20 most frequent itemsets to understand what items are bought together most often.

```python
top_fi = fi.sort_values(by="support", ascending=False).head(20).copy()
top_fi["itemsets_str"] = top_fi["itemsets"].apply(lambda x: " + ".join(list(x)))

fig = px.bar(
    top_fi,
    x="support",
    y="itemsets_str",
    orientation="h",
    title="Top 20 Frequent Itemsets by Support",
    labels={"support": "Support", "itemsets_str": "Itemset"},
    color="support",
    color_continuous_scale="Viridis",
)
fig.update_layout(yaxis={"categoryorder": "total ascending"})
fig.show()
```

---

## 3. Generating Association Rules

Generate association rules from frequent itemsets using the `confidence` metric.

```python
rules = association_rules(fi, num_itemsets=len(df), min_threshold=0.3)

print(f"Generated {len(rules)} association rules.")
rules.sort_values(by="lift", ascending=False).head()
```

### Filtering Rules

Filter for strong rules with high confidence **and** high lift.
High lift (> 1) indicates items are positively correlated.

```python
strong_rules = rules[(rules["confidence"] > 0.4) & (rules["lift"] > 1.2)]
strong_rules = strong_rules.sort_values(by="lift", ascending=False)
strong_rules.head(10)
```

### Visualizing Association Rules

A scatter plot of Support vs. Confidence helps identify the most valuable rules. Color represents `lift`.

```python
fig = px.scatter(
    rules,
    x="support",
    y="confidence",
    color="lift",
    hover_data=["antecedents", "consequents"],
    title="Association Rules: Support vs Confidence",
    color_continuous_scale="Plasma",
)
fig.show()
```

---

## 4. Polars Integration

`rusket` works natively with Polars without requiring expensive conversions.

```python
df_pl = pl.from_pandas(df)
fi_pl = fpgrowth(df_pl, min_support=0.05, use_colnames=True)

# Generate rules directly from a Polars DataFrame
rules_pl = association_rules(fi_pl, num_itemsets=df_pl.height, min_threshold=0.3)
rules_pl.head(5)
```

---

## 5. Working with PyArrow Outputs

`rusket` returns itemsets as zero-copy **PyArrow `ListArray`** structures. This eliminates Python object overhead and allows you to process millions of rules with minimal memory.

!!! note "PyArrow dtype"
    The `itemsets` column uses `pd.ArrowDtype(pa.list_(pa.string()))`. Standard Python `set` equality won't work directly — use PyArrow compute functions or `.apply(set)` on filtered subsets.

### Querying Itemsets with PyArrow Compute

```python
import pyarrow.compute as pc

# Find all itemsets where the first element is 'Milk'
contains_milk = pc.list_element(fi["itemsets"].array, 0) == "Milk"
fi[contains_milk].head()
```

### Converting to Python Sets

Only do this on small filtered subsets to avoid materializing Python objects for the full result.

```python
top_10 = fi.head(10).copy()
top_10["python_sets"] = top_10["itemsets"].apply(set)

print("Zero-Copy PyArrow Array Dtype:", fi["itemsets"].dtype)
top_10.head()
```
