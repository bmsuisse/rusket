# Rusket vs Industry Standards: Market Basket Analysis

In this cookbook, we will use a massive synthetic e-commerce dataset to demonstrate why `rusket` is the fastest association rule mining library in Python, completely crushing the standard `mlxtend` implementation. 

We will then use the discovered rules to perform actionable **Assortment Optimization (Cannibalization Detection)** and visualize the results using **Plotly**.


```python
import time

import numpy as np
import pandas as pd
import plotly.express as px

# Import standard Python baseline
from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth

from rusket import association_rules, mine
from rusket.analytics import find_substitutes
```

## 1. Generating a Massive Dataset
To accurately benchmark, we need a dataset large enough to make standard Python implementations struggle. We will generate 100,000 shopping baskets with 1,000 distinct possible items.


```python
def generate_basket_data(n_transactions=100_000, n_items=1000, density=0.03):
    np.random.seed(42)
    mat = np.random.rand(n_transactions, n_items) < density
    df = pd.DataFrame(mat, columns=[f"Product_{i}" for i in range(n_items)])
    return df


df = generate_basket_data(n_transactions=150_000, n_items=500, density=0.04)
print(f"Dataset shape: {df.shape}")
df.head(3)
```

    Dataset shape: (150000, 500)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product_0</th>
      <th>Product_1</th>
      <th>Product_2</th>
      <th>Product_3</th>
      <th>Product_4</th>
      <th>Product_5</th>
      <th>Product_6</th>
      <th>Product_7</th>
      <th>Product_8</th>
      <th>Product_9</th>
      <th>...</th>
      <th>Product_490</th>
      <th>Product_491</th>
      <th>Product_492</th>
      <th>Product_493</th>
      <th>Product_494</th>
      <th>Product_495</th>
      <th>Product_496</th>
      <th>Product_497</th>
      <th>Product_498</th>
      <th>Product_499</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 500 columns</p>
</div>



## 2. The Benchmark: Rusket vs MLxtend
We will mine all frequent product combinations that appear in at least 3% of baskets. `rusket` provides both FP-Growth and Eclat algorithms written in pure Rust.


```python
min_support = 0.03

# --- 1. Rusket FP-Growth ---
t0 = time.time()
rusket_res = mine(df, min_support=min_support, method="fpgrowth", use_colnames=True)
rusket_time = time.time() - t0
print(f"🚀 Rusket FP-Growth: {rusket_time:.4f}s (Found {len(rusket_res)} itemsets)")

# --- 2. Rusket ECLAT ---
t0 = time.time()
rusket_eclat_res = mine(df, min_support=min_support, method="eclat", use_colnames=True)
rusket_eclat_time = time.time() - t0
print(f"🚀 Rusket Eclat:     {rusket_eclat_time:.4f}s")

# --- 3. MLxtend FP-Growth ---
t0 = time.time()
mlxtend_res = mlxtend_fpgrowth(df, min_support=min_support, use_colnames=True)
mlxtend_time = time.time() - t0
print(f"🐢 MLxtend FP-Growth:{mlxtend_time:.4f}s (Found {len(mlxtend_res)} itemsets)")

print("-" * 40)
print(f"🏆 Rusket is {mlxtend_time / rusket_time:.1f}x faster than MLxtend!")
```

    🚀 Rusket FP-Growth: 0.6745s (Found 500 itemsets)


    🚀 Rusket Eclat:     0.3275s


    🐢 MLxtend FP-Growth:9.4472s (Found 500 itemsets)
    ----------------------------------------
    🏆 Rusket is 14.0x faster than MLxtend!


Let's visualize this complete destruction in performance speeds.


```python
fig = px.bar(
    x=["MLxtend (Python)", "Rusket Eclat (Rust)", "Rusket FP-Growth (Rust)"],
    y=[mlxtend_time, rusket_eclat_time, rusket_time],
    title="Execution Time (Lower is Better)",
    labels={"x": "Implementation", "y": "Seconds"},
    color=["baseline", "optimized", "optimized"],
    color_discrete_map={"baseline": "#EF553B", "optimized": "#00CC96"},
)
fig.show()
```



## 3. High-Speed Association Rules
Now that we have our frequent combinations blazing fast, we can generate the Association Rules ("If they buy A, they will buy B").


```python
t0 = time.time()
rules = association_rules(rusket_res, num_itemsets=len(df), min_threshold=0.1)
print(f"Generated {len(rules)} rules in {time.time() - t0:.4f}s")

rules.sort_values("lift", ascending=False).head(5)
```

    Generated 0 rules in 0.0033s





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>representativity</th>
      <th>leverage</th>
      <th>conviction</th>
      <th>zhangs_metric</th>
      <th>jaccard</th>
      <th>certainty</th>
      <th>kulczynski</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



## 4. Assortment Optimization (Cannibalization Detection)

Most tutorials stop at finding items bought together. But what about items that **prevent** each other from being bought? 

If Product A and Product B are both highly popular individually (high support), but they are *never* bought together (Lift < 1.0), they are **Substitutes**. They cannibalize each other's sales. Retailers use this to delist redundant inventory and save warehouse costs.

Rusket provides out-of-the-box business analytics to detect this:


```python
# Find products that cannibalize each other (Substitutes)
substitutes = find_substitutes(rules, max_lift=0.9)

print(f"Found {len(substitutes)} cannibalizing product pairs.")
substitutes.head(5)
```

    Found 0 cannibalizing product pairs.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>representativity</th>
      <th>leverage</th>
      <th>conviction</th>
      <th>zhangs_metric</th>
      <th>jaccard</th>
      <th>certainty</th>
      <th>kulczynski</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



### Visualizing Cannibalization
Let's plot Confidence vs Lift. 
- **Top Right** (High Lift, High Conf): Perfect Cross-Sell candidates.
- **Bottom Left** (Low Lift, Low Conf): Substitute / Cannibalizing products to delist.


```python
fig = px.scatter(
    rules,
    x="confidence",
    y="lift",
    size="support",
    color="lift",
    hover_data=["antecedents", "consequents"],
    color_continuous_scale="RdYlGn",  # Red is bad (substitutes), Green is good (cross-sells)
    title="Product Strategy: Cross-Sells vs Substitutes (Cannibalization)",
)

# Add a reference line for Lift = 1.0 (Independence)
fig.add_hline(y=1.0, line_dash="dash", line_color="white", annotation_text="Independent")
fig.show()
```


