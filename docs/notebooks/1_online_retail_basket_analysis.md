# Rusket vs MLxtend: Market Basket Analysis at Scale

In this notebook we use a **realistic synthetic retail dataset** — with genuine co-purchase correlations and a pair of competing substitute brands — to show why `rusket` is the fastest association-rule library in Python.

We then use the discovered rules to perform **Assortment Optimization (Cannibalization Detection)** and visualize the results with Plotly.


```python
import os
import pathlib
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth

from rusket import association_rules, mine
from rusket.analytics import find_substitutes

# Crisp dark theme for all charts
pio.templates.default = "plotly_dark"

# Nicer float display in DataFrames
pd.options.display.float_format = "{:.3f}".format

# Charts saved as self-contained HTML for MkDocs embedding
CHARTS_DIR = pathlib.Path("docs/notebooks/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

def save_chart(fig, name: str) -> None:
    path = CHARTS_DIR / f"{name}.html"
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    print(f"Chart saved → {path}")

```

## 1. Generating a Realistic Correlated Dataset

A **purely random** basket matrix (the typical benchmark approach) has no real signal: every item pair will have lift ≈ 1.0, and no rules will pass a meaningful confidence threshold. Instead we generate baskets from three customer **segments** with strong co-purchase behaviour, plus two competing cola brands that are negatively correlated (lift < 1 — genuine substitutes).


```python
def generate_basket_data(n_transactions: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """
    Segment-based basket generator with realistic co-purchase correlations.

    Three segments create strong *positive* correlations (high lift).
    Two competing cola brands are *negatively* correlated (lift ≈ 0.76, substitutes).
    """
    rng = np.random.default_rng(seed)
    n = n_transactions

    cols = [
        # Tech accessories cluster
        "Mouse", "Keyboard", "USB_Hub", "Webcam",
        # Barista / coffee cluster
        "Espresso_Beans", "Milk_Frother", "Travel_Mug",
        # Home-office cluster
        "Notebook", "Gel_Pen", "Highlighter",
        # Competing brands — negative correlation
        "Cola_A", "Cola_B",
    ]
    df = pd.DataFrame(False, index=range(n), columns=cols)

    # Tech buyers (40%) cluster
    seg = rng.random(n) < 0.40
    for p in ["Mouse", "Keyboard", "USB_Hub", "Webcam"]:
        df.loc[seg, p] = rng.random(seg.sum()) < 0.75

    # Coffee buyers (35%) cluster
    seg = rng.random(n) < 0.35
    for p in ["Espresso_Beans", "Milk_Frother", "Travel_Mug"]:
        df.loc[seg, p] = rng.random(seg.sum()) < 0.78

    # Home-office buyers (45%) cluster
    seg = rng.random(n) < 0.45
    for p in ["Notebook", "Gel_Pen", "Highlighter"]:
        df.loc[seg, p] = rng.random(seg.sum()) < 0.72

    # Substitutes: Cola_A is popular (~38%)
    # Cola_B can appear with A at only 16% probability → co-occurrence ~6%
    # Independence would predict ~8% → lift ≈ 0.76
    a_mask = rng.random(n) < 0.38
    b_with_a = a_mask & (rng.random(n) < 0.16)
    b_only = (~a_mask) & (rng.random(n) < 0.24)
    df["Cola_A"] = a_mask
    df["Cola_B"] = b_with_a | b_only

    return df


df = generate_basket_data(n_transactions=20_000)
print(f"Dataset: {df.shape[0]:,} baskets × {df.shape[1]} products")
print(f"Avg basket size: {df.sum(axis=1).mean():.1f} items")
print(f"Cola_A support: {df['Cola_A'].mean():.3f}")
print(f"Cola_B support: {df['Cola_B'].mean():.3f}")
print(f"Cola_A & Cola_B co-occurrence: {(df['Cola_A'] & df['Cola_B']).mean():.3f}")
df.head(5)
```

    Dataset: 20,000 baskets × 12 products
    Avg basket size: 3.6 items
    Cola_A support: 0.380
    Cola_B support: 0.212
    Cola_A & Cola_B co-occurrence: 0.061





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mouse</th>
      <th>Keyboard</th>
      <th>USB_Hub</th>
      <th>Webcam</th>
      <th>Espresso_Beans</th>
      <th>Milk_Frother</th>
      <th>Travel_Mug</th>
      <th>Notebook</th>
      <th>Gel_Pen</th>
      <th>Highlighter</th>
      <th>Cola_A</th>
      <th>Cola_B</th>
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
      <td>True</td>
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
      <td>True</td>
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
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



## 2. The Benchmark: Rusket vs MLxtend

We mine all product combinations appearing in at least 5% of baskets. `rusket` provides FP-Growth and Eclat — both written entirely in Rust.


```python
min_support = 0.05

# --- Rusket FP-Growth ---
t0 = time.time()
rusket_res = mine(df, min_support=min_support, method="fpgrowth", use_colnames=True)
rusket_time = time.time() - t0
print(f"🚀 Rusket FP-Growth: {rusket_time:.4f}s  ({len(rusket_res):,} itemsets)")

# --- Rusket Eclat ---
t0 = time.time()
rusket_eclat_res = mine(df, min_support=min_support, method="eclat", use_colnames=True)
rusket_eclat_time = time.time() - t0
print(f"🚀 Rusket Eclat:     {rusket_eclat_time:.4f}s")

# --- MLxtend FP-Growth ---
t0 = time.time()
mlxtend_res = mlxtend_fpgrowth(df, min_support=min_support, use_colnames=True)
mlxtend_time = time.time() - t0
print(f"🐢 MLxtend FP-Growth:{mlxtend_time:.4f}s  ({len(mlxtend_res):,} itemsets)")
print("-" * 50)
print(f"🏆 Rusket is {mlxtend_time / rusket_time:.1f}× faster than MLxtend!")
```

    🚀 Rusket FP-Growth: 0.0129s  (226 itemsets)
    🚀 Rusket Eclat:     0.0027s
    🐢 MLxtend FP-Growth:0.0421s  (226 itemsets)
    --------------------------------------------------
    🏆 Rusket is 3.3× faster than MLxtend!



```python
fig = px.bar(
    x=["MLxtend (Python)", "Rusket Eclat (Rust)", "Rusket FP-Growth (Rust)"],
    y=[mlxtend_time, rusket_eclat_time, rusket_time],
    title="⏱ Execution Time — Lower is Better",
    labels={"x": "Implementation", "y": "Time (seconds)"},
    color=["baseline", "optimized", "optimized"],
    color_discrete_map={"baseline": "#EF553B", "optimized": "#00CC96"},
    text_auto=".2f",
)
fig.update_traces(textfont_size=15)
fig.update_layout(showlegend=False, title_font_size=20)
save_chart(fig, "benchmark")
fig.show()
```

    Chart saved → docs/notebooks/charts/benchmark.html

<iframe src="charts/benchmark.html" width="100%" height="480" style="border:none;"></iframe>

---

## 3. Generating Cross-Sell Rules

From the frequent itemsets we generate association rules — "If a customer buys A, they will also buy B" — ranked by **lift** (how much more likely the co-purchase is versus random chance). Lift > 1 means a genuine affinity; lift < 1 means the products repel each other.


```python
t0 = time.time()
rules = association_rules(rusket_res, num_itemsets=len(df), min_threshold=0.01)
print(f"Generated {len(rules):,} rules in {time.time() - t0:.5f}s")

# Top cross-sell rules by lift
(
    rules[["antecedents", "consequents", "support", "confidence", "lift"]]
    .sort_values("lift", ascending=False)
    .head(8)
    .assign(
        support=lambda d: d["support"].round(3),
        confidence=lambda d: d["confidence"].round(3),
        lift=lambda d: d["lift"].round(2),
    )
    .reset_index(drop=True)
)
```

    Generated 1,412 rules in 0.00574s





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedents</th>
      <th>consequents</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(Gel_Pen, Espresso_Beans,)</td>
      <td>(Notebook, Travel_Mug,)</td>
      <td>0.051</td>
      <td>0.588</td>
      <td>6.670</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(Notebook, Travel_Mug,)</td>
      <td>(Gel_Pen, Espresso_Beans,)</td>
      <td>0.051</td>
      <td>0.583</td>
      <td>6.670</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Notebook, Espresso_Beans,)</td>
      <td>(Gel_Pen, Travel_Mug,)</td>
      <td>0.051</td>
      <td>0.586</td>
      <td>6.570</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(Gel_Pen, Travel_Mug,)</td>
      <td>(Notebook, Espresso_Beans,)</td>
      <td>0.051</td>
      <td>0.576</td>
      <td>6.570</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(Milk_Frother, Gel_Pen,)</td>
      <td>(Highlighter, Travel_Mug,)</td>
      <td>0.050</td>
      <td>0.571</td>
      <td>6.560</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(Highlighter, Travel_Mug,)</td>
      <td>(Milk_Frother, Gel_Pen,)</td>
      <td>0.050</td>
      <td>0.580</td>
      <td>6.560</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(Milk_Frother, Gel_Pen,)</td>
      <td>(Notebook, Travel_Mug,)</td>
      <td>0.050</td>
      <td>0.571</td>
      <td>6.480</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(Notebook, Travel_Mug,)</td>
      <td>(Milk_Frother, Gel_Pen,)</td>
      <td>0.050</td>
      <td>0.573</td>
      <td>6.480</td>
    </tr>
  </tbody>
</table>
</div>



## 4. Assortment Optimization — Substitute Detection

Most tutorials stop at finding items bought *together*. But what about items that **prevent** each other from being bought?

If Product A and Product B are both individually popular but their co-occurrence is lower than random chance (lift < 1), they are **substitutes** — customers choose one *instead of* the other. Retailers use this to:

- **Delist redundant SKUs** (reduce warehouse cost)
- **Negotiate better terms** with the weaker brand
- **Optimise shelf-space** by not displaying competing items side-by-side

`rusket` provides `find_substitutes` out of the box:


```python
substitutes = find_substitutes(rules, max_lift=0.9)
print(f"Found {len(substitutes)} cannibalizing product pair(s).")

(
    substitutes[["antecedents", "consequents", "support", "confidence", "lift"]]
    .assign(
        support=lambda d: d["support"].round(3),
        confidence=lambda d: d["confidence"].round(3),
        lift=lambda d: d["lift"].round(3),
    )
    .reset_index(drop=True)
)
```

    Found 2 cannibalizing product pair(s).





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedents</th>
      <th>consequents</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(Cola_B,)</td>
      <td>(Cola_A,)</td>
      <td>0.061</td>
      <td>0.290</td>
      <td>0.762</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(Cola_A,)</td>
      <td>(Cola_B,)</td>
      <td>0.061</td>
      <td>0.162</td>
      <td>0.762</td>
    </tr>
  </tbody>
</table>
</div>



### Visualizing the Product Strategy Quadrant

Plot every rule as a point: **Confidence** (x-axis) vs **Lift** (y-axis).

- **Top-right** (high confidence, high lift): perfect cross-sell candidates
- **Below the dashed line** (lift < 1): substitutes / cannibalizing products


```python
# Plot only singleton→singleton rules for readability
singleton_rules = rules[
    (rules["antecedents"].apply(len) == 1)
    & (rules["consequents"].apply(len) == 1)
].copy()

singleton_rules["rule_label"] = (
    singleton_rules["antecedents"].apply(lambda x: next(iter(x)))
    + " → "
    + singleton_rules["consequents"].apply(lambda x: next(iter(x)))
)

fig = px.scatter(
    singleton_rules,
    x="confidence",
    y="lift",
    size="support",
    color="lift",
    hover_name="rule_label",
    hover_data={"confidence": ":.3f", "lift": ":.3f", "support": ":.3f"},
    color_continuous_scale="RdYlGn",
    title="📊 Product Strategy: Cross-Sells vs Substitutes",
    labels={"confidence": "Confidence", "lift": "Lift"},
)
fig.add_hline(
    y=1.0, line_dash="dash", line_color="white",
    annotation_text="Lift = 1.0  (independent)",
    annotation_position="top left",
)
fig.add_annotation(
    x=0.85, y=singleton_rules["lift"].max() * 0.92,
    text="✅ Cross-sell", showarrow=False,
    font=dict(color="#00CC96", size=14),
)
fig.add_annotation(
    x=0.18, y=0.60,
    text="⚠️ Substitutes", showarrow=False,
    font=dict(color="#EF553B", size=14),
)
fig.update_layout(title_font_size=20)
save_chart(fig, "product_strategy")
fig.show()
```

    Chart saved → docs/notebooks/charts/product_strategy.html




<iframe src="charts/product_strategy.html" width="100%" height="520" style="border:none;"></iframe>

