# Rusket vs MLxtend: Market Basket Analysis at Scale

In this notebook we use a **realistic synthetic retail dataset** — with genuine co-purchase correlations and a pair of competing substitute brands — to show why `rusket` is the fastest association-rule library in Python.

We then use the discovered rules to perform **Assortment Optimization (Cannibalization Detection)** and visualize the results with Plotly.


```python
import os
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

# Directory for embedded chart HTML files
CHARTS_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')), 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)


def save_chart(fig, name: str) -> None:
    """Save a plotly figure as a self-contained HTML file for MkDocs embedding."""
    path = os.path.join(CHARTS_DIR, f"{name}.html")
    fig.write_html(path, include_plotlyjs='cdn', full_html=True)
    print(f"Chart saved → {path}")
```

## 1. Generating a Realistic Correlated Dataset

A **purely random** basket matrix has no signal: every item pair has lift ≈ 1.0, so no rules pass a meaningful confidence threshold. Instead we generate baskets from three customer **segments** with strong co-purchase patterns, plus two competing cola brands that are negatively correlated (lift < 1 — genuine substitutes).


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

    # Substitutes: Cola_A popular (~38%); Cola_B is mostly an alternative
    # Co-occurrence (~6%) is well below independence (~8%) → lift ≈ 0.76
    a_mask = rng.random(n) < 0.38
    b_with_a = a_mask & (rng.random(n) < 0.16)
    b_only = (~a_mask) & (rng.random(n) < 0.24)
    df["Cola_A"] = a_mask
    df["Cola_B"] = b_with_a | b_only

    return df


df = generate_basket_data(n_transactions=20_000)
print(f"Dataset: {df.shape[0]:,} baskets × {df.shape[1]} products")
print(f"Avg basket size: {df.sum(axis=1).mean():.1f} items")
print(f"Cola_A  support: {df['Cola_A'].mean():.3f}")
print(f"Cola_B  support: {df['Cola_B'].mean():.3f}")
print(f"Co-occurrence:   {(df['Cola_A'] & df['Cola_B']).mean():.3f}  (expected if independent: {df['Cola_A'].mean()*df['Cola_B'].mean():.3f})")
df.head(5)
```

## 2. The Benchmark: Rusket vs MLxtend

We mine all product combinations appearing in at least **5%** of baskets. `rusket` provides FP-Growth and Eclat — both written entirely in Rust.


```python
min_support = 0.05

t0 = time.time()
rusket_res = mine(df, min_support=min_support, method="fpgrowth", use_colnames=True)
rusket_time = time.time() - t0
print(f"🚀 Rusket FP-Growth: {rusket_time:.4f}s  ({len(rusket_res):,} itemsets)")

t0 = time.time()
rusket_eclat_res = mine(df, min_support=min_support, method="eclat", use_colnames=True)
rusket_eclat_time = time.time() - t0
print(f"🚀 Rusket Eclat:     {rusket_eclat_time:.4f}s")

t0 = time.time()
mlxtend_res = mlxtend_fpgrowth(df, min_support=min_support, use_colnames=True)
mlxtend_time = time.time() - t0
print(f"🐢 MLxtend FP-Growth:{mlxtend_time:.4f}s  ({len(mlxtend_res):,} itemsets)")
print("-" * 50)
print(f"🏆 Rusket is {mlxtend_time / rusket_time:.1f}× faster than MLxtend!")
```


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

## 3. Generating Cross-Sell Rules

From the frequent itemsets we derive association rules ranked by **lift** — how much more likely the co-purchase is versus random chance. Lift > 1 means genuine affinity; lift < 1 means the products repel each other.


```python
t0 = time.time()
rules = association_rules(rusket_res, num_itemsets=len(df), min_threshold=0.01)
print(f"Generated {len(rules):,} rules in {time.time() - t0:.5f}s")

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

## 4. Assortment Optimization — Substitute Detection

If Product A and Product B are both individually popular but their co-occurrence is lower than chance (lift < 1), they are **substitutes** — customers choose one *instead of* the other. Retailers use this to:

- **Delist redundant SKUs** to reduce warehouse cost
- **Negotiate better terms** with the weaker brand
- **Avoid shelf adjacency** for competing products

`rusket.analytics.find_substitutes` returns product pairs with lift below a threshold:


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

### Product Strategy Quadrant

Every rule plotted by **Confidence** vs **Lift**:

| Quadrant | Region | Action |
|---|---|---|
| ✅ Cross-sell | High confidence, lift > 1 | Bundle, recommend together |
| ⚠️ Substitutes | Low confidence, lift < 1 | Delist one, separate on shelf |


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
