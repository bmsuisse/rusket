# LightGCN for E-Commerce Product Recommendations

**Business problem:** An online retailer has millions of past orders but no explicit ratings. They want to:
1. **Personalise the homepage** — show each visitor the items they are most likely to buy next.
2. **Identify cross-sell opportunities** — for any product page, surface the top complementary items.
3. **Prioritise marketing spend** — score every user × campaign-item pair to find the highest-propensity audience.

**Why LightGCN?** Unlike ALS/BPR (which treat items as independent), LightGCN propagates signals across the *purchase graph*: if User A and User B both bought Candles and Mugs, LightGCN will also surface Teapots to User A — even if User A has never interacted with Teapots — because User B's graph neighbourhood connects them.

> **Dataset:** [UCI Online Retail II](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) — ~500k real UK gift/homeware transactions.


```python
import os
import time
import urllib.request

import numpy as np
import pandas as pd

from rusket import LightGCN
```

## 1. Load & Clean Transactional Data

Real retail data is messy: cancellations (InvoiceNo starting with `C`), negative quantities, and missing customer IDs all need to be removed before modelling.


```python
# ── Download dataset ─────────────────────────────────────────────────────────
DATA_PATH = "online_retail_II.xlsx"
if not os.path.exists(DATA_PATH):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
    print("Downloading Online Retail II dataset…")
    urllib.request.urlretrieve(url, DATA_PATH)

raw = pd.read_excel(DATA_PATH, sheet_name="Year 2010-2011", engine="openpyxl")
print(f"Raw rows: {len(raw):,}")

# ── Clean ─────────────────────────────────────────────────────────────────────
df = (
    raw.dropna(subset=["Customer ID", "StockCode", "Description"])
    .query("Quantity > 0 and Price > 0")
    .query("~Invoice.str.startswith('C')")
    .rename(columns={"Customer ID": "user_id", "StockCode": "item_id", "Description": "item_name", "InvoiceDate": "ts"})
    .assign(user_id=lambda d: d["user_id"].astype(int), revenue=lambda d: d["Quantity"] * d["Price"])
)

# Keep items with ≥ 5 purchases (prune long tail for cleaner embeddings)
item_counts = df["item_id"].value_counts()
popular_items = item_counts[item_counts >= 5].index
df = df[df["item_id"].isin(popular_items)]

# Deduplicate to one interaction per (user, item) pair
interactions = df.drop_duplicates(subset=["user_id", "item_id"])[["user_id", "item_id"]]

print(f"\nClean interactions : {len(interactions):,}")
print(f"Unique users       : {interactions['user_id'].nunique():,}")
print(f"Unique items       : {interactions['item_id'].nunique():,}")
interactions.head()
```

## 2. Train LightGCN

We use **3 graph-propagation layers** so that second-order neighbours ("customers who bought items bought by people who bought your items") influence the embeddings — a key advantage over matrix factorisation.


```python
t0 = time.perf_counter()

model = LightGCN.from_transactions(
    interactions,
    user_col="user_id",
    item_col="item_id",
    factors=64,  # embedding size
    k_layers=3,  # graph propagation depth
    learning_rate=1e-3,
    lambda_=1e-4,  # L2 regularisation
    iterations=30,
    random_state=42,
    verbose=0,
)

print(f"⚡ LightGCN trained in {time.perf_counter() - t0:.1f}s")
```

## 3. Personalised Homepage Recommendations

For each returning customer, we can instantly serve a personalised shelf of products they've never bought. The call returns original item IDs that can be joined back to the product catalogue.


```python
# Build item name lookup
item_names = df.drop_duplicates("item_id")[["item_id", "item_name"]].set_index("item_id")["item_name"]


def homepage_shelf(customer_id: int, n: int = 6) -> pd.DataFrame:
    """Return personalised product recs with human-readable names."""
    ids, scores = model.recommend_items(user_id=customer_id, n=n)
    return pd.DataFrame(
        {
            "item_id": ids,
            "product_name": [item_names.get(i, "Unknown") for i in ids],
            "relevance_score": np.round(scores, 4),
        }
    )


# Try three different customers to see variety
for cust in [12748, 14609, 17389]:
    print(f"\n👤 Customer {cust}")
    print(homepage_shelf(cust).to_string(index=False))
```

## 4. Similar-Item / Cross-Sell Suggestions

By comparing item embeddings directly (cosine similarity), we can power **"Customers also bought"** widgets — without needing individual user context.


```python

# Build item embedding matrix
item_emb = model._item_factors  # shape: (n_items, d)
item_index = list(model._item_map.keys())  # original item IDs

# Normalise once
norms = np.linalg.norm(item_emb, axis=1, keepdims=True)
item_emb_norm = item_emb / np.clip(norms, 1e-8, None)


def similar_products(item_id, n: int = 5):
    """Return the n most similar products by embedding cosine similarity."""
    internal_idx = model._item_map.get(item_id)
    if internal_idx is None:
        return pd.DataFrame()
    q = item_emb_norm[internal_idx : internal_idx + 1]
    sims = (item_emb_norm @ q.T).flatten()
    top = np.argsort(sims)[::-1][1 : n + 1]  # exclude self
    return pd.DataFrame(
        {
            "item_id": [item_index[i] for i in top],
            "product_name": [item_names.get(item_index[i], "?") for i in top],
            "similarity": sims[top].round(4),
        }
    )


# Example: find similar products to a specific candle holder
anchor_item = interactions["item_id"].value_counts().index[0]  # most popular item
print(f"\n🔍 Products similar to: {item_names.get(anchor_item, anchor_item)}")
print(similar_products(anchor_item).to_string(index=False))
```

## 5. Campaign Audience Scoring

The marketing team wants to promote **three hero products** in next week's email campaign. Instead of blasting the entire list, we score every customer and only contact those with a relevance score above a threshold — protecting sender reputation and reducing churn.


```python
# Pick three campaign items (e.g. seasonal bestsellers)
campaign_items = interactions["item_id"].value_counts().index[1:4].tolist()
print("Campaign items:")
for ci in campaign_items:
    print(f"  {ci}: {item_names.get(ci, '?')}")

# Internal indices
camp_internal = [model._item_map[ci] for ci in campaign_items if ci in model._item_map]
camp_emb = item_emb[camp_internal]  # (n_campaign, d)

# Score all users: shape (n_users, n_campaign)
all_user_emb = model._user_factors  # (n_users, d)
scores_matrix = all_user_emb @ camp_emb.T

# Build leaderboard for Item 0 of the campaign
user_ids = list(model._user_map.keys())
leaderboard = pd.DataFrame(
    {
        "customer_id": user_ids,
        "score": scores_matrix[:, 0],
    }
).sort_values("score", ascending=False)

THRESHOLD = leaderboard["score"].quantile(0.8)  # top 20%

target_audience = leaderboard[leaderboard["score"] >= THRESHOLD]
print(f"\n📣 Campaign audience (top 20%): {len(target_audience):,} customers")
print(f"   Score range: {target_audience['score'].min():.3f} – {target_audience['score'].max():.3f}")
target_audience.head(10)
```

## 6. Segment Analysis — Power vs Casual Buyers

Embedding coordinates encode purchase affinity. We can cluster users into natural segments and describe each segment by its top recommended categories.


```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

N_SEGMENTS = 4

km = KMeans(n_clusters=N_SEGMENTS, n_init=10, random_state=42)
labels = km.fit_predict(all_user_emb)

# Segment sizes
seg_counts = pd.Series(labels).value_counts().sort_index()
print("Segment sizes:")
print(seg_counts.to_string())

# For each segment, find its top-3 recommended items (centroid × item embeddings)
print("\nTop items per segment:")
for seg_id in range(N_SEGMENTS):
    centroid = km.cluster_centers_[seg_id]
    seg_scores = item_emb @ centroid
    top3 = np.argsort(seg_scores)[::-1][:3]
    names = [item_names.get(item_index[i], "?") for i in top3]
    print(f"  Segment {seg_id} ({seg_counts[seg_id]:,} users): {' | '.join(names)}")

# 2-D projection for visualisation
pca = PCA(n_components=2, random_state=0)
umap_2d = pca.fit_transform(all_user_emb)

_, ax = plt.subplots(figsize=(7, 5))
colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
for seg_id in range(N_SEGMENTS):
    mask = labels == seg_id
    ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1], s=6, alpha=0.4, color=colors[seg_id], label=f"Segment {seg_id}")
ax.legend(markerscale=3)
ax.set_title("User Embedding Space (PCA 2D)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.tight_layout()
plt.show()
```

## 7. Business Summary

| Capability | API | Use Case |
|---|---|---|
| Personalised shelf | `model.recommend_items(user_id, n)` | Homepage widget, email recommendations |
| Similar products | Cosine on `model._item_factors` | Product-page cross-sell, "You may also like" |
| Audience scoring | `user_factors @ item_factors.T` | Campaign targeting, propensity models |
| Segmentation | KMeans on `model._user_factors` | CRM clusters, personalised comms strategy |

LightGCN achieves all of the above with **a single 30-second training run** — no GPU required.
