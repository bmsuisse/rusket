# 🛒 E-Shop Vector Recommendations

Build a complete e-shop recommendation engine backed by a vector database.
This tutorial shows four high-value patterns that every online store needs:

| Pattern | Signal | Use Case |
|---|---|---|
| **Personalised "For You"** | User → Items | Homepage carousel, email campaigns |
| **Similar Items** | Item → Items | "You might also like" on PDP |
| **Also Bought** | Cart → Items | Cross-sell widget, checkout upsells |
| **Customer Lookalikes** | User → Users → Items | Cold-start for new users, CRM segments |

All patterns follow the same workflow:

```
Train model → Export embeddings → Query with native SDK
```

---

## Synthetic E-Shop Data

We'll use a small synthetic dataset throughout. Replace it with your real purchase
history and product catalogue.

```python
import numpy as np
import pandas as pd
import rusket

# ── Purchase history ──────────────────────────────────────────────
np.random.seed(42)
n_users, n_items = 500, 200

rows = []
for user_id in range(n_users):
    # Each user buys 5–20 items
    bought = np.random.choice(n_items, size=np.random.randint(5, 21), replace=False)
    for item_id in bought:
        rows.append({"user_id": user_id, "item_id": int(item_id)})

purchases = pd.DataFrame(rows)

# ── Product catalogue ─────────────────────────────────────────────
categories = ["Electronics", "Clothing", "Home & Kitchen", "Sports", "Books"]
catalog = pd.DataFrame({
    "item_id": range(n_items),
    "name": [f"Product {i}" for i in range(n_items)],
    "category": np.random.choice(categories, n_items),
    "price": np.round(np.random.uniform(5, 500, n_items), 2),
    "in_stock": np.random.choice([True, True, True, False], n_items),  # 75 % in stock
})

print(f"Purchases: {len(purchases):,}  |  Users: {n_users}  |  Items: {n_items}")
print(purchases.head())
```

---

## Step 1 — Train the Recommender

ALS produces both **user** and **item** factor matrices.
Both are needed for the patterns below.

```python
als = (
    rusket.ALS(factors=64, iterations=15, regularization=0.1)
    .from_transactions(purchases, user_col="user_id", item_col="item_id")
    .fit()
)

print(f"User factors : {als.user_factors.shape}")   # (500, 64)
print(f"Item factors : {als.item_factors.shape}")    # (200, 64)
```

!!! tip "Any latent-factor model works"
    `BPR`, `SVD`, `NMF`, or `LightGCN` all expose `.user_factors` / `.item_factors`.
    Pick whichever gives you the best offline metrics.

---

## Step 2 — Export to a Vector Database

We'll use **Qdrant** for the examples, but every pattern works with
[any supported backend](../vectordb.md#supported-backends).

### Export item embeddings

```python
from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

# Item embeddings — used for "Similar Items" and "Also Bought"
rusket.export_vectors(
    als.item_factors,
    client=client,
    collection_name="shop_items",
    ids=catalog["item_id"].tolist(),
    payloads=catalog.drop(columns="item_id").to_dict("records"),
)
```

### Export user embeddings

```python
# User embeddings — used for "For You" and "Customer Lookalikes"
rusket.export_vectors(
    als.user_factors,
    client=client,
    collection_name="shop_users",
)
```

---

## Pattern 1 — Personalised "For You"

> *"Show this user items they're most likely to buy."*

Query the **item** collection with the **user's** embedding vector.
Items closest to the user in latent space are the strongest recommendations.

```python
def recommend_for_you(
    user_id: int, n: int = 10, category: str | None = None
) -> list[dict]:
    """Homepage 'For You' carousel."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    user_vector = als.user_factors[user_id].tolist()

    filters = [FieldCondition(key="in_stock", match=MatchValue(value=True))]
    if category:
        filters.append(
            FieldCondition(key="category", match=MatchValue(value=category))
        )

    results = client.query_points(
        collection_name="shop_items",
        query=user_vector,
        query_filter=Filter(must=filters),
        limit=n,
        with_payload=True,
    )
    return [
        {"item_id": r.id, "score": round(r.score, 4), **r.payload}
        for r in results.points
    ]


# Try it
recs = recommend_for_you(user_id=0, n=5)
for r in recs:
    print(f"  {r['item_id']:>3}  {r['name']:<15}  {r['category']:<16}  score={r['score']}")
```

```
  142  Product 142      Electronics       score=0.9812
   67  Product 67       Electronics       score=0.9634
  ...
```

---

## Pattern 2 — Similar Items

> *"Customers who viewed **this** item also looked at…"*

Query the **item** collection with **another item's** embedding.
Nearest neighbours in item space share buying patterns.

```python
def similar_items(item_id: int, n: int = 5) -> list[dict]:
    """'You might also like' on the product detail page."""
    item_vector = als.item_factors[item_id].tolist()

    results = client.query_points(
        collection_name="shop_items",
        query=item_vector,
        limit=n + 1,          # +1 to exclude the query item itself
        with_payload=True,
    )
    return [
        {"item_id": r.id, "score": round(r.score, 4), **r.payload}
        for r in results.points
        if r.id != item_id
    ][:n]


similars = similar_items(item_id=42, n=5)
for s in similars:
    print(f"  {s['item_id']:>3}  {s['name']:<15}  {s['category']:<16}  sim={s['score']}")
```

```
   87  Product 87       Electronics       sim=0.973
  123  Product 123      Home & Kitchen    sim=0.961
  ...
```

---

## Pattern 3 — "Customers Who Bought X Also Bought Y"

> *Cross-sell widget on the cart or checkout page.*

**Idea:** Average the embeddings of items currently in the cart to create
a *session vector*, then search for the closest items that are **not**
already in the cart.

```python
import numpy as np

def also_bought(cart_item_ids: list[int], n: int = 5) -> list[dict]:
    """Cross-sell: items frequently co-purchased with the current cart."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    # Build a centroid from all cart item embeddings
    cart_vectors = np.array([als.item_factors[i] for i in cart_item_ids])
    centroid = cart_vectors.mean(axis=0).tolist()

    results = client.query_points(
        collection_name="shop_items",
        query=centroid,
        query_filter=Filter(
            must=[FieldCondition(key="in_stock", match=MatchValue(value=True))]
        ),
        limit=n + len(cart_item_ids),
        with_payload=True,
    )

    # Remove items already in the cart
    return [
        {"item_id": r.id, "score": round(r.score, 4), **r.payload}
        for r in results.points
        if r.id not in cart_item_ids
    ][:n]


# User has items 10 and 55 in cart
cross_sells = also_bought(cart_item_ids=[10, 55], n=5)
for c in cross_sells:
    print(f"  {c['item_id']:>3}  {c['name']:<15}  {c['category']:<16}  rel={c['score']}")
```

```
  132  Product 132      Sports            rel=0.952
   78  Product 78       Electronics       rel=0.943
  ...
```

!!! info "Why centroid averaging works"
    ALS learns a latent space where items bought together are close.
    Averaging the cart vectors moves the query point toward the cluster
    of items that overlap with *all* cart members — a natural cross-sell signal.

---

## Pattern 4 — Customer Lookalikes

> *"Find users with similar taste, then recommend what they bought."*

This is invaluable for **cold-start** users (few purchases) and
**CRM segmentation** ("find 1,000 users who look like our VIPs").

### 4a — Find similar customers

```python
def find_similar_customers(user_id: int, n: int = 5) -> list[dict]:
    """Find users with the most similar purchase behaviour."""
    user_vector = als.user_factors[user_id].tolist()

    results = client.query_points(
        collection_name="shop_users",
        query=user_vector,
        limit=n + 1,
        with_payload=True,
    )
    return [
        {"user_id": r.id, "score": round(r.score, 4)}
        for r in results.points
        if r.id != user_id
    ][:n]


lookalikes = find_similar_customers(user_id=0, n=5)
print("Users most similar to user 0:")
for l in lookalikes:
    print(f"  user {l['user_id']:>3}  similarity={l['score']}")
```

### 4b — Recommend from lookalike purchases

```python
def recommend_from_lookalikes(
    user_id: int, n_neighbours: int = 10, n_recs: int = 10
) -> list[dict]:
    """Recommend items that similar users bought but this user hasn't."""
    # Items this user already bought
    user_items = set(
        purchases.loc[purchases["user_id"] == user_id, "item_id"]
    )

    # Find lookalike users
    neighbours = find_similar_customers(user_id, n=n_neighbours)

    # Collect their purchases, weighted by similarity
    from collections import Counter
    item_scores: Counter = Counter()
    for nb in neighbours:
        nb_items = set(
            purchases.loc[purchases["user_id"] == nb["user_id"], "item_id"]
        )
        for item_id in nb_items - user_items:
            item_scores[item_id] += nb["score"]

    # Return top-N
    top_items = item_scores.most_common(n_recs)
    results = []
    for item_id, score in top_items:
        row = catalog.loc[catalog["item_id"] == item_id].iloc[0]
        results.append({
            "item_id": item_id,
            "name": row["name"],
            "category": row["category"],
            "score": round(score, 4),
        })
    return results


recs = recommend_from_lookalikes(user_id=0, n_recs=5)
for r in recs:
    print(f"  {r['item_id']:>3}  {r['name']:<15}  {r['category']:<16}  score={r['score']}")
```

---

## Putting It Together — FastAPI E-Shop Service

Wrap all four patterns into a production-ready API:

```python
from fastapi import FastAPI, Query as Q
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import numpy as np
import rusket

app = FastAPI(title="E-Shop Recommendations")

client = QdrantClient("localhost", port=6333)
model = rusket.load_model("trained_als.pkl")  # persist and reload


@app.get("/for-you/{user_id}")
async def for_you(user_id: int, n: int = Q(default=10, le=50)):
    """Personalised recommendations."""
    vec = model.user_factors[user_id].tolist()
    res = client.query_points(
        collection_name="shop_items", query=vec,
        query_filter=Filter(must=[
            FieldCondition(key="in_stock", match=MatchValue(value=True))
        ]),
        limit=n, with_payload=True,
    )
    return [{"id": r.id, "score": r.score, **r.payload} for r in res.points]


@app.get("/similar/{item_id}")
async def similar(item_id: int, n: int = Q(default=5, le=20)):
    """Items similar to the given item."""
    vec = model.item_factors[item_id].tolist()
    res = client.query_points(
        collection_name="shop_items", query=vec,
        limit=n + 1, with_payload=True,
    )
    return [
        {"id": r.id, "score": r.score, **r.payload}
        for r in res.points if r.id != item_id
    ][:n]


@app.post("/also-bought")
async def also_bought(cart: list[int], n: int = Q(default=5, le=20)):
    """Cross-sell based on current cart contents."""
    vecs = np.array([model.item_factors[i] for i in cart])
    centroid = vecs.mean(axis=0).tolist()
    res = client.query_points(
        collection_name="shop_items", query=centroid,
        query_filter=Filter(must=[
            FieldCondition(key="in_stock", match=MatchValue(value=True))
        ]),
        limit=n + len(cart), with_payload=True,
    )
    return [
        {"id": r.id, "score": r.score, **r.payload}
        for r in res.points if r.id not in cart
    ][:n]


@app.get("/lookalikes/{user_id}")
async def lookalikes(user_id: int, n: int = Q(default=10, le=50)):
    """Find customers with similar taste."""
    vec = model.user_factors[user_id].tolist()
    res = client.query_points(
        collection_name="shop_users", query=vec,
        limit=n + 1, with_payload=True,
    )
    return [
        {"id": r.id, "score": r.score}
        for r in res.points if r.id != user_id
    ][:n]
```

---

## Architecture Overview

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Purchase Log   │────▶│  rusket.ALS.fit() │────▶│  User & Item     │
│   (DataFrame)    │     │                  │     │  Factor Matrices │
└──────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                           │
                              rusket.export_vectors()      │
                         ┌─────────────────────────────────┘
                         ▼
               ┌──────────────────┐
               │   Vector DB      │
               │ ┌──────────────┐ │
               │ │ shop_items   │ │  ← item embeddings + catalogue metadata
               │ └──────────────┘ │
               │ ┌──────────────┐ │
               │ │ shop_users   │ │  ← user embeddings
               │ └──────────────┘ │
               └────────┬─────────┘
                        │
           Native SDK queries (Qdrant / pgvector / …)
                        │
       ┌────────────────┼────────────────┐
       ▼                ▼                ▼
  ┌──────────┐   ┌────────────┐   ┌──────────────┐
  │ For You  │   │ Similar    │   │ Also Bought  │
  │ (user→   │   │ Items      │   │ (cart avg →  │
  │  items)  │   │ (item→     │   │  items)      │
  └──────────┘   │  items)    │   └──────────────┘
                 └────────────┘
```

---

## Tips & Variations

!!! tip "Use BPR or LightGCN for implicit feedback"
    ALS works well, but `rusket.BPR` and `rusket.LightGCN` often yield
    better embeddings for click/purchase data with no explicit ratings.

!!! tip "Hybrid embeddings for richer signals"
    Fuse CF vectors with text/image embeddings using
    `rusket.HybridEmbeddingIndex` and export as multi-vectors.
    See the [Vector DB docs](../vectordb.md#multi-vector-export) for details.

!!! tip "pgvector alternative"
    Every pattern above works identically with PostgreSQL + pgvector.
    Just replace the Qdrant SDK calls with SQL queries — see the
    ["Also Bought" SQL example](../vectordb.md#customers-who-bought-x-also-bought-y-sql).

!!! tip "Refresh cadence"
    Retrain and re-export embeddings on a schedule (e.g. nightly).
    Vector DB queries keep working during the export via collection aliases.
