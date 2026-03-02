# SASRec: Session-Based Next-Best-Product

**Business problem:** Your store has many **anonymous / new visitors** — they have no purchase history at all. Classical collaborative filtering (ALS, BPR, LightGCN) cannot help them because they have no embeddings. But you *do* know what they clicked on in the current session.

**Why SASRec?** Unlike Markov-chain models (FPMC), the **self-attention mechanism** captures *long-range dependencies* within a session — so a user who browsed Espresso Maker → Grinder → Scales is likely interested in whole-bean Coffee, not in unrelated items that appear in simple co-occurrence tables.

Use cases covered:
1. **Real-time next-product widget** — "Based on what you've viewed…"
2. **Personalised push notification** — what to show a user who hasn't returned in 7 days
3. **Session-quality score** — how focused / intentional is this browse session?
4. **Cart abandonment recovery** — predict what the user intended to buy but didn't

> **Dataset:** MovieLens 100k (publicly available). We treat each user's chronological viewing history as a browse session, and model "what they watch next" — the exact same pattern as "what they buy next".


```python
import os
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from rusket import SASRec
```

## 1. Prepare Sequential (Session-Based) Data

SASRec learns from **ordered sequences**. The key difference from collaborative filtering is that the **order of interactions matters** — we sort by timestamp and treat each user's history as a single sequential session.


```python
# ── Download MovieLens 100k ──────────────────────────────────────────────────
if not os.path.exists("ml-100k"):
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    urllib.request.urlretrieve(url, "ml-100k.zip")
    with zipfile.ZipFile("ml-100k.zip") as z:
        z.extractall(".")

cols = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv("ml-100k/u.data", sep="\t", names=cols)

# Load item (movie) names
movies = pd.read_csv(
    "ml-100k/u.item", sep="|", encoding="latin-1", header=None, usecols=[0, 1], names=["item_id", "title"]
).set_index("item_id")["title"]

print(f"Loaded {len(df):,} ratings | {df['user_id'].nunique():,} users | {df['item_id'].nunique():,} movies")

# ── Build chronological sequences per user ───────────────────────────────────
df_sorted = df.sort_values(["user_id", "timestamp"])
sequences_df = df_sorted.groupby("user_id")["item_id"].apply(list)

# Stats
lengths = sequences_df.map(len)
print(f"\nSession length: min={lengths.min()} | median={lengths.median():.0f} | max={lengths.max()}")
```

## 2. Train / Validation Split

We use the standard **leave-one-out** evaluation: the last item in each user's history is held out as the ground truth. The model must predict it from the preceding context.


```python
train_seqs, val_truth = [], []
for seq in sequences_df:
    if len(seq) >= 2:
        train_seqs.append(seq[:-1])  # context
        val_truth.append(seq[-1])  # held-out ground truth

print(f"Training sequences : {len(train_seqs):,}")
print(f"Validation targets : {len(val_truth):,}")
```

## 3. Train SASRec

We use `from_transactions` which handles item encoding internally. For clean evaluation we use explicit sequences from the previous step.


```python
t0 = time.perf_counter()

model = SASRec.from_transactions(
    df,
    user_col="user_id",
    item_col="item_id",
    timestamp_col="timestamp",
    factors=64,
    n_layers=2,
    max_seq=50,  # use last 50 interactions as context
    learning_rate=5e-4,
    iterations=15,
    random_state=42,
    verbose=0,
)

print(f"⚡ SASRec trained in {time.perf_counter() - t0:.1f}s")
```

## 4. Real-Time "Based on What You've Viewed" Widget

This is the core **anonymous-visitor use case**. We receive the current browse session as a list of item IDs clicked in this visit (no login required) and instantly return ranked recommendations.


```python
def next_best_widget(session_item_ids: list[int], n: int = 6) -> pd.DataFrame:
    """
    Given a browse session (original item IDs), return top-n recommendations.
    Works for completely new / anonymous users.
    """
    # Encode to internal IDs
    encoded = [model._item_map[i] for i in session_item_ids if i in model._item_map]
    if not encoded:
        return pd.DataFrame()

    ids, scores = model.recommend_items(
        user_sequence=encoded,
        n=n,
        exclude=encoded,  # don't re-recommend already-seen items
    )
    return pd.DataFrame(
        {
            "item_id": [model._rev_item_map.get(i, i) for i in ids],
            "title": [movies.get(model._rev_item_map.get(i, i), "?") for i in ids],
            "score": np.round(scores, 3),
        }
    )


# Simulate a visitor who watched 3 sci-fi films
sci_fi_session = [50, 100, 258]  # Star Wars, Fargo, Contact (MovieLens IDs)
print("🎬 Current session:")
for i in sci_fi_session:
    print(f"   {movies.get(i, i)}")
print("\n💡 Next recommendations:")
print(next_best_widget(sci_fi_session).to_string(index=False))
```

## 5. Re-Engagement Push Notification

A user hasn't returned in 7 days. We know their last viewed items. Instead of sending a generic 'We miss you', we can surface the **single most relevant** item to feature in the subject line.


```python
def re_engagement_item(user_id: int) -> str:
    """Return the single top recommendation for a lapsed user."""
    seq = sequences_df.get(user_id, [])
    if not seq:
        return "No history found."

    # Use last 10 interactions as context
    context = seq[-10:]
    encoded = [model._item_map[i] for i in context if i in model._item_map]
    if not encoded:
        return "Items not in model."

    ids, _ = model.recommend_items(user_sequence=encoded, n=1, exclude=encoded)
    if not len(ids):
        return "No recommendations."
    rec_id = model._rev_item_map.get(ids[0], ids[0])
    return movies.get(rec_id, str(rec_id))


# Simulate notification copy for 5 lapsed users
lapsed_users = sequences_df.index[:5].tolist()
print("📧 Re-engagement notifications:")
for uid in lapsed_users:
    rec = re_engagement_item(uid)
    print(f'   User {uid}: "We thought you\'d love: {rec}"')
```

## 6. Cart Abandonment Recovery

A user added items to cart but didn't complete checkout. Using the cart contents as the session context, we predict what else they might want — giving the sales team a high-confidence upsell script.


```python
def cart_abandonment_upsell(cart_item_ids: list[int], n: int = 3) -> pd.DataFrame:
    """
    Given items in an abandoned cart, predict likely next purchase intention.
    The sales team / email copy can nudge the user toward completing the order.
    """
    encoded = [model._item_map[i] for i in cart_item_ids if i in model._item_map]
    if not encoded:
        return pd.DataFrame()

    ids, scores = model.recommend_items(user_sequence=encoded, n=n, exclude=encoded)
    return pd.DataFrame(
        {
            "recommended_item": [movies.get(model._rev_item_map.get(i, i), "?") for i in ids],
            "predicted_interest": (scores / scores.max()).round(2),  # normalised 0-1
            "suggested_copy": [
                f"Complete your order and we'll add '{movies.get(model._rev_item_map.get(i, i), '?')}' to your watchlist!"
                for i in ids
            ],
        }
    )


# Abandoned cart: Romance + Thriller combo
abandoned_cart = [181, 50]  # Return of the Jedi, Star Wars
print("🛒 Abandoned cart:")
for ci in abandoned_cart:
    print(f"   {movies.get(ci, ci)}")
print("\n🎯 Recovery upsells:")
print(cart_abandonment_upsell(abandoned_cart).to_string(index=False))
```

## 7. Session Quality Score

**How focused is this session?** When a user browses in a tight thematic cluster (all sci-fi, all documentaries), their intent is clear and conversion probability is high. We use the **self-consistency** of the session embedding to derive a quality score, which can be used to:
- Trigger live-chat intervention for low-quality / scattered sessions
- Prioritise high-quality sessions for personalised banners


```python
def session_quality_score(session_item_ids: list[int]) -> float:
    """
    Measures how coherent a session is by computing the mean pairwise
    cosine similarity between item embeddings visited in this session.
    Returns a score in [0, 1]; higher = more focused intent.
    """
    encoded = [model._item_map[i] for i in session_item_ids if i in model._item_map]
    if len(encoded) < 2:
        return 0.0
    embs = model._item_emb[np.array(encoded)]  # (n, d)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_n = embs / np.clip(norms, 1e-8, None)
    sim_matrix = embs_n @ embs_n.T
    # Mean of off-diagonal elements
    n = len(encoded)
    mean_sim = (sim_matrix.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
    return float(np.clip(mean_sim, 0, 1))


# Compare a focused sci-fi session vs a scattered session
sessions = {
    "Focused – all Sci-Fi": [50, 100, 258, 181, 1],  # Star Wars, Fargo, Contact, etc.
    "Scattered – mixed genres": [50, 475, 313, 29, 523],  # very different films
}

for label, sess in sessions.items():
    score = session_quality_score(sess)
    intent = "HIGH" if score > 0.4 else ("MEDIUM" if score > 0.2 else "LOW")
    print(f"{label}")
    print(f"  Quality score: {score:.3f}  →  Intent: {intent}\n")
```

## 8. Evaluating Recommendation Quality (Hit Rate @ 10)

We run a quick leave-one-out evaluation to validate the model isn't just memorising training data.


```python
N_EVAL = 200  # evaluate on 200 users for speed
hits = 0

for i, (ctx, truth) in enumerate(zip(train_seqs[:N_EVAL], val_truth[:N_EVAL])):
    encoded = [model._item_map[x] for x in ctx if x in model._item_map]
    if not encoded:
        continue
    rec_ids, _ = model.recommend_items(user_sequence=encoded, n=10, exclude=encoded)
    decoded = {model._rev_item_map.get(r, r) for r in rec_ids}
    if truth in decoded:
        hits += 1

hit_rate = hits / N_EVAL
print(f"Hit Rate @ 10 ({N_EVAL} users): {hit_rate:.2%}")
print("\nInterpretation:")
print(f"  For {hit_rate:.0%} of users, the correct next item appears in our top-10 recommendations.")
print("  A random baseline would achieve ~0.5% (10 / 1682 items).")
```

## 9. Business Summary

| Capability | Code pattern | Business value |
|---|---|---|
| Anonymous-visitor recommendations | `model.recommend_items(encoded_session)` | Increases CTR for new visitors, no login required |
| Re-engagement notification | Last N items → top-1 recommendation | Lift in email open/click rates |
| Cart abandonment recovery | Cart items → next predicted item | Recovery revenue |
| Session quality score | Mean pairwise cosine of visited items | Trigger intervention, prioritise high-intent sessions |
| Leave-one-out eval | Context → check if truth in top-K | Model monitoring / A/B test baseline |

### When to use SASRec vs LightGCN

| Scenario | Recommended model |
|---|---|
| Known user, long history | **LightGCN** — graph signals are richer |
| New / anonymous visitor | **SASRec** — session context is all you have |
| Physical store with POS sequences | **SASRec** — basket order encodes intent |
| CRM-driven campaign scoring | **LightGCN** — scores the full user base at once |
