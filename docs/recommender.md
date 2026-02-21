# Recommender Workflows

While `rusket` provides blazing-fast core algorithms like Alternating Least Squares (ALS) and FP-Growth, raw algorithms often require heavy lifting to translate into business outcomes. 

To bridge this gap, `rusket` includes a high-level **Business Recommender API** designed for e-commerce, content platforms, and marketing analytics.

---

## The Hybrid Recommender

The `Recommender` class blends the serendipity of **Collaborative Filtering** (ALS) with the strict deterministic logic of **Frequent Pattern Mining** (Association Rules) to provide the "Next Best Action" for any context.

```python
import pandas as pd
from rusket import ALS, Recommender, mine, association_rules

# 1. Train your Collaborative Filtering model
als = ALS(factors=64, iterations=15).fit(user_item_sparse_matrix)

# 2. Mine your Association Rules
freq = mine(user_item_sparse_matrix, min_support=0.01)
rules = association_rules(freq, num_itemsets=user_item_sparse_matrix.shape[0])

# 3. Create the Hybrid Engine
rec = Recommender(als_model=als, rules_df=rules)
```

### 1. Personalized Recommendations
Use `recommend_for_user` to generate a customized list of products for a returning customer based on their latent profile (ALS).

```python
# Get top 5 product recommendations for user 42
items, scores = rec.recommend_for_user(user_id=42, n=5)

print(f"Recommended Items: {items}")
# Recommended Items: [34, 12, 89, 7, 102]
```

### 2. Cart-based Cross-Selling
When a user adds items to their shopping cart, you want to show a "Frequently bought together" carousel. The `recommend_for_cart` method uses deterministic association rules to find the highest lift recommendations for the contents of the cart.

```python
# User has items 14 and 7 in their cart
suggested = rec.recommend_for_cart([14, 7], n=3)

print(f"Others also bought: {suggested}")
# Others also bought: [8, 45, 99] 
```

---

## Item-to-Item Similarity (i2i)

Often you don't have a user context (e.g., an anonymous visitor on a product description page). In these cases, you can use the latent item factors learned by the ALS model to find conceptually similar products.

The `similar_items` function performs ultra-fast Cosine Similarity over the ALS item embeddings.

```python
from rusket import similar_items
from rusket import ALS

# Fit ALS on your interaction data
als = ALS(factors=128).fit(interactions)

# Find 4 products most similar to product 99
similar, similarities = similar_items(als, item_id=99, n=4)

print(similar) 
# [100, 95, 102, 88]
print(similarities)
# [0.98, 0.94, 0.89, 0.85]
```

> **Note:** Because this operates on latent factors, it discovers implicit relationships. For example, it might identify that a high-end coffee grinder is similar to an expensive espresso machine, even if they aren't directly bought in the exact same cart.

---

## Cross-Selling Potential Scoring

For marketing teams, sending generic email blasts is inefficient. The `score_potential` function quantifies the "missed opportunity" by calculating the probability a user *should* have bought an item by now, but hasn't.

This is perfect for building highly targeted audiences for email campaigns or retargeting pixels.

```python
from rusket import score_potential

# user_history: list of item IDs each user has already interacted with
# target_categories: optional list of specific item/category IDs to score

potential_matrix = score_potential(
    user_history=[[14, 7], [99], [5, 6, 7]], 
    als_model=als,
    target_categories=[101, 102, 103] # e.g., the "Electronics" category
)

# The result is a dense numpy array of shape (n_users, len(target_categories))
# Values are raw ALS interaction scores. Items the user has already bought 
# are masked with -infinity.
```

By sorting users by their maximum score in `potential_matrix`, you can instantly generate a ranked list of the absolute best customers to target for an upcoming product launch in the `Electronics` category.
