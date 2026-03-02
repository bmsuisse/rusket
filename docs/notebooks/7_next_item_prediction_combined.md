# Next-Item Prediction: Active Cart vs. Sequential History

Predicting what a customer will buy next is one of the most critical applications in e-commerce and retail. However, "next-item prediction" can mean two very different things depending on the context:

1.  **The Active Cart (Association Rules):** What else should we suggest based on what the user has *currently* added to their shopping cart? (e.g., "Frequently Bought Together").
2.  **Sequential History (Sequential Recommendation):** What is the very next item the user will interact with based on the *chronological sequence* of their past actions? (e.g., they viewed a DSLR camera, then a lens, so what's next?).
3.  **The Best of Both Worlds (Hybrid Recommender):** How can we combine our knowledge of the specific user (Personalized Collaborative Filtering) with their immediate intent (Active Cart)?

`rusket` provides blazing-fast, Rust-backed tools for all three scenarios right out of the box.

---

## 1. The Active Cart Approach (Association Rules)

If you want to suggest items based on what is currently sitting in the user's shopping cart, you should use **FP-Growth** or **Eclat**. These algorithms find items that are commonly purchased together across all historical orders, regardless of the order they were added.

### How it works

```python
import pandas as pd
from rusket import FPGrowth

# 1. Provide historical order data (What items appear together in the same basket?)
orders = pd.DataFrame({
    "order_id": [1, 1, 1, 2, 2, 3, 3],
    "sku": ["Laptop", "Mouse", "Keyboard", "Laptop", "Bag", "Mouse", "Keyboard"],
})

# 2. Mine frequent patterns and train the recommender
# min_support=0.1 means items must appear in at least 10% of orders to be considered
model = FPGrowth.from_transactions(
    orders,
    transaction_col="order_id",
    item_col="sku",
    min_support=0.1, 
)

# 3. Predict the next item for a LIVE basket
current_cart = ["Laptop", "Mouse"]
suggestions = model.recommend_items(current_cart, n=3)

print("Suggested next items for active cart:", suggestions)
```

**When to use:** "Frequently Bought Together" widgets on product detail pages or in the shopping cart sidebar.

---

## 2. The Sequential History Approach (SASRec)

If you want to predict the very next item a user will interact with based on their **chronological history** (the *order* matters), use a sequential recommender like **SASRec** (Self-Attentive Sequential Recommendation). 

SASRec uses a Transformer architecture (similar to GPT) which significantly outperforms old Markov-chain methods (like FPMC) by understanding long-range dependencies within a session.

### How it works

```python
import pandas as pd
from rusket import SASRec

# 1. Provide chronological user history (timestamp or implicit order is required)
events = pd.DataFrame({
    "user_id": [101, 101, 101, 102, 102],
    "item_id": ["Phone", "Case", "ScreenProtector", "Phone", "Charger"],
    "timestamp": [1, 2, 3, 1, 2] # Defines the sequence order
})

# 2. Initialize and fit the model
model = SASRec(
    factors=64,
    n_layers=2,
    max_seq=50
).from_transactions(
    events, 
    user_col="user_id", 
    item_col="item_id", 
    timestamp_col="timestamp"
).fit()  # Always call .fit()

# 3. Predict the next item based on an ad-hoc chronological sequence
user_history = ["Phone", "Case"]
next_items, scores = model.recommend_items(user_sequence=user_history, n=3)

print("Predicted next items based on history:", next_items)
```

**When to use:** Identifying anonymous session intent ("Based on what you've viewed"), personalized push notifications, or cart abandonment recovery where the sequence matters.

---

## 3. The "Hybrid" Business Recommender

What if you want to combine Personalized Collaborative Filtering (knowing who the user is and their broad preferences) with the Active Cart (what they are buying right now)? 

`rusket` includes a powerful `Recommender` wrapper that effortlessly blends Collaborative Filtering embeddings (like `ALS`) with Association Rules (like `FPGrowth`).

### How it works

```python
import numpy as np
import pandas as pd
from scipy import sparse
from rusket import ALS, FPGrowth, Recommender

# Let's mock a user-item CSR matrix for ALS
num_users, num_items = 1000, 500
user_interactions_csr = sparse.random(num_users, num_items, density=0.01, format='csr')
user_interactions_csr.data = np.ones_like(user_interactions_csr.data)

# Let's mock a basket dataframe for FPGrowth
cart_data = pd.DataFrame({
    "order_id": [1, 1, 2, 2, 3, 3],
    "item_id": [10, 20, 10, 30, 20, 30]
})

# 1. Train ALS for personalization
als = ALS(factors=64).fit(user_interactions_csr)

# 2. Train FPGrowth for cart rules
rules = FPGrowth.from_transactions(
    cart_data, transaction_col="order_id", item_col="item_id", min_support=0.01
).association_rules()

# 3. Combine them into a Hybrid Engine
rec = Recommender(model=als, rules_df=rules)

# 4. Get predictions knowing both the customer AND their current cart
# The recommendations will boost items that are frequently bought with "Item 10" 
# while also factoring in User 1001's baseline ALS preferences.
add_ons, scores = rec.recommend_for_cart(
    items=[10],   # What's in the cart
    user_id=1001,  # Who is checking out (Optional: adds personalization weight)
    n=5
)

print("Hybrid Recommendations for User 1001's cart:", add_ons)
```

**When to use:** Production systems where you need to deliver the best cross-sell combinations while simultaneously respecting user-specific tastes and historical preferences.

---

## Summary

- Use **`FPGrowth`** for straightforward cart cross-selling when order doesn't matter.
- Use **`SASRec`** if the chronological sequence of interactions is important to the prediction.
- Use **`Recommender`** to seamlessly merge user personalization and live-cart intent into a single output score.
