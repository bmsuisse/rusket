# Hybrid Recommendations & User Potential with MovieLens

In this cookbook, we will use the classic **MovieLens 100k dataset** to showcase `rusket`'s collaborative filtering models (`ALS` and `BPR`) and demonstrate how to deploy the ultimate hybrid engine using `NextBestAction`.


```
import time

import numpy as np
import pandas as pd

from rusket import ALS, BPR, NextBestAction
from rusket.recommend import score_potential
```

## 1. Load MovieLens Data
We will download and parse the MovieLens 100k dataset into a Pandas DataFrame representing user ID, movie ID, and rating (1-5).


```
import os
import urllib.request
import zipfile

if not os.path.exists("ml-100k"):
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    urllib.request.urlretrieve(url, "ml-100k.zip")
    with zipfile.ZipFile("ml-100k.zip", "r") as zip_ref:
        zip_ref.extractall(".")

columns = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv("ml-100k/u.data", sep="\t", names=columns)

print(f"Loaded {len(df):,} ratings!")
df.head()
```

## 2. Fitting Alternating Least Squares (ALS)
`rusket` provides `ALS`, a highly optimized collaborative filtering engine that learns latent representations of users based on their interaction histories.

We can fit our model directly off the raw Pandas transaction dataframe using `fit_transactions`:


```
als_model = ALS(factors=64, iterations=15, alpha=15.0, regularization=0.01, seed=42)

t0 = time.time()
als_model.fit_transactions(df, user_col="user_id", item_col="item_id", rating_col="rating")
print(f"⚡ ALS training complete in {time.time() - t0:.4f}s!")
```

## 3. Bayesian Personalized Ranking (BPR)
While ALS is great for general score predictions, BPR natively optimizes for **Rank**. This makes it the superior choice when your goal is purely to rank Top-N recommendations based entirely on implicit (binary) views or clicks, ignoring explicit star ratings.


```
bpr_model = BPR(factors=64, iterations=150, learning_rate=0.05, regularization=0.01)

# Notice BPR trains on implicit behavior (user_col and item_col), completely ignoring ratings
t0 = time.time()
bpr_model.fit_transactions(df, user_col="user_id", item_col="item_id")
print(f"⚡ BPR training complete in {time.time() - t0:.4f}s!")
```

## 4. Next Best Action (Hybrid API)
The `NextBestAction` engine wraps these complicated matrices and index boundaries into a dead-simple business API for analysts. 

If we pass it our `als_model`, we can instantly ask for the top 5 next best products for a handful of target users.


```
nba = NextBestAction(als_model=als_model)

# Target users from our CRM
target_users = pd.DataFrame({"customer_id": [1, 5, 25, 42]})

# Predict the best 3 movies for these users to watch next!
recommendations = nba.predict_next_chunk(target_users, user_col="customer_id", k=3)
recommendations
```

## 5. Marketing Potential Score

Want to launch an email marketing campaign for specific movies (say movies `10`, `50`, and `100`), but only want to email users who are *highly primed* to buy them? 

We can use the `score_potential` API to predict their exact likelihood of interaction across the entire customer base.


```
user_histories = df.groupby("user_id")["item_id"].apply(list).tolist()

potential_scores = score_potential(user_history=user_histories, als_model=als_model, target_categories=[10, 50, 100])

print("Top 5 user scores for Movie ID 10:")
# Get users with the highest probability to interact with Movie 10
movie_10_scores = potential_scores[:, 0]  # First column corresponds to Item 10
best_users_for_campaign = np.argsort(movie_10_scores)[::-1][:5]

for u in best_users_for_campaign:
    print(f"User {u}: {movie_10_scores[u]:.2f}")
```
