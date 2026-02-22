"""
Translating Spark MLlib to rusket
=================================

This example translates the famous Recommendation example from Chapter 28
of 'Spark: The Definitive Guide' directly into `rusket` using pure Python and Pandas.

It demonstrates how to:
1. Load `sample_movielens_ratings.txt` using pandas.
2. Perform an 80/20 train/test split.
3. Train a `rusket.ALS` model.
4. Generate predictions for the test set.
5. Evaluate using RMSE.
"""

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from rusket import ALS
from rusket.recommend import score_potential


def download_sample_movielens(data_dir: Path) -> Path:
    """Downloads the sample movielens ratings dataset from the Spark repository."""
    url = "https://raw.githubusercontent.com/apache/spark/master/data/mllib/als/sample_movielens_ratings.txt"
    txt_path = data_dir / "sample_movielens_ratings.txt"

    if not txt_path.exists():
        print(f"Downloading sample movielens ratings dataset to {data_dir}...")
        data_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, txt_path)
        print("Download complete.")

    return txt_path


def main():
    # 1. Load the data using Pandas
    data_dir = Path(__file__).parent.parent / "benchdata"
    txt_path = download_sample_movielens(data_dir)

    print(f"Loading data from {txt_path}...")
    ratings = pd.read_csv(
        txt_path,
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
    )
    print(f"Loaded {len(ratings):,} ratings.")

    # 2. Random Split (80/20)
    print("Performing 80/20 train/test split...")
    shuffled = ratings.sample(frac=1.0, random_state=42)
    split_idx = int(len(shuffled) * 0.8)
    training = shuffled.iloc[:split_idx]
    test = shuffled.iloc[split_idx:]
    print(f"Training set: {len(training):,} rows. Test set: {len(test):,} rows.")

    # 3. Initialize and Fit the ALS Model
    # Note: rusket uses `factors` instead of `rank`, and `iterations` instead of `maxIter`.
    print("Training rusket.ALS model...")
    model = ALS(factors=10, iterations=5, regularization=0.01, seed=42)
    model.fit_transactions(
        training, user_col="userId", item_col="movieId", rating_col="rating"
    )
    print(f"Trained model with {model._n_users} users and {model._n_items} items.")

    # 4. Generate Predictions for the test set
    print("Generating predictions on the test set...")
    # We reconstruct the user's history from the training set to mask known interactions
    user_histories = training.groupby("userId")["movieId"].apply(list).to_dict()
    # Ensure all users in the test set exist in our history mapping, even if empty
    history_list = [user_histories.get(uid, []) for uid in range(model._n_users)]

    # Calculate raw prediction scores across all users and all items
    all_predictions = score_potential(history_list, model)

    # 5. Evaluate RMSE
    print("Evaluating RMSE...")
    # Extract only the actual ratings we care about from the test set
    test_users = test["userId"].values
    test_movies = test["movieId"].values
    actual_ratings = test["rating"].values

    # Map the raw pandas IDs to rusket's internal 0-indexed matrix IDs
    try:
        internal_user_ids = np.array([model._user_labels.index(u) for u in test_users])
        internal_movie_ids = np.array(
            [model._item_labels.index(str(m)) for m in test_movies]
        )

        # Extract predicted ratings
        predicted_ratings = all_predictions[internal_user_ids, internal_movie_ids]

        # Calculate RMSE
        valid_mask = ~np.isinf(predicted_ratings) & ~np.isnan(predicted_ratings)

        if not np.any(valid_mask):
            print(
                "Could not evaluate RMSE: No valid predictions found. "
                "(Likely due to cold-start users/items missing from the training set)."
            )
        else:
            rmse = np.sqrt(
                np.mean(
                    (predicted_ratings[valid_mask] - actual_ratings[valid_mask]) ** 2
                )
            )
            print(f"Root-mean-square error = {rmse:.4f}")
            print(f"Evaluated on {np.sum(valid_mask)} valid predictions.")

    except ValueError as e:
        # Handle cold-start users/items in the test set not seen in training
        print(
            f"Cold start warning: Some users/items in test set were not in training. Error: {e}"
        )

    print("\n--- Spark Script Equivalent Execution Complete ---")


if __name__ == "__main__":
    main()
