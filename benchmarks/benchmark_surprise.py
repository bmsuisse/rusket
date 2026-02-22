import time

# surprise
import implicit
import numpy as np
from scipy import sparse
from sklearn.metrics import ndcg_score
from surprise import SVD, Dataset
from surprise.model_selection import split

# rusket
from rusket import BPR


def load_ml100k():
    """Loads the MovieLens 100k dataset built into Surprise."""
    print("Loading MovieLens 100k dataset...")
    data = Dataset.load_builtin("ml-100k")
    return data


def build_sparse_matrix(trainset):
    """Converts a Surprise Trainset to a SciPy CSR matrix for rusket."""
    n_users = trainset.n_users
    n_items = trainset.n_items

    # BPR only needs implicit feedback (1s), so we ignore the actual ratings
    # Or we can binarize (rating >= 4 is a 1, else 0)
    rows, cols, data = [], [], []
    for u, i, r in trainset.all_ratings():
        if r >= 4.0:  # Keep only positive interactions for BPR
            rows.append(u)
            cols.append(i)
            data.append(1.0)

    mat = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    return mat


def evaluate_ndcg(algo, testset, trainset, n=10):
    """Evaluates an algorithm using NDCG@10."""
    user_true_items = {}
    for uid, iid, r in testset:
        if r >= 4.0:
            if uid not in user_true_items:
                user_true_items[uid] = set()
            user_true_items[uid].add(iid)

    # We only evaluate users that have at least 1 true positive in test
    valid_users = list(user_true_items.keys())

    # Calculate NDCG per user
    ndcg_list = []

    # For Rusket (predicts arrays)
    if hasattr(algo, "recommend_items"):
        for uid in valid_users:
            if uid not in trainset._raw2inner_id_users:
                continue
            inner_u = trainset.to_inner_uid(uid)

            recs, _ = algo.recommend_items(inner_u, n=n, exclude_seen=True)

            # Map back to raw generic IDs
            # If item not in training, it's missing. That's fine.
            raw_recs = []
            for inner_i in recs:
                try:
                    raw_recs.append(trainset.to_raw_iid(inner_i))
                except ValueError:
                    pass

            # True vector
            y_true = [1.0 if i in user_true_items[uid] else 0.0 for i in raw_recs]
            # Since rank order matters, we just score the top N
            if sum(y_true) > 0:
                y_pred = list(range(len(raw_recs), 0, -1))  # perfectly decaying scores
                ndcg = ndcg_score([y_true], [y_pred])
                ndcg_list.append(ndcg)
            else:
                ndcg_list.append(0.0)

    # For Surprise (predicts scalars)
    else:
        # Surprise doesn't have a fast `.recommend_items`. We have to predict all unrated manually.
        all_raw_items = list(trainset._raw2inner_id_items.keys())

        for uid in valid_users:
            # Get user true items in trainset
            try:
                inner_u = trainset.to_inner_uid(uid)
                train_items = {
                    trainset.to_raw_iid(j) for j, r in trainset.ur[inner_u]
                }
            except Exception:
                train_items = set()

            predictions = []
            for iid in all_raw_items:
                if iid not in train_items:
                    pred = algo.predict(uid, iid).est
                    predictions.append((iid, pred))

            predictions.sort(key=lambda x: x[1], reverse=True)
            top_n = predictions[:n]

            y_true = [1.0 if i in user_true_items[uid] else 0.0 for i, p in top_n]
            if sum(y_true) > 0:
                y_pred = list(range(len(top_n), 0, -1))
                ndcg = ndcg_score([y_true], [y_pred])
                ndcg_list.append(ndcg)
            else:
                ndcg_list.append(0.0)

    if not ndcg_list:
        return 0.0
    return np.mean(ndcg_list)


def evaluate_ndcg_implicit(algo, testset, trainset, mat, n=10):
    """Evaluates the implicit library using NDCG@10."""
    user_true_items = {}
    for uid, iid, r in testset:
        if r >= 4.0:
            if uid not in user_true_items:
                user_true_items[uid] = set()
            user_true_items[uid].add(iid)

    valid_users = list(user_true_items.keys())
    ndcg_list = []

    # We need the user-item sparse matrix for `implicit` to filter out already seen items
    # and we need to map raw Surprise IDs to inner row IDs.
    for uid in valid_users:
        if uid not in trainset._raw2inner_id_users:
            continue
        inner_u = trainset.to_inner_uid(uid)

        # implicit recommend takes user_id, user_items matrix
        ids, scores = algo.recommend(
            inner_u, mat.tocsr()[inner_u], N=n, filter_already_liked_items=True
        )

        raw_recs = []
        for inner_i in ids:
            try:
                raw_recs.append(trainset.to_raw_iid(inner_i))
            except ValueError:
                pass

        y_true = [1.0 if i in user_true_items[uid] else 0.0 for i in raw_recs]
        if sum(y_true) > 0:
            y_pred = list(range(len(raw_recs), 0, -1))
            ndcg = ndcg_score([y_true], [y_pred])
            ndcg_list.append(ndcg)
        else:
            ndcg_list.append(0.0)

    if not ndcg_list:
        return 0.0
    return np.mean(ndcg_list)


def main():
    data = load_ml100k()

    # 80/20 train/test split
    trainset, testset = split.train_test_split(data, test_size=0.2, random_state=42)

    print("\n--- Training Surprise SVD (Standard Baseline) ---")
    start_time = time.time()
    algo_svd = SVD(n_factors=64, n_epochs=50, lr_all=0.01, random_state=42)
    algo_svd.fit(trainset)
    surprise_time = time.time() - start_time
    print(f"Surprise SVD Training Time: {surprise_time:.2f}s")

    print("\n--- Training Rusket BPR ---")
    mat = build_sparse_matrix(trainset)
    print(f"Sparse Matrix shape: {mat.shape}, nnz: {mat.nnz}")

    start_time = time.time()
    algo_bpr = BPR(
        factors=64, iterations=50, learning_rate=0.01, regularization=0.01, seed=42
    )
    algo_bpr.fit(mat)
    rusket_time = time.time() - start_time
    print(f"Rusket BPR Training Time: {rusket_time:.2f}s")

    print("\n--- Training Implicit BPR ---")
    start_time = time.time()
    algo_implicit = implicit.bpr.BayesianPersonalizedRanking(
        factors=64,
        iterations=50,
        learning_rate=0.01,
        regularization=0.01,
        random_state=42,
    )
    algo_implicit.fit(mat)
    implicit_time = time.time() - start_time
    print(f"Implicit BPR Training Time: {implicit_time:.2f}s")

    print(f"\nSpeedup: Rusket vs Surprise: {surprise_time / rusket_time:.2f}x faster")
    print(f"Speedup: Rusket vs Implicit: {implicit_time / rusket_time:.2f}x faster")

    print("\n--- Evaluating NDCG@10 ---")
    print("Evaluating Rusket BPR... (this is fast natively)")
    score_bpr = evaluate_ndcg(algo_bpr, testset, trainset, n=10)
    print(f"Rusket BPR NDCG@10:   {score_bpr:.4f}")

    print("Evaluating Implicit BPR...")
    score_implicit = evaluate_ndcg_implicit(algo_implicit, testset, trainset, mat, n=10)
    print(f"Implicit BPR NDCG@10: {score_implicit:.4f}")

    print("Evaluating Surprise SVD... (this might take a minute)")
    score_svd = evaluate_ndcg(algo_svd, testset, trainset, n=10)
    print(f"Surprise SVD NDCG@10: {score_svd:.4f}")


if __name__ == "__main__":
    main()
