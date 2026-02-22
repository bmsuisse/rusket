import numpy as np
import scipy.sparse as sp
import pandas as pd
import time
import rusket


def bm25_weight(X, K1=1.2, B=0.75):
    """Weighs each item-user interaction by BM25."""
    # X is Users x Items
    X = sp.coo_matrix(X)

    # Calculate item frequencies (how many users bought each item)
    N = float(X.shape[0])
    idf = np.log(
        (N - np.bincount(X.col, minlength=X.shape[1]) + 0.5) / (np.bincount(X.col, minlength=X.shape[1]) + 0.5) + 1.0
    )

    # Calculate user frequencies (how many items each user bought)
    user_lens = np.bincount(X.row, minlength=X.shape[0])
    avg_len = user_lens.mean()

    # Weight
    weight = (X.data * (K1 + 1.0)) / (X.data + K1 * (1.0 - B + B * user_lens[X.row] / avg_len))
    weight = weight * idf[X.col]

    return sp.csr_matrix((weight, (X.row, X.col)), shape=X.shape)


def fit_bm25_item_knn(X, K=20):
    t0 = time.time()
    # Apply BM25
    X_weight = bm25_weight(X)
    t1 = time.time()

    # Item-item similarity
    W = X_weight.T.dot(X_weight)

    # We should keep top-K per row
    # (Leaving it dense-ish sparse for now just to measure time)
    W.setdiag(0)
    W.eliminate_zeros()
    t2 = time.time()

    print(f"BM25 Weighting: {t1 - t0:.3f}s")
    print(f"BM25 Similarity: {t2 - t1:.3f}s")

    return W


def benchmark():
    df = pd.read_parquet("tests/.dataset_cache/online_retail_II_sample.parquet")
    print(f"Loaded dataset: {df.shape[0]} interactions")

    users = df["Customer_ID"].astype("category").cat.codes
    items = df["Description"].astype("category").cat.codes
    data = np.ones(len(df), dtype=np.float32)
    X = sp.csr_matrix((data, (users, items)))
    print(f"Sparse matrix shape: {X.shape}, nnz: {X.nnz}")

    print("\n--- Fitting Models ---")

    # BM25 Item-KNN
    t0 = time.time()
    W = fit_bm25_item_knn(X)
    bm25_fit_time = time.time() - t0
    print(f"BM25 fit time: {bm25_fit_time:.4f}s")

    print("\n--- Inference ---")
    t0 = time.time()
    for u in range(X.shape[0]):
        u_vec = X[u]
        scores = u_vec.dot(W)
        # Not sorting to just measure raw matmul, but sorting is fast
    inf_time = time.time() - t0
    print(f"BM25 inference: {inf_time:.4f}s")


if __name__ == "__main__":
    benchmark()
