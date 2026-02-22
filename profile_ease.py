import scipy.sparse as sp
import numpy as np
import pandas as pd
import time

df = pd.read_parquet("tests/.dataset_cache/online_retail_II_sample.parquet")
users = df["Customer_ID"].astype("category").cat.codes
items = df["Description"].astype("category").cat.codes
data = np.ones(len(df), dtype=np.float32)
X = sp.csr_matrix((data, (users, items)))
n_items = X.shape[1]

t0 = time.time()
G = X.T.dot(X)
t1 = time.time()
G_dense = G.toarray()
t2 = time.time()
diag_indices = np.diag_indices(n_items)
G_dense[diag_indices] += 500.0
P = np.linalg.inv(G_dense)
t3 = time.time()

print(f"X^T * X sparse: {t1 - t0:.3f}s")
print(f"toarray: {t2 - t1:.3f}s")
print(f"inv: {t3 - t2:.3f}s")
