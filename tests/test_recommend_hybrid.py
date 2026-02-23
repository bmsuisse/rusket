import numpy as np

from rusket.als import ALS
from rusket.recommend import Recommender


def test_hybrid_recommender():
    # 3 users, 6 items
    import scipy.sparse as sp

    # User 0: 0, 1, 2
    # User 1: 1, 2, 3
    # User 2: 4, 5
    row = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    col = np.array([0, 1, 2, 1, 2, 3, 4, 5])
    data = np.ones(8, dtype=np.float32)
    csr = sp.csr_matrix((data, (row, col)), shape=(3, 6))

    model = ALS(factors=4, iterations=10, seed=42)
    model.fit(csr)

    # 6 items, 2d embeddings
    # Item 0 and 5 are highly similar
    embeddings = np.array(
        [
            [1.0, 0.0],  # item 0
            [0.0, 1.0],  # item 1
            [0.0, 1.0],  # item 2
            [1.0, 1.0],  # item 3
            [-1.0, 1.0],  # item 4
            [1.0, 0.0],  # item 5
        ]
    )

    rec = Recommender(model=model, item_embeddings=embeddings)

    # User 0 saw [0, 1, 2].
    # Semantic: anchor on item 0 -> [1,0]
    # Nearest item by semantic is item 5 -> [1,0]

    # Pure CF (alpha=1.0)
    cf_recs, _ = rec.recommend_for_user(0, n=2, alpha=1.0)

    # Pure Semantic (alpha=0.0) anchored on item 0
    sem_recs, _ = rec.recommend_for_user(0, n=2, alpha=0.0, target_item_for_semantic=0)

    # Hybrid
    hyb_recs, _ = rec.recommend_for_user(0, n=2, alpha=0.5, target_item_for_semantic=0)

    assert len(cf_recs) == 2
    assert len(sem_recs) == 2
    assert 5 in sem_recs, "Semantic similarity should prioritize item 5 since it perfectly matches item 0"
