import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from rusket.item_knn import ItemKNN


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tx_id": [1, 1, 2, 2, 3, 3, 3, 4],
            "item_id": ["A", "B", "A", "C", "A", "B", "C", "B"],
        }
    )


def test_item_knn_fit_sparse() -> None:
    data = np.ones(8)
    row = np.array([0, 0, 1, 1, 2, 2, 2, 3])
    col = np.array([0, 1, 0, 2, 0, 1, 2, 1])
    X = sp.csr_matrix((data, (row, col)), shape=(4, 3))

    model = ItemKNN(method="bm25", k=2)
    model.fit(X)

    assert model.w_indptr is not None
    assert model.w_indices is not None
    assert model.w_data is not None


def test_item_knn_from_transactions(sample_transactions: pd.DataFrame) -> None:
    model = ItemKNN.from_transactions(sample_transactions, user_col="tx_id", item_col="item_id", method="bm25", k=2)
    assert hasattr(model, "w_indptr")

    ids, scores = model.recommend_items(0, 2)
    assert len(ids) > 0
    assert len(scores) == len(ids)

    # recommend_users is not supported
    with pytest.raises(NotImplementedError):
        model.recommend_users(0, 2)


def test_item_knn_methods(sample_transactions: pd.DataFrame) -> None:
    for method in ["bm25", "tfidf", "cosine", "count"]:
        model = ItemKNN.from_transactions(
            sample_transactions,
            user_col="tx_id",
            item_col="item_id",
            method=method,
            k=2,  # type: ignore
        )
        assert model.w_indptr is not None


def test_item_knn_unfitted() -> None:
    model = ItemKNN()
    with pytest.raises(RuntimeError):
        model.recommend_items(0, 10)


def test_item_knn_out_of_bounds(sample_transactions: pd.DataFrame) -> None:
    model = ItemKNN.from_transactions(sample_transactions, user_col="tx_id", item_col="item_id")
    with pytest.raises(ValueError, match="out of bounds"):
        model.recommend_items(99, 10)
