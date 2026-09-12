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
    model = ItemKNN.from_transactions(
        sample_transactions, user_col="tx_id", item_col="item_id", method="bm25", k=2
    ).fit()
    assert hasattr(model, "w_indptr")

    ids, scores = model.recommend_items(0, 2)
    assert len(ids) > 0
    assert len(scores) == len(ids)

    # recommend_users is not supported
    with pytest.raises(NotImplementedError):
        model.recommend_users(0, 2)  # type: ignore


def test_item_knn_methods(sample_transactions: pd.DataFrame) -> None:
    for method in ["bm25", "tfidf", "cosine", "count"]:
        model = ItemKNN.from_transactions(
            sample_transactions,
            user_col="tx_id",
            item_col="item_id",
            method=method,
            k=2,  # type: ignore
        ).fit()
        assert model.w_indptr is not None  # type: ignore


def test_item_knn_unfitted() -> None:
    model = ItemKNN()
    with pytest.raises(RuntimeError):
        model.recommend_items(0, 10)


def test_item_knn_out_of_bounds(sample_transactions: pd.DataFrame) -> None:
    model = ItemKNN.from_transactions(sample_transactions, user_col="tx_id", item_col="item_id").fit()
    with pytest.raises(ValueError, match="out of bounds"):
        model.recommend_items(99, 10)


@pytest.mark.parametrize("method", ["bm25", "tfidf", "cosine", "count"])
def test_item_knn_fused_gram_matches_old_scipy_path(method: str) -> None:
    """The new fused Rust `itemknn_gram_top_k` must reproduce the old
    `scipy gram -> eliminate_zeros -> itemknn_top_k` path exactly: same
    top-k neighbour *set* per item, with matching scores (row order is by
    index in both, since both sort by index after selecting top-k; only the
    selection among score ties for the k-th slot may differ)."""
    from rusket import _rusket as _rust
    from rusket.recommenders.item_knn import _bm25_weight, _cosine_weight, _tfidf_weight

    rng = np.random.default_rng(0)
    n_users, n_items = 12, 9
    dense = (rng.random((n_users, n_items)) > 0.55).astype(np.float64) * rng.integers(1, 5, (n_users, n_items))
    X = sp.csr_matrix(dense)
    X.eliminate_zeros()
    k = 3

    if method == "bm25":
        X_weighted = _bm25_weight(X)
    elif method == "tfidf":
        X_weighted = _tfidf_weight(X)
    elif method == "cosine":
        X_weighted = _cosine_weight(X)
    else:
        X_weighted = X

    # --- OLD path: full scipy Gram, then Rust top-k prune ---
    if method == "cosine":
        W = X_weighted.T.dot(X_weighted)
    else:
        W = X_weighted.T.dot(X)
    W = W.tocsr()
    W.eliminate_zeros()
    old_ip, old_ix, old_dt = _rust.itemknn_top_k(
        W.indptr.astype(np.int64), W.indices.astype(np.int32), W.data.astype(np.float32), k
    )

    # --- NEW path: fused Rust gram + top-k, Gram never materialised ---
    b = X_weighted if method == "cosine" else X
    b = b.tocsr()
    b.eliminate_zeros()
    new_ip, new_ix, new_dt = _rust.itemknn_gram_top_k(
        X_weighted.indptr.astype(np.int64),
        X_weighted.indices.astype(np.int32),
        X_weighted.data.astype(np.float32),
        b.indptr.astype(np.int64),
        b.indices.astype(np.int32),
        b.data.astype(np.float32),
        n_users,
        n_items,
        k,
    )

    assert len(old_ip) == len(new_ip) == n_items + 1
    for row in range(n_items):
        os_, oe_ = old_ip[row], old_ip[row + 1]
        ns_, ne_ = new_ip[row], new_ip[row + 1]
        old_pairs = dict(zip(old_ix[os_:oe_].tolist(), old_dt[os_:oe_].tolist(), strict=True))
        new_pairs = dict(zip(new_ix[ns_:ne_].tolist(), new_dt[ns_:ne_].tolist(), strict=True))
        # The two paths can legitimately choose DIFFERENT tied neighbours for
        # the k-th slot: both use select_nth_unstable, and real Gram rows have
        # exact ties (e.g. a tfidf row here has four neighbours at 33.8629 for
        # the 3rd slot). So the invariant is the selected SCORES, not the ids.
        assert sorted(new_pairs.values()) == pytest.approx(sorted(old_pairs.values()), rel=1e-5, abs=1e-6), (
            f"row {row} selected score multiset mismatch"
        )
        # and every id chosen must really be one of the tied best
        if old_pairs:
            cutoff = min(old_pairs.values())
            # relative slack: the fused path accumulates in a different order
            # than scipy's gram, so equal-in-exact-arithmetic scores can differ
            # in the last f32 ulps.
            slack = 1e-4 * max(abs(cutoff), 1.0)
            assert all(v >= cutoff - slack for v in new_pairs.values()), f"row {row} selected a worse-than-cutoff neighbour"
        # Any id present in BOTH must agree on its score.
        for idx in set(old_pairs) & set(new_pairs):
            assert new_pairs[idx] == pytest.approx(old_pairs[idx], rel=1e-4, abs=1e-5), (
                f"row {row} item {idx} score mismatch"
            )
