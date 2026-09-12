"""Regression tests for BUG 1: models pickled by an earlier release (raw
scipy dtypes on ``_fit_*``/``w_*`` attributes) must still work after
``load_model()``/``.load()``, even though ``fit()`` now casts dtypes once
up front instead of on every ``recommend_items()`` call.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from rusket import ItemKNN, UserKNN, load_model
from rusket.recommenders.rules import RuleBasedRecommender


def _make_interactions() -> sp.csr_matrix:
    data = np.ones(8)
    row = np.array([0, 0, 1, 1, 2, 2, 2, 3])
    col = np.array([0, 1, 0, 2, 0, 1, 2, 1])
    return sp.csr_matrix((data, (row, col)), shape=(4, 3))


def _downgrade_dtypes(model) -> None:
    """Simulate a model pickled under <= v0.1.97: raw scipy dtypes instead
    of the exact i64/i32/f32 Rust now requires."""
    model._fit_indptr = model._fit_indptr.astype(np.int32)
    model._fit_indices = model._fit_indices.astype(np.int64)
    model._fit_data = model._fit_data.astype(np.float64)
    model.w_indptr = model.w_indptr.astype(np.int32)
    model.w_indices = model.w_indices.astype(np.int64)
    model.w_data = model.w_data.astype(np.float64)


@pytest.mark.parametrize(
    "make_model",
    [
        lambda X: ItemKNN(method="bm25", k=2).fit(X),
        lambda X: UserKNN(method="cosine", k=2).fit(X),
        lambda X: RuleBasedRecommender(rules={0: [1]}).fit(X),
    ],
    ids=["item_knn", "user_knn", "rule_based"],
)
def test_legacy_dtype_model_recommend_items_after_load(make_model) -> None:
    X = _make_interactions()
    model = make_model(X)
    _downgrade_dtypes(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.pkl"
        model.save(path)

        loaded = load_model(path)

        # Before the fix this raises TypeError from the Rust extension
        # because the restored attributes keep the wrong (legacy) dtypes.
        ids, scores = loaded.recommend_items(0, n=2)
        assert len(ids) == len(scores)

        # Dtypes should have been normalized once, at load time.
        assert loaded._fit_indptr.dtype == np.int64
        assert loaded._fit_indices.dtype == np.int32
        assert loaded._fit_data.dtype == np.float32
        assert loaded.w_indptr.dtype == np.int64
        assert loaded.w_indices.dtype == np.int32
        assert loaded.w_data.dtype == np.float32


def test_legacy_plain_pickle_object_normalized(tmp_path) -> None:
    """The 'legacy: plain pickled object' branch in load_model()/.load()
    must also normalize dtypes."""
    import pickle

    X = _make_interactions()
    model = ItemKNN(method="bm25", k=2).fit(X)
    _downgrade_dtypes(model)

    path = tmp_path / "legacy.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)

    loaded = load_model(path)
    ids, scores = loaded.recommend_items(0, n=2)
    assert len(ids) == len(scores)
    assert loaded._fit_data.dtype == np.float32
