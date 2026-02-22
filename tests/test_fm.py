import numpy as np
import pytest
from scipy import sparse

from rusket import FM


def test_fm_fit_predict():
    # Simple XOR-like logical problem to test if FM can capture interactions
    # y = x0 XOR x1
    # Features: [x0, x1]
    # We will expand it slightly: [x0, x1, user_id, item_id]

    # Let's just create a sparse matrix
    X = np.array(
        [
            [1, 0, 1],  # user 1, item 0, cat 1 -> positive
            [0, 1, 0],  # user 0, item 1, cat 0 -> negative
            [1, 1, 0],  # user 1, item 1, cat 0 -> positive
            [0, 0, 1],  # user 0, item 0, cat 1 -> negative
        ]
    )
    y = np.array([1.0, 0.0, 1.0, 0.0])

    # Make it sparse
    X_sparse = sparse.csr_matrix(X)

    model = FM(factors=4, iterations=50, learning_rate=0.1, seed=42)
    model.fit(X_sparse, y)

    assert model.fitted
    assert model.w0_ is not None
    assert model.w_ is not None
    assert model.v_ is not None

    # Predict
    preds = model.predict_proba(X_sparse)

    assert len(preds) == 4
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)

    # Predictions should somewhat match the labels
    # Not strictly enforcing exact match because 50 iters might not perfectly converge,
    # but the direction should be correct.
    assert preds[0] > 0.5
    assert preds[1] < 0.5
    assert preds[2] > 0.5
    assert preds[3] < 0.5


def test_fm_invalid_inputs():
    model = FM()

    with pytest.raises(TypeError):
        model.fit("not a matrix", [1, 0])  # type: ignore

    # Mismatched shapes
    X = np.array([[1, 0], [0, 1]])
    y = np.array([1.0, 0.0, 1.0])  # 3 labels, 2 samples

    with pytest.raises(ValueError):
        model.fit(X, y)
