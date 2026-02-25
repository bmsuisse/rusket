import numpy as np
import pytest
from scipy import sparse

from rusket import FM


@pytest.fixture
def simple_data():
    X = np.array(
        [
            [1, 0, 1],  # user 1, item 0, cat 1 -> positive
            [0, 1, 0],  # user 0, item 1, cat 0 -> negative
            [1, 1, 0],  # user 1, item 1, cat 0 -> positive
            [0, 0, 1],  # user 0, item 0, cat 1 -> negative
        ]
    )
    y = np.array([1.0, 0.0, 1.0, 0.0])
    return sparse.csr_matrix(X), y


def test_fm_fit_predict(simple_data):
    X_sparse, y = simple_data
    model = FM(factors=4, iterations=50, learning_rate=0.1, seed=42)
    model.fit(X_sparse, y)

    assert model.fitted
    assert model.w0_ is not None
    assert model.w_ is not None
    assert model.v_ is not None

    preds = model.predict_proba(X_sparse)

    assert len(preds) == 4
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)

    # Predictions should somewhat match the labels
    assert preds[0] > 0.5
    assert preds[1] < 0.5
    assert preds[2] > 0.5
    assert preds[3] < 0.5


def test_fm_predict_alias(simple_data):
    """predict() should be an alias for predict_proba()."""
    X_sparse, y = simple_data
    model = FM(factors=4, iterations=50, learning_rate=0.1, seed=42)
    model.fit(X_sparse, y)

    proba = model.predict_proba(X_sparse)
    alias = model.predict(X_sparse)
    np.testing.assert_array_equal(proba, alias)


def test_fm_repr():
    model = FM(factors=8, iterations=100, learning_rate=0.01, regularization=0.02)
    r = repr(model)
    assert "FM(" in r
    assert "factors=8" in r
    assert "iterations=100" in r


def test_fm_double_fit(simple_data):
    """Fitting twice should raise RuntimeError."""
    X_sparse, y = simple_data
    model = FM(factors=4, iterations=10, seed=42)
    model.fit(X_sparse, y)
    with pytest.raises(RuntimeError):
        model.fit(X_sparse, y)


def test_fm_predict_before_fit():
    """predict before fit should raise RuntimeError."""
    model = FM()
    X = sparse.csr_matrix(np.eye(3))
    with pytest.raises(RuntimeError):
        model.predict_proba(X)


def test_fm_feature_mismatch(simple_data):
    """Predicting with wrong number of features should raise ValueError."""
    X_sparse, y = simple_data
    model = FM(factors=4, iterations=10, seed=42)
    model.fit(X_sparse, y)

    X_wrong = sparse.csr_matrix(np.eye(5))  # 5 features != 3
    with pytest.raises(ValueError, match="features"):
        model.predict_proba(X_wrong)


def test_fm_from_transactions():
    """FM does not support from_transactions — should raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        FM.from_transactions([["a"], ["b"]])


def test_fm_invalid_inputs():
    model = FM()

    with pytest.raises(TypeError):
        model.fit("not a matrix", [1, 0])  # type: ignore

    # Mismatched shapes
    X = np.array([[1, 0], [0, 1]])
    y = np.array([1.0, 0.0, 1.0])  # 3 labels, 2 samples

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_fm_dense_input():
    """FM should also accept dense numpy arrays."""
    X = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
    y = np.array([1.0, 0.0, 1.0, 0.0])

    model = FM(factors=4, iterations=50, learning_rate=0.1, seed=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 4
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)


def test_fm_larger_dataset():
    """Test FM with a larger synthetic dataset."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 200, 20
    X = rng.integers(0, 2, size=(n_samples, n_features)).astype(np.float32)
    # Simple rule: if sum of first 3 features >= 2, positive
    y = (X[:, :3].sum(axis=1) >= 2).astype(np.float32)

    X_sparse = sparse.csr_matrix(X)
    model = FM(factors=8, iterations=100, learning_rate=0.05, seed=42)
    model.fit(X_sparse, y)

    preds = model.predict_proba(X_sparse)
    # Check basic convergence: accuracy should be well above random (>60%)
    predicted_labels = (preds > 0.5).astype(float)
    accuracy = np.mean(predicted_labels == y)
    assert accuracy > 0.6, f"FM accuracy on training data too low: {accuracy:.2%}"
