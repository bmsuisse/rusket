import pytest
import numpy as np
from scipy.sparse import csr_matrix
from rusket import BPR, ALS, ItemKNN, FPMC, FM
import yaml
import os
import shutil

# Trying to mimic the config-based testing seen in RecBole's tests
# RecBole has a `quick_test` function that initializes models with a dictionary.
# Let's write similar wrapper test functions for the models we implemented in Rusket.


@pytest.fixture
def mock_interactions():
    # Simple explicit interactions matrix
    # Users: 4, Items: 5
    # 1 indicates interaction
    data = np.array([[1, 0, 1, 0, 0], [0, 1, 0, 0, 1], [1, 1, 0, 1, 0], [0, 0, 1, 0, 1]], dtype=np.float32)
    return csr_matrix(data)


@pytest.fixture
def mock_sequences():
    # Simple list of sequences for sequential models like FPMC
    return [[0, 2, 0, 2], [1, 4, 1], [0, 1, 3], [2, 4]]


@pytest.fixture
def mock_context_data():
    # Context data for FM
    # Interactions represented as binary features
    # Columns: [user_0, user_1, item_0, item_1, context_0, context_1]
    X = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [1, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1]], dtype=np.float32)
    y = np.array([1.0, 0.0, 1.0, 0.0])
    return csr_matrix(X), y


def test_recbole_bpr_equivalent(mock_interactions):
    """Equivalent to TestGeneralRecommender.test_bpr"""
    model = BPR(factors=10, iterations=10, seed=42)
    model.fit(mock_interactions)
    assert model.fitted

    recs, scores = model.recommend_items(0, n=2, exclude_seen=True)
    assert len(recs) == 2


def test_recbole_itemknn_equivalent(mock_interactions):
    """Equivalent to TestGeneralRecommender.test_itemknn"""
    model = ItemKNN(k=2, shrinkage=0.5)
    model.fit(mock_interactions)
    assert model.fitted

    recs, scores = model.recommend_items(0, n=2, exclude_seen=True)
    assert len(recs) == 2


def test_recbole_fpmc_equivalent(mock_sequences):
    """Equivalent to TestSequentialRecommender.test_fpmc"""
    model = FPMC(factors=10, iterations=10)
    model.fit(mock_sequences, n_items=5)
    assert model.fitted

    recs, scores = model.recommend_items(0, n=2, exclude_seen=False)
    assert len(recs) == 2


def test_recbole_fm_equivalent(mock_context_data):
    """Equivalent to TestContextRecommender.test_fm"""
    X, y = mock_context_data
    model = FM(factors=4, iterations=10)
    model.fit(X, y)
    assert model.fitted

    preds = model.predict_proba(X)
    assert len(preds) == 4
    assert np.all(preds >= 0) and np.all(preds <= 1)
