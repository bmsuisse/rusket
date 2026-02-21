import numpy as np
import pytest
from rusket import ALS, similar_items

def test_similar_items_basic():
    """Test similarities using a manual ALS model mockup."""
    als = ALS()
    # Mock item factors (4 items, 2 factors)
    als._item_factors = np.array([
        [1.0, 0.0],  # item 0
        [0.8, 0.2],  # item 1 (similar to 0)
        [0.0, 1.0],  # item 2 (orthogonal to 0)
        [-1.0, 0.0], # item 3 (opposite to 0)
    ], dtype=np.float32)
    als._user_factors = np.ones((1, 1), dtype=np.float32)
    als.fitted = True
    
    # Test item 0
    top_items, scores = similar_items(als, item_id=0, n=2)
    
    assert len(top_items) == 2
    assert top_items[0] == 1
    assert top_items[1] == 2
    
    # Cosine similar to itself would be 1.0, but we exclude it.
    # similarity between [1, 0] and [0.8, 0.2] is 0.8 / sqrt(0.64 + 0.04)
    # Norm of [0.8, 0.2] is sqrt(0.68) ~ 0.8246
    # 0.8 / 0.8246 ~ 0.97
    assert scores[0] > 0.9
    assert scores[1] == 0.0  # orthogonal
    
def test_similar_items_out_of_bounds():
    als = ALS()
    als._item_factors = np.ones((2, 2))
    als._user_factors = np.ones((1, 1), dtype=np.float32)
    als.fitted = True
    with pytest.raises(ValueError):
        similar_items(als, item_id=2, n=1)
        
def test_similar_items_n_greater_than_items():
    als = ALS()
    als._item_factors = np.array([
        [1.0, 0.0],
        [0.8, 0.2],
        [0.0, 1.0],
    ], dtype=np.float32)
    als._user_factors = np.ones((1, 1), dtype=np.float32)
    als.fitted = True
    
    # We ask for 10 items, but only 2 others are available
    top_items, _ = similar_items(als, item_id=0, n=10)
    assert len(top_items) == 2
