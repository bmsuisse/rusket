import numpy as np
from .als import ALS

def similar_items(
    als_model: ALS,
    item_id: int,
    n: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the most similar items to a given item ID based on ALS latent factors.

    Computes the Cosine Similarity between the specified item's latent vector
    and all other item vectors in the `item_factors` matrix.

    Parameters
    ----------
    als_model : ALS
        A fitted `rusket.ALS` model instance.
    item_id : int
        The internal integer index of the target item.
    n : int
        Number of most similar items to return.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of (item_ids, cosine_similarities) sorted in descending order of similarity.
    """
    if als_model.item_factors is None:
        raise ValueError("Model has not been fitted yet.")
        
    factors = als_model.item_factors
    n_items, _ = factors.shape
    
    if item_id < 0 or item_id >= n_items:
        raise ValueError(f"item_id {item_id} out of bounds for {n_items} items.")
        
    target_vector = factors[item_id]
    target_norm = np.linalg.norm(target_vector)
    
    if target_norm == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        
    # Dot products between all items and target
    scores = np.dot(factors, target_vector)
    
    # L2 Norms of all items
    norms = np.linalg.norm(factors, axis=1)
    
    # Compute Cosine similarity (avoiding division by zero)
    valid = norms > 0
    scores[valid] = scores[valid] / (norms[valid] * target_norm)
    scores[~valid] = 0.0
    
    # Exclude the target item itself
    scores[item_id] = -np.inf
    
    # Determine how many items to return
    if n >= n_items:
        n = n_items - 1
        
    if n <= 0:
         return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
         
    # Fast top-N selection using argpartition (O(N) time)
    top_indices = np.argpartition(scores, -n)[-n:]
    
    # Sort the top N elements in descending order
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    
    return top_indices.astype(np.int32), scores[top_indices].astype(np.float32)
