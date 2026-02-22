import numpy as np

from .typing import SupportsItemFactors


def similar_items(
    model: SupportsItemFactors,
    item_id: int,
    n: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the most similar items to a given item ID based on latent factors.

    Computes cosine similarity between the specified item's latent vector
    and all other item vectors in the ``item_factors`` matrix.

    Parameters
    ----------
    model : SupportsItemFactors
        A fitted model instance with an ``item_factors`` property.
    item_id : int
        The internal integer index of the target item.
    n : int
        Number of most similar items to return.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(item_ids, cosine_similarities)`` sorted in descending order.
    """
    factors = model.item_factors  # raises if not fitted
    n_items, _ = factors.shape

    if item_id < 0 or item_id >= n_items:
        raise ValueError(f"item_id {item_id} out of bounds for {n_items} items.")

    target = factors[item_id]
    target_norm = np.linalg.norm(target)

    if target_norm == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    scores = np.dot(factors, target)
    norms = np.linalg.norm(factors, axis=1)

    valid = norms > 0
    scores[valid] = scores[valid] / (norms[valid] * target_norm)
    scores[~valid] = 0.0
    scores[item_id] = -np.inf

    n = min(n, n_items - 1)
    if n <= 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    top_indices = np.argpartition(scores, -n)[-n:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return top_indices.astype(np.int32), scores[top_indices].astype(np.float32)
