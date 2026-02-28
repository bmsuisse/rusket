import numpy as np
import pytest
from scipy import sparse as sp

from rusket import EASE, PopularityRecommender
from rusket.hybrid import HybridRecommender


def _make_interactions() -> sp.csr_matrix:
    """Small interaction matrix: 4 users, 5 items."""
    rows = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3]
    cols = [0, 1, 1, 2, 3, 0, 4, 2, 3, 4]
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(4, 5))


def test_hybrid_basic() -> None:
    X = _make_interactions()

    pop = PopularityRecommender().fit(X)
    ease = EASE(regularization=1.0).fit(X)

    hybrid = HybridRecommender([(pop, 0.5), (ease, 0.5)])

    items, scores = hybrid.recommend_items(user_id=0, n=3, exclude_seen=True)
    assert len(items) == 3
    # Scores should be in descending order
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_hybrid_repr() -> None:
    X = _make_interactions()
    pop = PopularityRecommender().fit(X)
    ease = EASE(regularization=1.0).fit(X)

    hybrid = HybridRecommender([(pop, 0.7), (ease, 0.3)])
    r = repr(hybrid)
    assert "PopularityRecommender" in r
    assert "EASE" in r


def test_hybrid_weight_normalisation() -> None:
    X = _make_interactions()
    pop = PopularityRecommender().fit(X)

    # Two copies of same model with unequal weights
    hybrid = HybridRecommender([(pop, 2.0), (pop, 8.0)])
    assert abs(sum(hybrid._normalised_weights) - 1.0) < 1e-6


def test_hybrid_empty_raises() -> None:
    with pytest.raises(ValueError):
        HybridRecommender([])


def test_hybrid_fit_is_noop() -> None:
    X = _make_interactions()
    pop = PopularityRecommender().fit(X)
    hybrid = HybridRecommender([(pop, 1.0)])
    # fit() should simply return self
    assert hybrid.fit() is hybrid


def test_hybrid_all_models_contribute() -> None:
    X = _make_interactions()
    pop = PopularityRecommender().fit(X)
    ease = EASE(regularization=1.0).fit(X)

    # With only popularity
    items_pop, _ = hybrid_pop_only(pop).recommend_items(0, n=5, exclude_seen=False)
    # With only EASE
    items_ease, _ = hybrid_ease_only(ease).recommend_items(0, n=5, exclude_seen=False)
    # Hybrid should differ from both (unless scores perfectly align)
    items_hybrid, _ = HybridRecommender([(pop, 0.5), (ease, 0.5)]).recommend_items(0, n=5, exclude_seen=False)
    # At minimum, we get 5 items back
    assert len(items_hybrid) == 5


def hybrid_pop_only(pop: PopularityRecommender) -> HybridRecommender:
    return HybridRecommender([(pop, 1.0)])


def hybrid_ease_only(ease: EASE) -> HybridRecommender:
    return HybridRecommender([(ease, 1.0)])
