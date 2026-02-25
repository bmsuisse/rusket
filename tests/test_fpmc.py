import numpy as np
import pytest

from rusket import FPMC


@pytest.fixture
def simple_sequences():
    return [[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6], [1, 2]]


def test_fpmc_fit_predict(simple_sequences):
    model = FPMC(factors=8, iterations=20, seed=42)
    model.fit(simple_sequences)

    assert model.fitted
    assert model._vu is not None
    assert model._viu is not None
    assert model._n_users == 3
    assert model._n_items == 7  # max item is 6, so 7 items total (0 to 6)

    recs, scores = model.recommend_items(0, n=3, exclude_seen=False)
    assert len(recs) == 3
    assert len(scores) == 3

    recs1, scores1 = model.recommend_items(1, n=3, exclude_seen=False)
    assert len(recs1) == 3
    assert len(scores1) == 3


def test_fpmc_exclude_seen(simple_sequences):
    """Excluding seen items should return items not in the user's history (for valid scores)."""
    model = FPMC(factors=8, iterations=20, seed=42)
    model.fit(simple_sequences)

    recs, scores = model.recommend_items(0, n=4, exclude_seen=True)
    import numpy as np

    seen_items = {1, 2, 3}
    # Only check items with finite scores (argpartition may include -inf tail)
    valid = scores > -np.inf
    for item in recs[valid]:
        assert item not in seen_items, f"Seen item {item} should be excluded"


def test_fpmc_repr():
    model = FPMC(factors=32, iterations=100, learning_rate=0.01, regularization=0.05)
    r = repr(model)
    assert "FPMC(" in r
    assert "factors=32" in r


def test_fpmc_double_fit(simple_sequences):
    """Fitting twice should raise RuntimeError."""
    model = FPMC(factors=8, iterations=10, seed=42)
    model.fit(simple_sequences)
    with pytest.raises(RuntimeError):
        model.fit(simple_sequences)


def test_fpmc_predict_before_fit():
    """Recommending before fit should raise RuntimeError."""
    model = FPMC()
    with pytest.raises(RuntimeError):
        model.recommend_items(0, n=3)


def test_fpmc_user_out_of_bounds(simple_sequences):
    """Out-of-bounds user_id should raise ValueError."""
    model = FPMC(factors=8, iterations=10, seed=42)
    model.fit(simple_sequences)

    with pytest.raises(ValueError, match="out of bounds"):
        model.recommend_items(99, n=3)

    with pytest.raises(ValueError, match="out of bounds"):
        model.recommend_items(-1, n=3)


def test_fpmc_invalid_inputs():
    model = FPMC()

    with pytest.raises(TypeError):
        model.fit("not a list of lists")  # type: ignore

    with pytest.raises(TypeError):
        model.fit([1, 2, 3])  # type: ignore


def test_fpmc_single_item_sequences():
    """Users with a single item should still work."""
    seqs = [[5], [6], [7]]
    model = FPMC(factors=4, iterations=10, seed=42)
    model.fit(seqs)

    recs, scores = model.recommend_items(0, n=2, exclude_seen=True)
    assert len(recs) <= 2  # May have fewer than 2 unseen items
    assert all(r != 5 for r in recs)


def test_fpmc_scores_sorted(simple_sequences):
    """Scores should be returned in descending order."""
    model = FPMC(factors=8, iterations=20, seed=42)
    model.fit(simple_sequences)

    _, scores = model.recommend_items(0, n=5, exclude_seen=False)
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], "Scores should be in descending order"


def test_fpmc_larger_dataset():
    """Test FPMC with a moderately sized dataset."""
    rng = np.random.default_rng(42)
    n_users = 50
    sequences = [rng.integers(0, 30, size=rng.integers(3, 15)).tolist() for _ in range(n_users)]

    model = FPMC(factors=16, iterations=30, seed=42)
    model.fit(sequences)

    assert model.fitted
    for u in range(min(5, n_users)):
        recs, scores = model.recommend_items(u, n=5)
        assert len(recs) <= 5
