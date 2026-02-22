import pytest

from rusket import FPMC


def test_fpmc_fit_predict():
    sequences = [[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6], [1, 2]]

    model = FPMC(factors=8, iterations=20, seed=42)
    model.fit(sequences)

    # Check that model is fitted
    assert model.fitted
    assert model._vu is not None
    assert model._viu is not None
    assert model._n_users == 3
    assert model._n_items == 7  # max item is 6, so 7 items total (0 to 6)

    # Recommend for user 0
    recs, scores = model.recommend_items(0, n=3, exclude_seen=False)

    assert len(recs) == 3
    assert len(scores) == 3

    # Recommend for user 1
    recs1, scores1 = model.recommend_items(1, n=3, exclude_seen=False)

    assert len(recs1) == 3
    assert len(scores1) == 3


def test_fpmc_invalid_inputs():
    model = FPMC()

    with pytest.raises(TypeError):
        model.fit("not a list of lists")  # type: ignore

    with pytest.raises(TypeError):
        model.fit([1, 2, 3])  # type: ignore
