import numpy as np
import pandas as pd
import pytest

from rusket import ContentBased


def _make_items_df() -> pd.DataFrame:
    """Small item catalogue with text descriptions."""
    return pd.DataFrame(
        {
            "item_id": ["A", "B", "C", "D"],
            "description": [
                "action adventure movie with explosions and car chases",
                "action thriller movie with guns and espionage",
                "romantic comedy film about love and relationships",
                "romantic drama film about love and family",
            ],
        }
    )


def test_content_based_fit() -> None:
    df = _make_items_df()
    model = ContentBased.from_dataframe(df, item_col="item_id", text_col="description").fit()

    assert model.fitted
    assert model.similarity_matrix.shape == (4, 4)
    # Diagonal should be zero (item doesn't recommend itself)
    np.testing.assert_array_equal(np.diag(model.similarity_matrix), np.zeros(4))


def test_content_based_recommend_similar() -> None:
    df = _make_items_df()
    model = ContentBased.from_dataframe(df, item_col="item_id", text_col="description").fit()

    # Item A (action) should be most similar to Item B (also action)
    ids, scores = model.recommend_similar("A", n=1)
    assert len(ids) == 1
    assert ids[0] == "B"
    assert scores[0] > 0


def test_content_based_recommend_romantic() -> None:
    df = _make_items_df()
    model = ContentBased.from_dataframe(df, item_col="item_id", text_col="description").fit()

    # Item C (romantic comedy) should be most similar to Item D (romantic drama)
    ids, scores = model.recommend_similar("C", n=1)
    assert ids[0] == "D"


def test_content_based_unknown_item() -> None:
    df = _make_items_df()
    model = ContentBased.from_dataframe(df, item_col="item_id", text_col="description").fit()

    with pytest.raises(ValueError, match="not found"):
        model.recommend_similar("UNKNOWN")


def test_content_based_unfitted_raises() -> None:
    df = _make_items_df()
    model = ContentBased.from_dataframe(df, item_col="item_id", text_col="description")

    with pytest.raises(RuntimeError):
        model.recommend_similar("A")


def test_content_based_already_fitted_raises() -> None:
    df = _make_items_df()
    model = ContentBased.from_dataframe(df, item_col="item_id", text_col="description").fit()

    with pytest.raises(RuntimeError):
        model.fit()


def test_content_based_from_transactions_raises() -> None:
    with pytest.raises(NotImplementedError):
        ContentBased.from_transactions(pd.DataFrame())


def test_content_based_max_features() -> None:
    df = _make_items_df()
    model = ContentBased.from_dataframe(df, item_col="item_id", text_col="description", max_features=10).fit()

    assert model.fitted
    ids, scores = model.recommend_similar("A", n=3)
    assert len(ids) == 3
    # Scores should be in descending order
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
