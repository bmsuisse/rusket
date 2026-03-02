import pandas as pd
import pytest

from rusket import ALS, HybridRecommender, RuleBasedRecommender


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "user": [1, 1, 2, 2, 3],
            "item": ["A", "B", "C", "D", "E"],
            "rating": [1, 1, 1, 1, 1],
        }
    )


def test_rule_based_recommender_dict(sample_data):
    # A -> C, D. C -> E
    rules = {
        "A": ["C", "D"],
        "C": ["E"],
    }

    model = RuleBasedRecommender.from_transactions(sample_data, rules=rules, user_col="user", item_col="item").fit()

    assert model._user_labels is not None
    assert model.item_names is not None

    # Find internal user 1 index (user 1 in our frame corresponds to internal id 0 usually)
    # Actually, we can use the labels to be robust
    user_id = model._user_labels.index(1)  # user 1

    ids, scores = model.recommend_items(user_id=user_id, n=2, exclude_seen=True)
    recs = [model.item_names[i] for i in ids]

    # User 1 has interacted with "A" and "B".
    # Since "A" maps to "C" and "D" with score 1.0, user 1 should be recommended C and D.
    assert "C" in recs
    assert "D" in recs
    assert list(scores) == [1.0, 1.0]


def test_rule_based_recommender_dataframe(sample_data):
    rules_df = pd.DataFrame(
        {
            "antecedent": ["A", "A", "C"],
            "consequent": ["C", "D", "E"],
            "score": [2.0, 1.5, 3.0],
        }
    )

    model = RuleBasedRecommender.from_transactions(sample_data, rules=rules_df, user_col="user", item_col="item").fit()

    user_id = model._user_labels.index(1)

    ids, scores = model.recommend_items(user_id=user_id, n=2, exclude_seen=True)
    recs = [model.item_names[i] for i in ids]

    # "A" maps to C (2.0) and D (1.5)
    assert recs[0] == "C"
    assert recs[1] == "D"
    assert scores[0] == 2.0
    assert scores[1] == 1.5


def test_rule_based_hybrid_integration(sample_data):
    rules = {"A": ["E"]}

    rule_model = RuleBasedRecommender.from_transactions(
        sample_data, rules=rules, user_col="user", item_col="item"
    ).fit()

    als_model = ALS.from_transactions(
        sample_data, factors=2, iterations=1, random_state=42, user_col="user", item_col="item"
    ).fit()

    hybrid = HybridRecommender(
        [
            (als_model, 0.5),
            (rule_model, 0.5),
        ]
    )

    assert rule_model._user_labels is not None
    assert rule_model.item_names is not None

    user_id = rule_model._user_labels.index(1)

    ids, scores = hybrid.recommend_items(user_id=user_id, n=3, exclude_seen=True)
    recs = [rule_model.item_names[i] for i in ids]

    # User 1 (A, B) -> Rule says E is strongly recommended.
    # Because rule_model gives E a score of 1.0 (weight * 0.5 = 0.5), E should appear in the top recs.
    assert "E" in recs


def test_rule_based_unknown_items(sample_data):
    # Tests that unknown items are ignored rather than causing errors.
    rules = {
        "A": ["UNKNOWN_ITEM_X", "C"],
        "UNKNOWN_ITEM_Y": ["B"],
    }

    model = RuleBasedRecommender.from_transactions(sample_data, rules=rules, user_col="user", item_col="item").fit()

    user_id = model._user_labels.index(1)
    ids, scores = model.recommend_items(user_id=user_id, n=2, exclude_seen=True)
    recs = [model.item_names[i] for i in ids]

    # Since UNKNOWN_ITEM_X and Y mapped to nothing/were ignored, "A" -> "C" is the only valid rule.
    assert "C" in recs
    assert len(recs) == 1
    assert list(scores) == [1.0]


def test_missing_score_column(sample_data):
    # DF missing a score column should fallback to default_score
    rules_df = pd.DataFrame(
        {
            "antecedent": ["A"],
            "consequent": ["D"],
        }
    )

    model = RuleBasedRecommender.from_transactions(
        sample_data, rules=rules_df, default_score=4.2, user_col="user", item_col="item"
    ).fit()

    user_id = model._user_labels.index(1)
    ids, scores = model.recommend_items(user_id=user_id, n=1, exclude_seen=True)
    recs = [model.item_names[i] for i in ids]

    assert recs == ["D"]
    assert scores[0] == 4.2
