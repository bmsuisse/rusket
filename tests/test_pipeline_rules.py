import pandas as pd
import pytest

from rusket import ALS, ItemKNN, Pipeline, RuleBasedRecommender


@pytest.fixture
def transactions():
    return pd.DataFrame(
        {
            "user": [1, 1, 1, 2, 2, 3],
            "item": ["A", "B", "C", "A", "D", "E"],
            "rating": [1, 1, 1, 1, 1, 1],
        }
    )


def test_pipeline_with_rules(transactions):
    als = (
        ALS(factors=2, iterations=1, random_state=42)
        .from_transactions(transactions, user_col="user", item_col="item")
        .fit()
    )

    knn = ItemKNN(k=5).from_transactions(transactions, user_col="user", item_col="item").fit()

    # Rules say A strongly implies D. User 1 has interacted with A, B, C.
    rules = pd.DataFrame(
        {
            "antecedent": ["A", "B"],
            "consequent": ["D", "E"],
            "score": [2.0, 1.5],
        }
    )

    rule_model = RuleBasedRecommender.from_transactions(
        transactions, rules=rules, user_col="user", item_col="item"
    ).fit()

    pipeline = Pipeline(
        retrieve=[als, knn],
        rerank=als,  # Re-score candidates with ALS
        rules=rule_model,  # Inject B->Y, A->X
    )

    assert rule_model._user_labels is not None
    assert rule_model.item_names is not None

    user_id = rule_model._user_labels.index(1)

    # Run pipeline for a single user
    ids, scores = pipeline.recommend(user_id=user_id, n=3, exclude_seen=True)
    recs = [rule_model.item_names[i] for i in ids]

    # Because scores are boosted by 1,000,000, D and E should be the top recommendations.
    assert "D" in recs
    assert "E" in recs
    # D score (2.0) > E score (1.5), so D should be first
    assert recs[0] == "D"
    assert recs[1] == "E"
    assert scores[0] >= 1_000_000.0


def test_pipeline_batch_with_rules(transactions):
    als = (
        ALS(factors=2, iterations=1, random_state=42)
        .from_transactions(transactions, user_col="user", item_col="item")
        .fit()
    )

    knn = ItemKNN(k=5).from_transactions(transactions, user_col="user", item_col="item").fit()

    rules = pd.DataFrame(
        {
            "antecedent": ["E"],
            "consequent": ["C"],
            "score": [5.0],
        }
    )

    rule_model = RuleBasedRecommender.from_transactions(
        transactions, rules=rules, user_col="user", item_col="item"
    ).fit()

    pipeline = Pipeline(
        retrieve=[als, knn],
        rules=rule_model,
    )

    # User 3 has interacted with E. They should get C recommended.
    # User 1 hasn't interacted with E. They shouldn't get C.

    batch = pipeline.recommend_batch(n=2)

    df_res = pd.DataFrame(batch)

    assert rule_model.item_names is not None

    # Map back the first suggestion for each user
    df_res["top_rec"] = df_res["item_ids"].apply(
        lambda x: rule_model.item_names[x[0]] if len(x) > 0 else None  # type: ignore[index]
    )

    assert rule_model._user_labels is not None

    # For user 3 (label=3), top_rec should be C
    assert df_res.loc[df_res["user_id"] == rule_model._user_labels.index(3), "top_rec"].iloc[0] == "C"
