import pandas as pd

from rusket.analytics import customer_saturation, find_substitutes


def test_find_substitutes():
    # Mocking association rules dataframe where A and B cannibalize
    data = {
        "antecedents": [{"A"}, {"C"}],
        "consequents": [{"B"}, {"D"}],
        "support": [0.05, 0.4],
        "confidence": [0.1, 0.8],
        "lift": [0.4, 2.5],  # 0.4 indicates negative correlation (cannibalization)
    }

    rules_df = pd.DataFrame(data)

    subs = find_substitutes(rules_df, max_lift=0.8)

    assert len(subs) == 1
    assert list(subs["antecedents"])[0] == {"A"}
    assert list(subs["consequents"])[0] == {"B"}
    assert subs.iloc[0]["lift"] == 0.4


def test_customer_saturation():
    # User 1 buys all 5 categories
    # User 2 buys 2
    # User 3 buys 1
    # User 4-10 buy 1 to populate deciles
    data = {
        "user_id": [1, 1, 1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "cat_id": [
            "A",
            "B",
            "C",
            "D",
            "E",
            "A",
            "C",
            "A",
            "B",
            "A",
            "C",
            "D",
            "E",
            "A",
            "B",
        ],
    }

    df = pd.DataFrame(data)

    sat_df = customer_saturation(df, user_col="user_id", category_col="cat_id")

    assert len(sat_df) == 10

    u1_stat = sat_df[sat_df["user_id"] == 1].iloc[0]
    assert u1_stat["unique_count"] == 5
    assert u1_stat["saturation_pct"] == 1.0
    assert u1_stat["decile"] == 1  # Top decile

    u10_stat = sat_df[sat_df["user_id"] == 10].iloc[0]
    assert u10_stat["unique_count"] == 1
    # u10 is the very last user tested, so it gets the worst rank
    assert u10_stat["decile"] == 10
