import pandas as pd

from rusket import leave_one_out_split, train_test_split


def test_train_test_split():
    df = pd.DataFrame({
        "user": [1, 1, 1, 2, 2, 3, 3, 3, 3, 4],
        "item": [10, 20, 30, 10, 40, 50, 60, 70, 80, 90]
    })

    train_df, test_df = train_test_split(df, "user", "item", test_size=0.3)

    assert len(train_df) == 7
    assert len(test_df) == 3
    assert list(train_df.columns) == ["user", "item"]


def test_leave_one_out_split_random():
    df = pd.DataFrame({
        "user": [1, 1, 1, 2, 2, 3, 4],
        "item": [10, 20, 30, 10, 40, 50, 90]
    })

    train_df, test_df = leave_one_out_split(df, "user", "item")

    # Users 1 and 2 have >1 item, so they will hold out 1
    # User 3 and 4 have 1 item, so they will hold out 0 (retained in train)
    assert len(test_df) == 2
    assert set(test_df["user"].unique()) == {1, 2}

    assert len(train_df) == 5
    assert set(train_df["user"].unique()) == {1, 2, 3, 4}


def test_leave_one_out_split_temporal():
    df = pd.DataFrame({
        "user": [1, 1, 1, 2, 2],
        "item": [10, 20, 30, 10, 40],
        "ts": [1.0, 3.0, 2.0, 10.0, 20.0]
    })

    train_df, test_df = leave_one_out_split(df, "user", "item", timestamp_col="ts")

    assert len(test_df) == 2

    # For user 1, max ts is 3.0 (item 20)
    user1_test = test_df[test_df["user"] == 1]
    assert user1_test["item"].iloc[0] == 20

    # For user 2, max ts is 20.0 (item 40)
    user2_test = test_df[test_df["user"] == 2]
    assert user2_test["item"].iloc[0] == 40
