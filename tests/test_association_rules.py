"""Association rules tests – adapted from mlxtend/tests/test_association_rules.py."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_raises as numpy_assert_raises

from rusket import fpgrowth, association_rules

# ---------------------------------------------------------------------------
# Shared fixtures (module-level, built once)
# ---------------------------------------------------------------------------

one_ary = np.array(
    [
        [0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
        [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0],
    ]
)

cols = [
    "Apple",
    "Corn",
    "Dill",
    "Eggs",
    "Ice cream",
    "Kidney Beans",
    "Milk",
    "Nutmeg",
    "Onion",
    "Unicorn",
    "Yogurt",
]

df = pd.DataFrame(one_ary, columns=cols).astype(bool)
df_freq_items = fpgrowth(df, min_support=0.6)
df_freq_items_with_colnames = fpgrowth(df, min_support=0.6, use_colnames=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default() -> None:
    res_df = association_rules(df_freq_items, len(df))
    assert res_df.shape[0] > 0
    assert "antecedents" in res_df.columns
    assert "consequents" in res_df.columns


def test_datatypes() -> None:
    res_df = association_rules(df_freq_items, len(df))
    for i in res_df["antecedents"]:
        assert isinstance(i, frozenset)
    for i in res_df["consequents"]:
        assert isinstance(i, frozenset)


def test_no_support_col() -> None:
    df_no_support_col = df_freq_items.loc[:, ["itemsets"]]
    numpy_assert_raises(ValueError, association_rules, df_no_support_col, len(df))


def test_no_itemsets_col() -> None:
    df_no_itemsets_col = df_freq_items.loc[:, ["support"]]
    numpy_assert_raises(ValueError, association_rules, df_no_itemsets_col, len(df))


def test_wrong_metric() -> None:
    numpy_assert_raises(
        ValueError, association_rules, df_freq_items, len(df), None, False, "unicorn"
    )


def test_empty_result() -> None:
    res_df = association_rules(df_freq_items, len(df), min_threshold=2)
    assert res_df.shape[0] == 0


def test_leverage() -> None:
    res_df = association_rules(
        df_freq_items, len(df), min_threshold=0.1, metric="leverage"
    )
    assert res_df.shape[0] == 6

    res_df = association_rules(
        df_freq_items_with_colnames, len(df), min_threshold=0.1, metric="leverage"
    )
    assert res_df.shape[0] == 6


def test_conviction() -> None:
    res_df = association_rules(
        df_freq_items, len(df), min_threshold=1.5, metric="conviction"
    )
    assert res_df.shape[0] == 11

    res_df = association_rules(
        df_freq_items_with_colnames, len(df), min_threshold=1.5, metric="conviction"
    )
    assert res_df.shape[0] == 11


def test_lift() -> None:
    res_df = association_rules(df_freq_items, len(df), min_threshold=1.1, metric="lift")
    assert res_df.shape[0] == 6

    res_df = association_rules(
        df_freq_items_with_colnames, len(df), min_threshold=1.1, metric="lift"
    )
    assert res_df.shape[0] == 6


def test_confidence() -> None:
    res_df = association_rules(
        df_freq_items, len(df), min_threshold=0.8, metric="confidence"
    )
    assert res_df.shape[0] == 9

    res_df = association_rules(
        df_freq_items_with_colnames, len(df), min_threshold=0.8, metric="confidence"
    )
    assert res_df.shape[0] == 9


def test_representativity() -> None:
    res_df = association_rules(
        df_freq_items, len(df), min_threshold=1.0, metric="representativity"
    )
    assert res_df.shape[0] == 16

    res_df = association_rules(
        df_freq_items_with_colnames,
        len(df),
        min_threshold=1.0,
        metric="representativity",
    )
    assert res_df.shape[0] == 16


def test_jaccard() -> None:
    res_df = association_rules(
        df_freq_items, len(df), min_threshold=0.7, metric="jaccard"
    )
    assert res_df.shape[0] == 8

    res_df = association_rules(
        df_freq_items_with_colnames, len(df), min_threshold=0.7, metric="jaccard"
    )
    assert res_df.shape[0] == 8


def test_certainty() -> None:
    res_df = association_rules(
        df_freq_items, len(df), metric="certainty", min_threshold=0.6
    )
    assert res_df.shape[0] == 3

    res_df = association_rules(
        df_freq_items_with_colnames, len(df), metric="certainty", min_threshold=0.6
    )
    assert res_df.shape[0] == 3


def test_kulczynski() -> None:
    res_df = association_rules(
        df_freq_items, len(df), metric="kulczynski", min_threshold=0.9
    )
    assert res_df.shape[0] == 2

    res_df = association_rules(
        df_freq_items_with_colnames, len(df), metric="kulczynski", min_threshold=0.6
    )
    assert res_df.shape[0] == 16


def test_frozenset_selection() -> None:
    res_df = association_rules(df_freq_items, len(df))
    sel = res_df[res_df["consequents"] == frozenset((3, 5))]
    assert sel.shape[0] == 1
    sel = res_df[res_df["consequents"] == frozenset((5, 3))]
    assert sel.shape[0] == 1
    sel = res_df[res_df["consequents"] == {3, 5}]
    assert sel.shape[0] == 1
    sel = res_df[res_df["antecedents"] == frozenset((8, 3))]
    assert sel.shape[0] == 1


def test_override_metric_with_support() -> None:
    res_df = association_rules(df_freq_items_with_colnames, len(df), min_threshold=0.8)
    assert res_df.shape[0] == 9

    res_df = association_rules(
        df_freq_items_with_colnames, len(df), min_threshold=0.8, metric="support"
    )
    assert res_df.shape[0] == 2

    res_df = association_rules(
        df_freq_items_with_colnames, len(df), min_threshold=0.8, support_only=True
    )
    assert res_df.shape[0] == 2


def test_on_df_with_missing_entries() -> None:
    d = {
        "itemsets": [
            [177, 176],
            [177, 179],
            [176, 178],
            [176, 179],
            [93, 100],
            [177, 178],
            [177, 176, 178],
        ],
        "support": [
            0.253623,
            0.253623,
            0.217391,
            0.217391,
            0.181159,
            0.108696,
            0.108696,
        ],
    }
    df_missing = pd.DataFrame(d)
    numpy_assert_raises(KeyError, association_rules, df_missing, len(df))


def test_on_df_with_missing_entries_support_only() -> None:
    d = {
        "itemsets": [
            [177, 176],
            [177, 179],
            [176, 178],
            [176, 179],
            [93, 100],
            [177, 178],
            [177, 176, 178],
        ],
        "support": [
            0.253623,
            0.253623,
            0.217391,
            0.217391,
            0.181159,
            0.108696,
            0.108696,
        ],
    }
    df_missing = pd.DataFrame(d)
    df_result = association_rules(
        df_missing, len(df), support_only=True, min_threshold=0.1
    )
    assert df_result["support"].shape == (18,)
    assert int(np.isnan(df_result["support"].values).any()) != 1


def test_with_empty_dataframe() -> None:
    df_freq = df_freq_items_with_colnames.iloc[:0]
    with pytest.raises(ValueError):
        association_rules(df_freq, len(df))


# ---------------------------------------------------------------------------
# Apache Spark MLlib Ported Tests
# ---------------------------------------------------------------------------


def _to_dataframe(transactions: list[list[str]]) -> pd.DataFrame:
    items = sorted(set(item for t in transactions for item in t))
    data = [{item: (item in t) for item in items} for t in transactions]
    return pd.DataFrame(data)


def test_spark_mllib_association_rules() -> None:
    # Ported from Spark MLlib: association rules using String type
    transactions = [
        "r z h k p".split(" "),
        "z y x w v u t s".split(" "),
        "s x o n r".split(" "),
        "x z y m t s q e".split(" "),
        ["z"],
        "x z y r q t p".split(" "),
    ]
    df = _to_dataframe(transactions)

    # Fit FP-Growth first with min_support=0.5
    freq_items = fpgrowth(df, min_support=0.5, use_colnames=True)

    # In Spark MLlib Association Rules test with minConfidence=0.9
    # generates 23 rules, 23 with confidence == 1.0 (with absolute tolerance 1e-6)
    ar = association_rules(freq_items, len(df), metric="confidence", min_threshold=0.9)
    # rusket generates all possible combinations (like mlxtend), Spark only single consequents
    assert len(ar) == 37
    assert (ar["confidence"] >= 0.999999).sum() == 37
    # Verify we matched Spark's 23 exactly
    ar_single_cons = ar[ar["consequents"].apply(len) == 1]
    assert len(ar_single_cons) == 23
    assert (ar_single_cons["confidence"] >= 0.999999).sum() == 23

    # In Spark MLlib Association Rules test with minConfidence=0.0
    ar_all = association_rules(
        freq_items, len(df), metric="confidence", min_threshold=0.0
    )
    assert (
        len(ar_all) == 52
    )  # rusket generates all multi-consequent combos (Spark gives 50 single-only)
    ar_all_single_cons = ar_all[ar_all["consequents"].apply(len) == 1]
    assert len(ar_all_single_cons) == 30
    assert (ar_all_single_cons["confidence"] >= 0.999999).sum() == 23
