"""Kosarak dataset regression test for association_rules.

Ported from cpearce/arm-rs (Apache 2.0 license).
Tests that association_rules produces correct confidence/lift/support values
on a well-known dataset (Kosarak, 990,002 transactions, minsup=0.05).
"""

from __future__ import annotations

import pandas as pd
import pytest

from rusket import association_rules

# ---------------------------------------------------------------------------
# Kosarak itemsets mined with min_support = 0.05 (count / 990002)
# (items, count)
# ---------------------------------------------------------------------------
DATASET_SIZE = 990002

KOSARAK_ITEMSETS: list[tuple[list[int], int]] = [
    ([1, 6, 11], 86092),
    ([1, 11], 91882),
    ([1, 3, 6], 57802),
    ([1, 3], 84660),
    ([1, 6], 132113),
    ([1], 197522),
    ([55], 65412),
    ([4], 78097),
    ([6], 601374),
    ([3, 6, 11], 143682),
    ([3, 11], 161286),
    ([6, 11], 324013),
    ([11], 364065),
    ([6, 148, 218], 56838),
    ([6, 11, 148, 218], 49866),
    ([11, 148, 218], 50098),
    ([148, 218], 58823),
    ([6, 11, 148], 55230),
    ([11, 148], 55759),
    ([6, 148], 64750),
    ([148], 69922),
    ([6, 11, 218], 60630),
    ([11, 218], 61656),
    ([6, 218], 77675),
    ([218], 88598),
    ([6, 7, 11], 55835),
    ([7, 11], 57074),
    ([6, 7], 73610),
    ([7], 86898),
    ([3, 6], 265180),
    ([3], 450031),
    ([6, 27], 59418),
    ([27], 72134),
]

# (antecedent, consequent) -> (confidence, lift, support)
EXPECTED_RULES: dict[tuple[frozenset[int], frozenset[int]], tuple[float, float, float]] = {
    (frozenset([6]), frozenset([1, 11])): (0.143, 1.542, 0.0870),
    (frozenset([11]), frozenset([1, 6])): (0.236, 1.772, 0.0870),
    (frozenset([218]), frozenset([148])): (0.664, 9.400, 0.059),
    (frozenset([148, 218]), frozenset([6])): (0.966, 1.591, 0.057),
    (frozenset([1, 6]), frozenset([11])): (0.652, 1.772, 0.087),
    (frozenset([11, 218]), frozenset([6, 148])): (0.809, 12.366, 0.050),
    (frozenset([11]), frozenset([7])): (0.157, 1.786, 0.058),
    (frozenset([11]), frozenset([6, 148, 218])): (0.137, 2.386, 0.050),
    (frozenset([11]), frozenset([148, 218])): (0.138, 2.316, 0.051),
    (frozenset([11, 218]), frozenset([6])): (0.983, 1.619, 0.061),
    (frozenset([7, 11]), frozenset([6])): (0.978, 1.610, 0.056),
    (frozenset([148]), frozenset([11])): (0.797, 2.168, 0.056),
    (frozenset([11]), frozenset([6, 148])): (0.152, 2.319, 0.056),
    (frozenset([218]), frozenset([11])): (0.696, 1.892, 0.062),
    (frozenset([218]), frozenset([11, 148])): (0.565, 10.040, 0.051),
    (frozenset([148]), frozenset([6])): (0.926, 1.524, 0.065),
    (frozenset([6, 11]), frozenset([148])): (0.170, 2.413, 0.056),
    (frozenset([11]), frozenset([6, 7])): (0.153, 2.063, 0.056),
    (frozenset([11, 148]), frozenset([218])): (0.898, 10.040, 0.051),
    (frozenset([148]), frozenset([6, 11, 218])): (0.713, 11.645, 0.050),
    (frozenset([6]), frozenset([11, 148, 218])): (0.083, 1.639, 0.050),
    (frozenset([7]), frozenset([6, 11])): (0.643, 1.963, 0.056),
    (frozenset([6, 11, 148]), frozenset([218])): (0.903, 10.089, 0.050),
    (frozenset([148]), frozenset([6, 218])): (0.813, 10.360, 0.057),
    (frozenset([148]), frozenset([6, 11])): (0.790, 2.413, 0.056),
    (frozenset([6, 148]), frozenset([218])): (0.878, 9.809, 0.057),
    (frozenset([11]), frozenset([148])): (0.153, 2.168, 0.056),
    (frozenset([11, 148]), frozenset([6])): (0.991, 1.631, 0.056),
    (frozenset([6, 148, 218]), frozenset([11])): (0.877, 2.386, 0.050),
    (frozenset([6]), frozenset([148, 218])): (0.095, 1.591, 0.057),
    (frozenset([11]), frozenset([6, 218])): (0.167, 2.123, 0.061),
    (frozenset([218]), frozenset([6, 148])): (0.642, 9.809, 0.057),
    (frozenset([6, 148]), frozenset([11])): (0.853, 2.319, 0.056),
    (frozenset([6, 11]), frozenset([7])): (0.172, 1.963, 0.056),
    (frozenset([218]), frozenset([6, 11, 148])): (0.563, 10.089, 0.050),
    (frozenset([148, 218]), frozenset([11])): (0.852, 2.316, 0.051),
    (frozenset([6, 148]), frozenset([11, 218])): (0.770, 12.366, 0.050),
    (frozenset([148]), frozenset([11, 218])): (0.716, 11.504, 0.051),
    (frozenset([218]), frozenset([6, 11])): (0.684, 2.091, 0.061),
    (frozenset([11, 148, 218]), frozenset([6])): (0.995, 1.639, 0.050),
    (frozenset([11]), frozenset([218])): (0.169, 1.892, 0.062),
    (frozenset([1, 11]), frozenset([6])): (0.937, 1.542, 0.087),
    (frozenset([6, 11]), frozenset([218])): (0.187, 2.091, 0.061),
    (frozenset([6]), frozenset([148])): (0.108, 1.524, 0.065),
    (frozenset([6]), frozenset([11, 148])): (0.092, 1.631, 0.056),
    (frozenset([148, 218]), frozenset([6, 11])): (0.848, 2.590, 0.050),
    (frozenset([6, 218]), frozenset([11])): (0.781, 2.123, 0.061),
    (frozenset([6, 7]), frozenset([11])): (0.759, 2.063, 0.056),
    (frozenset([6]), frozenset([11, 218])): (0.101, 1.619, 0.061),
    (frozenset([11, 218]), frozenset([148])): (0.813, 11.504, 0.051),
    (frozenset([6, 11]), frozenset([148, 218])): (0.154, 2.590, 0.050),
    (frozenset([148]), frozenset([218])): (0.841, 9.400, 0.059),
    (frozenset([7]), frozenset([11])): (0.657, 1.786, 0.058),
    (frozenset([6, 218]), frozenset([11, 148])): (0.642, 11.398, 0.050),
    (frozenset([6, 11, 218]), frozenset([148])): (0.822, 11.645, 0.050),
    (frozenset([6, 218]), frozenset([148])): (0.732, 10.360, 0.057),
    (frozenset([6]), frozenset([7, 11])): (0.093, 1.610, 0.056),
    (frozenset([11, 148]), frozenset([6, 218])): (0.894, 11.398, 0.050),
}


def _build_itemset_dataframe() -> pd.DataFrame:
    """Build a DataFrame matching the rusket association_rules input format."""
    rows = []
    for items, count in KOSARAK_ITEMSETS:
        rows.append(
            {
                "itemsets": frozenset(items),
                "support": count / DATASET_SIZE,
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["num_itemsets"] = DATASET_SIZE
    return df


def _fuzzy_eq(a: float, b: float, tol: float = 0.002) -> bool:
    return abs(a - b) < tol


class TestKosarakRules:
    """Kosarak regression — 58 rules expected at confidence≥0.05, lift≥1.5."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        itemset_df = _build_itemset_dataframe()
        self.rules = association_rules(
            itemset_df,
            metric="confidence",
            min_threshold=0.05,
            return_metrics=["confidence", "lift", "support"],
        )

    def test_rule_count(self) -> None:
        """The lift filter is applied post-hoc, count all rules with lift≥1.5."""
        filtered = self.rules[self.rules["lift"] >= 1.5]
        assert len(filtered) == len(EXPECTED_RULES), (
            f"Expected {len(EXPECTED_RULES)} rules with lift>=1.5, got {len(filtered)}"
        )

    def test_all_expected_rules_present(self) -> None:
        """Every expected (antecedent, consequent) pair must be present."""
        generated_keys = {
            (frozenset(row["antecedents"]), frozenset(row["consequents"])) for _, row in self.rules.iterrows()
        }
        for key in EXPECTED_RULES:
            ant, con = key
            # The expected rules with lift >= 1.5 should all be present
            assert key in generated_keys, f"Missing rule: {ant} -> {con}"

    def test_confidence_values(self) -> None:
        """Confidence values must match within tolerance."""
        for _, row in self.rules.iterrows():
            key = (frozenset(row["antecedents"]), frozenset(row["consequents"]))
            if key in EXPECTED_RULES:
                expected_conf, _, _ = EXPECTED_RULES[key]
                assert _fuzzy_eq(row["confidence"], expected_conf), (
                    f"Confidence mismatch for {key}: expected {expected_conf}, got {row['confidence']:.3f}"
                )

    def test_lift_values(self) -> None:
        """Lift values must match within tolerance."""
        for _, row in self.rules.iterrows():
            key = (frozenset(row["antecedents"]), frozenset(row["consequents"]))
            if key in EXPECTED_RULES:
                _, expected_lift, _ = EXPECTED_RULES[key]
                assert _fuzzy_eq(row["lift"], expected_lift), (
                    f"Lift mismatch for {key}: expected {expected_lift}, got {row['lift']:.3f}"
                )

    def test_support_values(self) -> None:
        """Support values must match within tolerance."""
        for _, row in self.rules.iterrows():
            key = (frozenset(row["antecedents"]), frozenset(row["consequents"]))
            if key in EXPECTED_RULES:
                _, _, expected_sup = EXPECTED_RULES[key]
                assert _fuzzy_eq(row["support"], expected_sup), (
                    f"Support mismatch for {key}: expected {expected_sup}, got {row['support']:.3f}"
                )
