"""Tests using Faker-generated e-commerce data through the full mining pipeline.

Validates from_transactions → fpgrowth/eclat → association_rules with realistic
product catalogues, diverse transaction sizes, and mixed-type identifiers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from faker import Faker

import rusket
from rusket import association_rules, fpgrowth, from_transactions

# ---------------------------------------------------------------------------
# Helpers — Faker-powered dataset generators
# ---------------------------------------------------------------------------

SEED = 42


def _make_product_catalogue(fake: Faker, n: int = 80) -> list[str]:
    """Generate *n* unique Faker product names."""
    seen: set[str] = set()
    products: list[str] = []
    while len(products) < n:
        name = f"{fake.word().capitalize()} {fake.color_name()}"
        if name not in seen:
            seen.add(name)
            products.append(name)
    return products


def _make_basket_df(
    n_transactions: int = 500,
    n_products: int = 60,
    seed: int = SEED,
) -> pd.DataFrame:
    """Synthetic market-basket with Zipf-like product popularity."""
    fake = Faker()
    Faker.seed(seed)
    rng = np.random.default_rng(seed)

    products = _make_product_catalogue(fake, n_products)
    rank = np.arange(1, n_products + 1, dtype=float)
    support = np.clip(0.55 / rank**0.55, 0.005, 0.55)
    matrix = rng.random((n_transactions, n_products)) < support
    return pd.DataFrame(matrix.astype(bool), columns=products)


def _make_long_transactions(
    n_orders: int = 300,
    n_products: int = 40,
    basket_size_range: tuple[int, int] = (1, 8),
    seed: int = SEED,
) -> pd.DataFrame:
    """Long-format transaction DataFrame with Faker-generated names."""
    fake = Faker()
    Faker.seed(seed)
    rng = np.random.default_rng(seed)

    products = _make_product_catalogue(fake, n_products)
    rows: list[dict[str, str | int]] = []
    for order_id in range(1, n_orders + 1):
        k = rng.integers(*basket_size_range)
        items = rng.choice(products, size=k, replace=False)
        for item in items:
            rows.append({"order_id": order_id, "product": item})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# from_transactions — realistic data
# ---------------------------------------------------------------------------


class TestFakerFromTransactions:
    """from_transactions with Faker-generated product names."""

    def test_long_format_pandas(self) -> None:
        df = _make_long_transactions()
        ohe = from_transactions(df, transaction_col="order_id", item_col="product")
        assert isinstance(ohe, pd.DataFrame)
        assert ohe.dtypes.apply(pd.api.types.is_bool_dtype).all()
        # Each column is a product name string
        assert all(isinstance(c, str) and len(c) > 0 for c in ohe.columns)

    def test_long_format_polars(self) -> None:
        pl = pytest.importorskip("polars")
        df_pd = _make_long_transactions()
        df_pl = pl.from_pandas(df_pd)
        ohe = from_transactions(df_pl, transaction_col="order_id", item_col="product")
        assert isinstance(ohe, pl.DataFrame)
        assert ohe.shape[0] > 0

    def test_pandas_polars_equivalence(self) -> None:
        pl = pytest.importorskip("polars")
        df_pd = _make_long_transactions(n_orders=100, seed=99)
        df_pl = pl.from_pandas(df_pd)

        ohe_pd = from_transactions(df_pd, transaction_col="order_id", item_col="product")
        ohe_pl = from_transactions(df_pl, transaction_col="order_id", item_col="product").to_pandas()

        common: list[str] = sorted(set(ohe_pd.columns) & set(ohe_pl.columns))
        ohe_pd_sorted = ohe_pd.reindex(columns=common).sort_values(by=common, ignore_index=True).astype(bool)
        ohe_pl_sorted = ohe_pl.reindex(columns=common).sort_values(by=common, ignore_index=True).astype(bool)
        pd.testing.assert_frame_equal(
            ohe_pd_sorted,
            ohe_pl_sorted,
            check_dtype=False,
        )

    def test_list_of_lists(self) -> None:
        fake = Faker()
        Faker.seed(SEED)
        products = _make_product_catalogue(fake, 30)
        rng = np.random.default_rng(SEED)
        baskets = [rng.choice(products, size=rng.integers(2, 6), replace=False).tolist() for _ in range(200)]
        ohe = from_transactions(baskets)
        assert isinstance(ohe, pd.DataFrame)
        assert ohe.shape[0] == 200
        assert ohe.dtypes.apply(pd.api.types.is_bool_dtype).all()

    def test_min_item_count_with_faker(self) -> None:
        df = _make_long_transactions(n_orders=200, n_products=50, seed=77)
        ohe_full = from_transactions(df, transaction_col="order_id", item_col="product")
        ohe_filtered = from_transactions(df, transaction_col="order_id", item_col="product", min_item_count=10)
        # Filtering should remove infrequent items
        assert ohe_filtered.shape[1] <= ohe_full.shape[1]
        assert ohe_filtered.shape[0] == ohe_full.shape[0]


# ---------------------------------------------------------------------------
# fpgrowth / eclat with Faker data
# ---------------------------------------------------------------------------


class TestFakerMining:
    """Frequent-pattern mining on Faker-generated baskets."""

    @pytest.fixture(scope="class")
    def basket_df(self) -> pd.DataFrame:
        return _make_basket_df(n_transactions=500, n_products=50)

    def test_fpgrowth_produces_results(self, basket_df: pd.DataFrame) -> None:
        freq = fpgrowth(basket_df, min_support=0.05, use_colnames=True)
        assert len(freq) > 0
        assert "support" in freq.columns
        assert "itemsets" in freq.columns
        # All itemsets should contain real product names
        for itemset in freq["itemsets"]:
            for item in itemset:
                assert isinstance(item, str)

    def test_eclat_produces_results(self, basket_df: pd.DataFrame) -> None:
        freq = rusket.eclat(basket_df, min_support=0.05, use_colnames=True)
        assert len(freq) > 0

    def test_fpgrowth_eclat_agreement(self, basket_df: pd.DataFrame) -> None:
        """FPGrowth and Eclat must produce identical itemsets on the same data."""
        fp = fpgrowth(basket_df, min_support=0.08, use_colnames=True)
        ec = rusket.eclat(basket_df, min_support=0.08, use_colnames=True)

        fp_sets = {tuple(sorted(row)) for row in fp["itemsets"]}
        ec_sets = {tuple(sorted(row)) for row in ec["itemsets"]}
        assert fp_sets == ec_sets

    def test_all_methods_agree(self, basket_df: pd.DataFrame) -> None:
        """All mining methods agree on Faker-generated data."""
        methods = ["fpgrowth", "eclat", "fin", "lcm"]
        results: dict[str, set[tuple[str, ...]]] = {}
        for method in methods:
            res = fpgrowth(basket_df, min_support=0.1, use_colnames=True, method=method)
            results[method] = {tuple(sorted(row)) for row in res["itemsets"]}

        ref = results["fpgrowth"]
        for m in methods[1:]:
            assert results[m] == ref, f"{m} disagrees with fpgrowth"

    def test_high_support_filters_correctly(self, basket_df: pd.DataFrame) -> None:
        freq_low = fpgrowth(basket_df, min_support=0.05, use_colnames=True)
        freq_high = fpgrowth(basket_df, min_support=0.3, use_colnames=True)
        assert len(freq_high) < len(freq_low)
        # Every high-support itemset must also appear in low-support results
        high_sets = {tuple(sorted(r)) for r in freq_high["itemsets"]}
        low_sets = {tuple(sorted(r)) for r in freq_low["itemsets"]}
        assert high_sets.issubset(low_sets)




# ---------------------------------------------------------------------------
# Association rules with Faker data
# ---------------------------------------------------------------------------


class TestFakerAssociationRules:
    """Association rules on Faker-generated baskets."""

    @pytest.fixture(scope="class")
    def rules_df(self) -> pd.DataFrame:
        basket = _make_basket_df(n_transactions=500, n_products=40, seed=7)
        freq = fpgrowth(basket, min_support=0.08, use_colnames=True)
        return association_rules(freq, num_itemsets=len(basket), metric="lift", min_threshold=1.0)

    def test_rules_non_empty(self, rules_df: pd.DataFrame) -> None:
        assert len(rules_df) > 0

    def test_rules_columns(self, rules_df: pd.DataFrame) -> None:
        expected = {"antecedents", "consequents", "support", "confidence", "lift"}
        assert expected.issubset(set(rules_df.columns))

    def test_rules_lift_above_threshold(self, rules_df: pd.DataFrame) -> None:
        assert (rules_df["lift"] >= 1.0).all()

    def test_antecedent_consequent_disjoint(self, rules_df: pd.DataFrame) -> None:
        for _, row in rules_df.iterrows():
            a = set(row["antecedents"])
            c = set(row["consequents"])
            assert a.isdisjoint(c), f"Overlap: {a & c}"

    def test_multiple_metrics(self) -> None:
        basket = _make_basket_df(n_transactions=300, n_products=30, seed=13)
        freq = fpgrowth(basket, min_support=0.1, use_colnames=True)
        for metric in ["confidence", "lift", "leverage", "conviction", "jaccard"]:
            rules = association_rules(freq, num_itemsets=len(basket), metric=metric, min_threshold=0.01)
            assert isinstance(rules, pd.DataFrame)


# ---------------------------------------------------------------------------
# Edge cases with extreme Faker data
# ---------------------------------------------------------------------------


class TestFakerEdgeCases:
    """Edge-case tests using Faker-generated data."""

    def test_very_sparse_baskets(self) -> None:
        """Only 1-2 items per basket — should produce singletons only at low support."""
        fake = Faker()
        Faker.seed(123)
        products = _make_product_catalogue(fake, 20)
        rng = np.random.default_rng(123)
        baskets = [rng.choice(products, size=1, replace=False).tolist() for _ in range(50)]
        ohe = from_transactions(baskets)
        freq = fpgrowth(ohe, min_support=0.01, use_colnames=True)
        multi = [r for r in freq["itemsets"] if len(r) > 1]
        assert len(multi) == 0

    def test_large_baskets(self) -> None:
        """Baskets of 15-25 items — produces many multi-item itemsets."""
        fake = Faker()
        Faker.seed(456)
        products = _make_product_catalogue(fake, 30)
        rng = np.random.default_rng(456)
        baskets = [rng.choice(products, size=rng.integers(15, 26), replace=False).tolist() for _ in range(100)]
        ohe = from_transactions(baskets)
        freq = fpgrowth(ohe, min_support=0.3, use_colnames=True)
        assert len(freq) > 0
        multi = [r for r in freq["itemsets"] if len(r) >= 2]
        assert len(multi) > 0

    def test_unicode_product_names(self) -> None:
        """Faker can generate names with accents — ensure pipeline handles them."""
        fake = Faker(["de_DE", "fr_FR", "ja_JP"])
        Faker.seed(789)
        products: list[str] = []
        seen: set[str] = set()
        while len(products) < 20:
            name = fake.word()
            if name not in seen:
                seen.add(name)
                products.append(name)

        rng = np.random.default_rng(789)
        baskets = [rng.choice(products, size=rng.integers(2, 6), replace=False).tolist() for _ in range(100)]
        ohe = from_transactions(baskets)
        freq = fpgrowth(ohe, min_support=0.1, use_colnames=True)
        assert len(freq) > 0

    def test_integer_and_string_order_ids(self) -> None:
        """Faker-generated string order IDs should work like integers."""
        fake = Faker()
        Faker.seed(42)
        products = _make_product_catalogue(fake, 15)
        rng = np.random.default_rng(42)
        rows = []
        for _ in range(200):
            order_id = fake.uuid4()[:8]
            items = rng.choice(products, size=rng.integers(2, 5), replace=False)
            for item in items:
                rows.append({"order": order_id, "product": item})
        df = pd.DataFrame(rows)
        ohe = from_transactions(df, transaction_col="order", item_col="product")
        assert ohe.shape[0] > 0
        assert ohe.dtypes.apply(pd.api.types.is_bool_dtype).all()
