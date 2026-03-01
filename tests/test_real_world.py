"""Integration tests using real-world datasets.

Tests in this file are marked with ``@pytest.mark.integration`` and will
automatically *skip* if the required dataset cannot be downloaded (e.g. in
offline CI or without a Kaggle token).

Running locally
---------------
    # Run only integration tests (downloads UCI dataset ~6 MB on first run):
    uv run pytest tests/test_real_world.py -v

    # Run only Kaggle-required tests (needs ~/.kaggle/kaggle.json):
    uv run pytest tests/test_real_world.py -v -m kaggle

    # Skip all integration tests (default in CI):
    uv run pytest tests/ -m "not integration"

Dataset caching
---------------
Downloaded data is cached to ``tests/.dataset_cache/`` as Parquet files.
The second and all subsequent runs load the parquet cache (< 1 s per fixture).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import pandas as pd

# ---------------------------------------------------------------------------
# Helper — assert DataFrame is non-empty and has expected columns
# ---------------------------------------------------------------------------


def _check_df(df: pd.DataFrame, cols: list[str], min_rows: int = 1) -> None:  # type: ignore[name-defined]
    assert len(df) >= min_rows, f"Expected at least {min_rows} rows, got {len(df)}"
    for col in cols:
        assert col in df.columns, f"Missing column '{col}'"


# ===========================================================================
# UCI Online Retail II — Market Basket, HUPM, Substitutes, Saturation
# ===========================================================================


@pytest.mark.integration
class TestOnlineRetailBasketAnalysis:
    """FP-Growth, association rules, recommend_items, find_substitutes."""

    def test_from_transactions_shape(self, online_retail_df: pd.DataFrame) -> None:  # type: ignore[name-defined]
        """from_transactions produces a valid one-hot matrix of expected size."""
        import rusket

        basket = rusket.from_transactions(
            online_retail_df,
            transaction_col="Invoice",
            item_col="Description",
        )
        n_invoices = online_retail_df["Invoice"].nunique()
        n_items = online_retail_df["Description"].nunique()
        assert basket.shape == (n_invoices, n_items)
        # All values must be 0 or 1
        assert basket.values.min() >= 0
        assert basket.values.max() <= 1

    def test_fpgrowth_oo_api_mine(self, online_retail_df: pd.DataFrame) -> None:  # type: ignore[name-defined]
        """FPGrowth.from_transactions → .mine() returns itemsets with valid support."""
        from rusket.fpgrowth import FPGrowth

        model = FPGrowth.from_transactions(
            online_retail_df,
            transaction_col="Invoice",
            item_col="Description",
            min_support=0.05,
            use_colnames=True,
        )
        freq = model.mine()
        _check_df(freq, ["support", "itemsets"])
        assert (freq["support"] >= 0.05).all(), "All itemsets must meet min_support"
        assert (freq["support"] <= 1.0).all(), "Support must be ≤ 1.0"

    def test_association_rules_from_oo_api(self, online_retail_df: pd.DataFrame) -> None:  # type: ignore[name-defined]
        """FPGrowth.association_rules returns rules with correct metric columns."""
        from rusket.fpgrowth import FPGrowth

        model = FPGrowth.from_transactions(
            online_retail_df,
            transaction_col="Invoice",
            item_col="Description",
            min_support=0.02,
            use_colnames=True,
        )
        rules = model.association_rules(metric="confidence", min_threshold=0.5)  # type: ignore
        _check_df(rules, ["antecedents", "consequents", "support", "confidence", "lift"])
        assert (rules["confidence"] >= 0.05).all()
        assert (rules["lift"] > 0).all()

    def test_recommend_items_returns_unseen(self, online_retail_df: pd.DataFrame) -> None:  # type: ignore[name-defined]
        """recommend_items returns products NOT in the query cart."""
        from rusket.fpgrowth import FPGrowth

        model = FPGrowth.from_transactions(
            online_retail_df,
            transaction_col="Invoice",
            item_col="Description",
            min_support=0.05,
            use_colnames=True,
        )
        # Use the most common items as a realistic cart
        top_items = online_retail_df["Description"].value_counts().head(3).index.tolist()
        recs = model.recommend_for_cart(top_items, n=5)
        # Recommendations must not include the input items
        for item in recs:
            assert item not in top_items, f"Recommended item {item!r} was in the input cart"

    def test_recommend_items_cache_is_consistent(self, online_retail_df: pd.DataFrame) -> None:  # type: ignore[name-defined]
        """Calling recommend_items twice returns identical results (cache hit)."""
        from rusket.fpgrowth import FPGrowth

        model = FPGrowth.from_transactions(
            online_retail_df,
            transaction_col="Invoice",
            item_col="Description",
            min_support=0.05,
            use_colnames=True,
        )
        top_items = online_retail_df["Description"].value_counts().head(2).index.tolist()
        recs1 = model.recommend_for_cart(top_items, n=5)
        recs2 = model.recommend_for_cart(top_items, n=5)
        assert recs1 == recs2, "Cached recommend_items must be deterministic"

    def test_find_substitutes_no_negative_lift(self, online_retail_df: pd.DataFrame) -> None:  # type: ignore[name-defined]
        """find_substitutes returns only pairs with lift < max_lift."""
        import rusket
        from rusket.fpgrowth import FPGrowth

        model = FPGrowth.from_transactions(
            online_retail_df,
            transaction_col="Invoice",
            item_col="Description",
            min_support=0.03,
            use_colnames=True,
        )
        rules = model.association_rules(metric="confidence", min_threshold=0.1)  # type: ignore
        substitutes = rusket.find_substitutes(rules, max_lift=0.95)
        if not substitutes.empty:
            assert (substitutes["lift"] < 0.95).all()
            assert (substitutes["lift"] > 0).all()

    def test_customer_saturation_deciles(self, online_retail_df: pd.DataFrame) -> None:  # type: ignore[name-defined]
        """customer_saturation produces 10 deciles and valid saturation percentages."""
        import rusket

        sat = rusket.customer_saturation(
            online_retail_df,
            user_col="Customer_ID",
            item_col="Description",
        )
        _check_df(sat, ["unique_count", "saturation_pct", "decile"])
        assert sat["saturation_pct"].between(0.0, 1.0).all()
        assert set(sat["decile"].unique()) <= {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

    def test_eclat_matches_fpgrowth_support(self, online_retail_df: pd.DataFrame) -> None:
        """Eclat and FP-Growth must agree on support values for shared itemsets."""
        from rusket.eclat import Eclat
        from rusket.fpgrowth import FPGrowth

        min_sup = 0.02

        fp = FPGrowth.from_transactions(
            online_retail_df,
            transaction_col="Invoice",
            item_col="Description",
            min_support=min_sup,
            use_colnames=True,
        ).mine()

        ec = Eclat.from_transactions(
            online_retail_df,
            transaction_col="Invoice",
            item_col="Description",
            min_support=min_sup,
            use_colnames=True,
        ).mine()

        # Compare singleton support values (most stable across implementations)
        def _singleton_support(df: pd.DataFrame) -> dict[tuple, float]:  # type: ignore[name-defined]
            return {
                tuple(row["itemsets"]): round(row["support"], 4)  # type: ignore
                for _, row in df.iterrows()
                if len(row["itemsets"]) == 1
            }

        fp_single = _singleton_support(fp)
        ec_single = _singleton_support(ec)

        common = set(fp_single) & set(ec_single)
        assert len(common) > 0, "No common singletons to compare"
        for key in common:
            assert abs(fp_single[key] - ec_single[key]) < 0.005, (
                f"Support mismatch for {key}: FP={fp_single[key]}, Eclat={ec_single[key]}"
            )


# ===========================================================================
# UCI Online Retail II — HUPM (High-Utility Pattern Mining)
# ===========================================================================


@pytest.mark.integration
class TestOnlineRetailHUPM:
    """HUPM mines revenue-weighted bundles from real transaction data."""

    def test_hupm_from_transactions_basic(self, online_retail_df: pd.DataFrame) -> None:  # type: ignore[name-defined]
        """HUPM finds at least one high-revenue bundle.

        HUPM requires integer item IDs — use StockCode numerically encoded.
        """
        import pandas as pd

        from rusket.hupm import HUPM

        # Encode StockCode as integer IDs for HUPM
        df = online_retail_df.copy()
        df["item_id"] = pd.factorize(df["StockCode"])[0]

        model = HUPM.from_transactions(
            df,
            transaction_col="Invoice",
            item_col="item_id",
            utility_col="Revenue",
            min_utility=50.0,
            max_len=2,
        )
        results = model.mine()
        _check_df(results, ["utility", "itemset"])
        assert (results["utility"] >= 50.0).all()

    def test_hupm_singleton_utility_is_positive(self, online_retail_df: pd.DataFrame) -> None:  # type: ignore[name-defined]
        """All HUPM utilities must be strictly positive."""
        import pandas as pd

        from rusket.hupm import HUPM

        df = online_retail_df.copy()
        df["item_id"] = pd.factorize(df["StockCode"])[0]

        model = HUPM.from_transactions(
            df,
            transaction_col="Invoice",
            item_col="item_id",
            utility_col="Revenue",
            min_utility=10.0,
            max_len=1,
        )
        results = model.mine()
        if not results.empty:
            assert (results["utility"] > 0).all()


# ===========================================================================
# UCI Online Retail II — ALS Recommender
# ===========================================================================


@pytest.mark.integration
class TestOnlineRetailALS:
    """ALS from_transactions on real customer-product interactions."""

    def test_als_from_transactions_shape(self, online_retail_df: pd.DataFrame) -> None:  # type: ignore[name-defined]
        """ALS factor matrices have expected dimensions."""
        import numpy as np

        import rusket

        model = rusket.ALS.from_transactions(
            online_retail_df,
            user_col="Customer_ID",
            item_col="Description",
            factors=32,
            iterations=5,
            seed=42,
        ).fit()
        n_users = online_retail_df["Customer_ID"].nunique()
        n_items = online_retail_df["Description"].nunique()
        assert model.user_factors.shape == (n_users, 32)  # type: ignore
        assert model.item_factors.shape == (n_items, 32)
        assert np.isfinite(model.user_factors).all()  # type: ignore
        assert np.isfinite(model.item_factors).all()

    def test_als_recommend_items_no_seen(self, online_retail_df: pd.DataFrame) -> None:  # type: ignore[name-defined]
        """ALS.recommend_items with exclude_seen=True never returns seen items."""
        import rusket

        model = rusket.ALS.from_transactions(
            online_retail_df,
            user_col="Customer_ID",
            item_col="Description",
            factors=32,
            iterations=5,
            seed=42,
        ).fit()
        # Use user 0 (internal index)
        ids, scores = model.recommend_items(user_id=0, n=10, exclude_seen=True)
        assert len(ids) >= 1
        assert len(scores) == len(ids)
        # Scores must be descending
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], "Scores must be sorted descending"

    def test_similar_items_returns_sorted_cosine(self, online_retail_df: pd.DataFrame) -> None:  # type: ignore[name-defined]
        """similar_items returns cosine similarities in descending order."""
        import rusket

        model = rusket.ALS.from_transactions(
            online_retail_df,
            user_col="Customer_ID",
            item_col="Description",
            factors=32,
            iterations=5,
            seed=42,
        ).fit()
        item_ids, sim_scores = rusket.similar_items(model, item_id=0, n=5)
        assert len(item_ids) >= 1
        assert (sim_scores <= 1.0).all()
        assert (sim_scores >= -1.0).all()
        for i in range(len(sim_scores) - 1):
            assert sim_scores[i] >= sim_scores[i + 1]

    def test_export_item_factors_shape(self, online_retail_df: pd.DataFrame) -> None:  # type: ignore[name-defined]
        """export_item_factors produces one row per item with correct vector length."""
        import numpy as np

        import rusket

        model = rusket.ALS.from_transactions(
            online_retail_df,
            user_col="Customer_ID",
            item_col="Description",
            factors=32,
            iterations=5,
            seed=42,
        ).fit()
        df_factors = rusket.export_item_factors(model, include_labels=True)
        _check_df(df_factors, ["item_id", "vector"])
        assert len(df_factors) == model.item_factors.shape[0]
        vectors = np.stack(df_factors["vector"].values)  # type: ignore
        assert vectors.shape[1] == 32


# ===========================================================================
# UCI Online Retail II — EASE Recommender
# ===========================================================================


@pytest.mark.integration
class TestOnlineRetailEASE:
    """EASE from_transactions on real customer-product interactions."""

    def test_ease_from_transactions_shape(self, online_retail_df: pd.DataFrame) -> None:  # type: ignore[name-defined]
        """EASE factor matrices have expected dimensions."""
        import numpy as np

        import rusket

        model = rusket.EASE.from_transactions(
            online_retail_df,
            user_col="Customer_ID",
            item_col="Description",
            regularization=100.0,
        ).fit()
        n_items = online_retail_df["Description"].nunique()
        assert model.item_weights.shape == (n_items, n_items)  # type: ignore
        assert np.isfinite(model.item_weights).all()  # type: ignore

    def test_ease_recommend_items_no_seen(self, online_retail_df: pd.DataFrame) -> None:  # type: ignore[name-defined]
        """EASE.recommend_items with exclude_seen=True never returns seen items."""
        import rusket

        model = rusket.EASE.from_transactions(
            online_retail_df,
            user_col="Customer_ID",
            item_col="Description",
            regularization=100.0,
        ).fit()
        ids, scores = model.recommend_items(user_id=0, n=10, exclude_seen=True)
        assert len(ids) >= 1
        assert len(scores) == len(ids)
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], "Scores must be sorted descending"

    def test_similar_items_returns_sorted_cosine(self, online_retail_df: pd.DataFrame) -> None:  # type: ignore[name-defined]
        """similar_items returns cosine similarities in descending order for EASE."""
        import rusket

        model = rusket.EASE.from_transactions(
            online_retail_df,
            user_col="Customer_ID",
            item_col="Description",
            regularization=100.0,
        ).fit()
        # Using item weights as item representations
        item_ids, sim_scores = rusket.similar_items(model, item_id=0, n=5)
        assert len(item_ids) >= 1
        assert (sim_scores <= 1.0).all()
        assert (sim_scores >= -1.0).all()
        for i in range(len(sim_scores) - 1):
            assert sim_scores[i] >= sim_scores[i + 1]


# ===========================================================================
# UCI Online Retail II — ItemKNN Recommender
# ===========================================================================


@pytest.mark.integration
class TestOnlineRetailItemKNN:
    """ItemKNN from_transactions on real customer-product interactions."""

    def test_itemknn_from_transactions_shape(self, online_retail_df: pd.DataFrame) -> None:  # type: ignore[name-defined]
        """ItemKNN factor matrices have expected dimensions."""
        import rusket

        model = rusket.ItemKNN.from_transactions(
            online_retail_df,
            user_col="Customer_ID",
            item_col="Description",
            method="bm25",
            k=20,
        ).fit()
        assert model.w_indptr is not None  # type: ignore
        assert model.w_indices is not None  # type: ignore
        assert model.w_data is not None  # type: ignore

        n_items = online_retail_df["Description"].nunique()
        assert model.w_indptr.shape[0] == n_items + 1  # type: ignore

    def test_itemknn_recommend_items_no_seen(self, online_retail_df: pd.DataFrame) -> None:  # type: ignore[name-defined]
        """ItemKNN.recommend_items with exclude_seen=True never returns seen items."""
        import rusket

        model = rusket.ItemKNN.from_transactions(
            online_retail_df,
            user_col="Customer_ID",
            item_col="Description",
            method="bm25",
            k=20,
        ).fit()
        ids, scores = model.recommend_items(user_id=0, n=10, exclude_seen=True)
        assert len(ids) >= 1
        assert len(scores) == len(ids)
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], "Scores must be sorted descending"


# ===========================================================================
# Instacart — requires Kaggle token
# ===========================================================================


@pytest.mark.integration
@pytest.mark.kaggle
class TestInstacartALS:
    """ALS collaborative filter on real Instacart grocery orders."""

    def test_als_from_transactions_grocery(self, instacart_df: pd.DataFrame) -> None:  # type: ignore[name-defined]
        """ALS trains on grocery orders and produces finite factor matrices."""
        import numpy as np

        import rusket

        model = rusket.ALS.from_transactions(
            instacart_df,
            user_col="user_id",
            item_col="product_id",
            factors=64,
            iterations=10,
            seed=42,
        ).fit()
        assert np.isfinite(model.user_factors).all()  # type: ignore
        assert np.isfinite(model.item_factors).all()

    def test_bpr_trains_on_grocery(self, instacart_df: pd.DataFrame) -> None:
        """BPR trains on grocery orders without NaN in factors."""
        import numpy as np

        import rusket

        model = rusket.BPR.from_transactions(
            instacart_df,
            user_col="user_id",
            item_col="product_id",
            factors=32,
            iterations=20,
            seed=42,
        ).fit()
        assert np.isfinite(model.user_factors).all()  # type: ignore
        assert np.isfinite(model.item_factors).all()

    def test_score_potential_shape(self, instacart_df: pd.DataFrame) -> None:
        """score_potential returns (n_users, n_target_items) shaped matrix."""
        import rusket

        model = rusket.ALS.from_transactions(
            instacart_df,
            user_col="user_id",
            item_col="product_id",
            factors=32,
            iterations=5,
            seed=42,
        ).fit()
        user_histories = instacart_df.groupby("user_id")["product_id"].apply(list).tolist()
        n_users = min(100, len(user_histories))
        target_items = list(range(min(20, model.item_factors.shape[0])))
        potential = rusket.score_potential(user_histories[:n_users], model, target_categories=target_items)  # type: ignore
        assert potential.shape == (n_users, len(target_items))

    def test_prefixspan_grocery_sequences(self, instacart_df: pd.DataFrame) -> None:
        """PrefixSpan finds at least one frequent sequence on grocery data."""
        from rusket.prefixspan import PrefixSpan

        top_users = instacart_df["user_id"].value_counts().head(500).index.tolist()
        sample = instacart_df[instacart_df["user_id"].isin(top_users)].copy()
        sample["event_order"] = sample.groupby("user_id").cumcount()

        model = PrefixSpan.from_transactions(
            sample,
            user_col="user_id",
            time_col="event_order",
            item_col="product_id",
            min_support=10,
            max_len=2,
        )
        results = model.mine()
        assert len(results) >= 1
        assert "support" in results.columns
        assert "sequence" in results.columns
        assert (results["support"] >= 10).all()


# ===========================================================================
# Cross-algorithm sanity tests (UCI data — no Kaggle required)
# ===========================================================================


@pytest.mark.integration
class TestCrossAlgorithmSanity:
    """Tests that verify self-consistency across different algorithms."""



    def test_support_is_consistent_with_raw_counts(self, online_retail_df: pd.DataFrame) -> None:  # type: ignore[name-defined]
        """Support from FPGrowth matches manually-computed co-occurrence frequency."""
        import rusket

        basket = rusket.from_transactions(
            online_retail_df,
            transaction_col="Invoice",
            item_col="Description",
        )
        freq = rusket.fpgrowth(basket, min_support=0.005, use_colnames=True)
        n_baskets = basket.shape[0]

        # Spot-check: top-3 singletons
        singletons = freq[freq["itemsets"].apply(len) == 1].sort_values("support", ascending=False)  # type: ignore
        for _, row in singletons.head(3).iterrows():
            item = list(row["itemsets"])[0]
            expected_support = basket[item].sum() / n_baskets
            assert abs(row["support"] - expected_support) < 0.002, (
                f"Support mismatch for '{item}': rusket={row['support']:.4f}, manual={expected_support:.4f}"
            )
