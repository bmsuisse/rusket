"""Meta-test: enforce API consistency conventions across all rusket classes.

This test introspects every public class in rusket and verifies that:
- All recommenders inherit from the correct base class.
- All classes use `verbose: int` (not bool) and `seed: int` (not random_state).
- All classes have `__repr__`, `fitted`, and `from_transactions`.
- Recommenders expose `recommend_items` and `recommend_users`.
- Miners expose `mine`, `association_rules`, and `recommend_for_cart`.
- All docstrings use NumPy-style formatting.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

import rusket
from rusket.model import BaseModel, ImplicitRecommender, Miner, RuleMinerMixin, SequentialRecommender

# ─── Class registries ────────────────────────────────────────────────────────

IMPLICIT_RECOMMENDERS = [rusket.ALS, rusket.BPR, rusket.EASE, rusket.ItemKNN, rusket.SVD, rusket.LightGCN]
SEQUENTIAL_RECOMMENDERS = [rusket.FPMC, rusket.SASRec]
ALL_RECOMMENDERS = IMPLICIT_RECOMMENDERS + SEQUENTIAL_RECOMMENDERS

RULE_MINERS = [rusket.FPGrowth, rusket.Eclat, rusket.AutoMiner]

# FM is standalone — no from_transactions, no recommend_items in the usual sense
STANDALONE = [rusket.FM]

ALL_CLASSES = ALL_RECOMMENDERS + RULE_MINERS + STANDALONE


# ─── Hierarchy ────────────────────────────────────────────────────────────────


class TestClassHierarchy:
    """Every recommender / miner must inherit from the correct base."""

    @pytest.mark.parametrize("cls", IMPLICIT_RECOMMENDERS, ids=lambda c: c.__name__)
    def test_implicit_recommender_base(self, cls: type) -> None:
        assert issubclass(cls, ImplicitRecommender), (
            f"{cls.__name__} must inherit ImplicitRecommender"
        )

    @pytest.mark.parametrize("cls", SEQUENTIAL_RECOMMENDERS, ids=lambda c: c.__name__)
    def test_sequential_recommender_base(self, cls: type) -> None:
        assert issubclass(cls, SequentialRecommender), (
            f"{cls.__name__} must inherit SequentialRecommender"
        )

    @pytest.mark.parametrize("cls", RULE_MINERS, ids=lambda c: c.__name__)
    def test_miner_base(self, cls: type) -> None:
        assert issubclass(cls, Miner), f"{cls.__name__} must inherit Miner"
        assert issubclass(cls, RuleMinerMixin), f"{cls.__name__} must inherit RuleMinerMixin"


# ─── __init__ signature conventions ──────────────────────────────────────────


# Deterministic models don't need seed (EASE, ItemKNN)
STOCHASTIC_MODELS = [
    cls for cls in ALL_RECOMMENDERS + STANDALONE
    if cls not in (rusket.EASE, rusket.ItemKNN)
]


class TestInitSignature:
    """All classes must use `verbose: int` and `seed: int`."""

    @pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
    def test_verbose_is_int(self, cls: type) -> None:
        sig = inspect.signature(cls.__init__)
        if "verbose" not in sig.parameters:
            pytest.skip(f"{cls.__name__} has no verbose param")

        p = sig.parameters["verbose"]
        # Default must be an int, not a bool
        assert isinstance(p.default, int) and not isinstance(p.default, bool), (
            f"{cls.__name__}.__init__ verbose default must be int, got {type(p.default).__name__}"
        )

    @pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
    def test_no_random_state(self, cls: type) -> None:
        sig = inspect.signature(cls.__init__)
        assert "random_state" not in sig.parameters, (
            f"{cls.__name__}.__init__ uses 'random_state' — rename to 'seed'"
        )

    @pytest.mark.parametrize("cls", STOCHASTIC_MODELS, ids=lambda c: c.__name__)
    def test_seed_param_exists(self, cls: type) -> None:
        sig = inspect.signature(cls.__init__)
        assert "seed" in sig.parameters, (
            f"{cls.__name__}.__init__ must have a 'seed' parameter"
        )


# ─── __repr__ ────────────────────────────────────────────────────────────────


class TestRepr:
    """Every class must define its own __repr__."""

    @pytest.mark.parametrize("cls", ALL_RECOMMENDERS, ids=lambda c: c.__name__)
    def test_recommender_has_repr(self, cls: type) -> None:
        # Must be defined on the class itself, not inherited from object
        assert "__repr__" in cls.__dict__, (
            f"{cls.__name__} must define __repr__"
        )


# ─── from_transactions ───────────────────────────────────────────────────────


class TestFromTransactions:
    """Non-standalone classes must have from_transactions."""

    @pytest.mark.parametrize("cls", ALL_RECOMMENDERS + RULE_MINERS, ids=lambda c: c.__name__)
    def test_has_from_transactions(self, cls: type) -> None:
        assert hasattr(cls, "from_transactions"), (
            f"{cls.__name__} must have from_transactions"
        )
        assert callable(cls.from_transactions)


# ─── Recommender methods ─────────────────────────────────────────────────────


class TestRecommenderMethods:
    """All recommenders must expose recommend_items and recommend_users."""

    @pytest.mark.parametrize("cls", ALL_RECOMMENDERS, ids=lambda c: c.__name__)
    def test_has_recommend_items(self, cls: type) -> None:
        assert hasattr(cls, "recommend_items"), (
            f"{cls.__name__} must have recommend_items"
        )

    @pytest.mark.parametrize("cls", ALL_RECOMMENDERS, ids=lambda c: c.__name__)
    def test_recommend_items_signature(self, cls: type) -> None:
        sig = inspect.signature(cls.recommend_items)
        params = list(sig.parameters.keys())
        # Must accept (self, user_id, n, ...) at minimum
        assert "user_id" in params or len(params) >= 2, (
            f"{cls.__name__}.recommend_items must accept user_id"
        )

    @pytest.mark.parametrize("cls", IMPLICIT_RECOMMENDERS, ids=lambda c: c.__name__)
    def test_has_recommend_users(self, cls: type) -> None:
        assert hasattr(cls, "recommend_users"), (
            f"{cls.__name__} must have recommend_users (even if it raises NotImplementedError)"
        )


# ─── Miner methods ───────────────────────────────────────────────────────────


class TestMinerMethods:
    """All miners must expose mine, association_rules, recommend_for_cart."""

    @pytest.mark.parametrize("cls", RULE_MINERS, ids=lambda c: c.__name__)
    def test_has_mine(self, cls: type) -> None:
        assert hasattr(cls, "mine"), f"{cls.__name__} must have mine()"

    @pytest.mark.parametrize("cls", RULE_MINERS, ids=lambda c: c.__name__)
    def test_has_association_rules(self, cls: type) -> None:
        assert hasattr(cls, "association_rules"), (
            f"{cls.__name__} must have association_rules()"
        )

    @pytest.mark.parametrize("cls", RULE_MINERS, ids=lambda c: c.__name__)
    def test_has_recommend_for_cart(self, cls: type) -> None:
        assert hasattr(cls, "recommend_for_cart"), (
            f"{cls.__name__} must have recommend_for_cart()"
        )

    @pytest.mark.parametrize("cls", RULE_MINERS, ids=lambda c: c.__name__)
    def test_deprecated_recommend_items_alias(self, cls: type) -> None:
        """RuleMinerMixin.recommend_items must still exist as backward-compat alias."""
        assert hasattr(cls, "recommend_items"), (
            f"{cls.__name__} must keep backward-compat recommend_items alias"
        )


# ─── Docstring style ─────────────────────────────────────────────────────────


class TestDocstrings:
    """Key methods must use NumPy-style docstrings (Parameters / Returns)."""

    @pytest.mark.parametrize("cls", ALL_RECOMMENDERS, ids=lambda c: c.__name__)
    def test_recommend_items_docstring_style(self, cls: type) -> None:
        doc = cls.recommend_items.__doc__
        if doc is None:
            pytest.skip(f"{cls.__name__}.recommend_items has no docstring")

        # NumPy-style uses "Parameters\n----------" or "Returns\n-------"
        # Google-style uses "Args:" or "Returns:"
        assert "Args:" not in doc, (
            f"{cls.__name__}.recommend_items uses Google-style docstring — use NumPy-style"
        )

    @pytest.mark.parametrize("cls", ALL_RECOMMENDERS, ids=lambda c: c.__name__)
    def test_class_docstring_exists(self, cls: type) -> None:
        assert cls.__doc__ is not None, f"{cls.__name__} must have a class docstring"


# ─── exclude_seen parameter ──────────────────────────────────────────────────


class TestExcludeSeen:
    """All recommender recommend_items must support exclude_seen."""

    @pytest.mark.parametrize("cls", ALL_RECOMMENDERS, ids=lambda c: c.__name__)
    def test_has_exclude_seen(self, cls: type) -> None:
        sig = inspect.signature(cls.recommend_items)
        assert "exclude_seen" in sig.parameters, (
            f"{cls.__name__}.recommend_items must have exclude_seen parameter"
        )
