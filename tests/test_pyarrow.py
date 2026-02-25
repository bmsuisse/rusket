"""Tests for PyArrow Table as first-class input/output in rusket algorithms.

Every test follows the same pattern:
  pa.Table in  →  rusket algorithm  →  pa.Table out
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import rusket


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def long_format_table() -> pa.Table:
    """Long-format transactional PyArrow Table (order_id, item)."""
    return pa.table(
        {
            "order_id": pa.array([1, 1, 1, 2, 2, 3, 3, 3, 4], type=pa.int64()),
            "item": pa.array(["milk", "bread", "butter", "milk", "bread", "milk", "butter", "eggs", "eggs"]),
        }
    )


@pytest.fixture()
def ohe_table(long_format_table: pa.Table) -> pa.Table:
    """One-hot-encoded boolean PyArrow Table produced by from_transactions."""
    return rusket.from_transactions(long_format_table)


# ---------------------------------------------------------------------------
# from_transactions / from_arrow
# ---------------------------------------------------------------------------


def test_from_transactions_returns_arrow(long_format_table: pa.Table) -> None:
    result = rusket.from_transactions(long_format_table)
    assert isinstance(result, pa.Table), f"Expected pa.Table, got {type(result)}"
    # All columns must be boolean
    for field in result.schema:
        assert pa.types.is_boolean(field.type), f"Column {field.name!r} is not bool: {field.type}"


def test_from_arrow_convenience(long_format_table: pa.Table) -> None:
    result = rusket.from_arrow(long_format_table)
    assert isinstance(result, pa.Table)
    assert result.schema == rusket.from_transactions(long_format_table).schema


def test_from_transactions_columns(long_format_table: pa.Table) -> None:
    result = rusket.from_transactions(long_format_table)
    items = set(result.schema.names)
    assert items == {"bread", "butter", "eggs", "milk"}


def test_from_transactions_arrow_explicit_cols() -> None:
    table = pa.table({"txn": [1, 1, 2], "prod": ["a", "b", "a"]})
    result = rusket.from_transactions(table, transaction_col="txn", item_col="prod")
    assert isinstance(result, pa.Table)
    assert set(result.schema.names) == {"a", "b"}


def test_from_transactions_min_item_count(long_format_table: pa.Table) -> None:
    """Items appearing < min_item_count times should be excluded."""
    result = rusket.from_transactions(long_format_table, min_item_count=3)
    assert isinstance(result, pa.Table)
    # butter appears 2x, eggs 2x; milk 3x, bread 2x — so only milk with min=3
    assert "eggs" not in result.schema.names or result.num_columns < 4


# ---------------------------------------------------------------------------
# fpgrowth / eclat / mine
# ---------------------------------------------------------------------------


def test_fpgrowth_returns_arrow(ohe_table: pa.Table) -> None:
    result = rusket.fpgrowth(ohe_table, min_support=0.5, use_colnames=True)
    assert isinstance(result, pa.Table), f"Expected pa.Table, got {type(result)}"
    assert "support" in result.schema.names
    assert "itemsets" in result.schema.names


def test_eclat_returns_arrow(ohe_table: pa.Table) -> None:
    result = rusket.eclat(ohe_table, min_support=0.5, use_colnames=True)
    assert isinstance(result, pa.Table), f"Expected pa.Table, got {type(result)}"
    assert "support" in result.schema.names


def test_mine_returns_arrow(ohe_table: pa.Table) -> None:
    result = rusket.mine(ohe_table, min_support=0.5, use_colnames=True)
    assert isinstance(result, pa.Table), f"Expected pa.Table, got {type(result)}"
    assert "support" in result.schema.names


def test_fpgrowth_without_colnames(ohe_table: pa.Table) -> None:
    result = rusket.fpgrowth(ohe_table, min_support=0.5, use_colnames=False)
    assert isinstance(result, pa.Table)
    # itemsets should contain integer indices, not strings
    assert result.num_rows > 0


# ---------------------------------------------------------------------------
# FPGrowth / Eclat OOP API
# ---------------------------------------------------------------------------


def test_fpgrowth_class_mine(ohe_table: pa.Table) -> None:
    model = rusket.FPGrowth(ohe_table, min_support=0.5, use_colnames=True)
    result = model.mine()
    assert isinstance(result, pa.Table)


def test_eclat_class_mine(ohe_table: pa.Table) -> None:
    model = rusket.Eclat(ohe_table, min_support=0.5, use_colnames=True)
    result = model.mine()
    assert isinstance(result, pa.Table)


def test_fpgrowth_class_from_transactions(long_format_table: pa.Table) -> None:
    model = rusket.FPGrowth.from_transactions(long_format_table, min_support=0.5, use_colnames=True)
    result = model.mine()
    assert isinstance(result, pa.Table)


# ---------------------------------------------------------------------------
# association_rules round-trip
# ---------------------------------------------------------------------------


def test_association_rules_round_trip(ohe_table: pa.Table) -> None:
    freq_table = rusket.fpgrowth(ohe_table, min_support=0.5, use_colnames=True)
    assert isinstance(freq_table, pa.Table)

    # association_rules accepts the pa.Table directly (it converts internally)
    rules = rusket.association_rules(freq_table, metric="confidence", min_threshold=0.5)
    # association_rules always returns pd.DataFrame (not type-mirrored), which is fine
    import pandas as pd

    assert isinstance(rules, pd.DataFrame)
    assert "antecedents" in rules.columns


def test_fpgrowth_class_association_rules(ohe_table: pa.Table) -> None:
    model = rusket.FPGrowth(ohe_table, min_support=0.5, use_colnames=True)
    rules = model.fit().association_rules(metric="confidence", min_threshold=0.5)
    assert isinstance(rules, pa.Table)
    assert "antecedents" in rules.schema.names
    assert "consequents" in rules.schema.names


# ---------------------------------------------------------------------------
# ALS / BPR from_transactions with PyArrow input
# ---------------------------------------------------------------------------


def test_als_from_transactions_arrow() -> None:
    """ALS should accept a PyArrow Table in long format and train successfully."""
    table = pa.table(
        {
            "user": pa.array([1, 1, 2, 2, 3, 3, 4], type=pa.int64()),
            "item": pa.array([10, 20, 10, 30, 20, 30, 10], type=pa.int64()),
        }
    )
    model = rusket.ALS(factors=4, iterations=3)
    model = model.from_transactions(table, transaction_col="user", item_col="item")
    model.fit(model._prepared_interactions)
    assert model.fitted


def test_bpr_from_transactions_arrow() -> None:
    """BPR should accept a PyArrow Table in long format."""
    table = pa.table(
        {
            "user": pa.array([1, 1, 2, 2, 3], type=pa.int64()),
            "item": pa.array([10, 20, 10, 30, 20], type=pa.int64()),
        }
    )
    model = rusket.BPR(factors=4, iterations=3)
    model = model.from_transactions(table, transaction_col="user", item_col="item")
    model.fit(model._prepared_interactions)
    assert model.fitted
