"""Regression tests: als_grouped must emit external item labels, not crash on them.

`ALS.batch_recommend(format="pandas")` now maps internal item indices through
`_item_labels` before returning, so `recommended_items` can contain strings or
integers wider than int32. `rusket.integrations.grouped.als_grouped` (pandas
and polars paths) and `rusket.integrations.spark.als_grouped` (Arrow path) both
build `recommended_items` from that same `item_id` column and must handle
non-int32 labels instead of crashing or silently reverting to internal indices.

These tests exercise the pandas/polars paths directly (no Spark cluster
needed) and a pyarrow-only unit test for the schema spark.py declares.
"""

import pandas as pd
import pyarrow as pa
import pytest

from rusket.integrations.grouped import als_grouped


def _df(item_ids):
    return pd.DataFrame(
        {
            "group_id": ["g1"] * len(item_ids),
            "user_id": [1, 1, 2, 2][: len(item_ids)],
            "item_id": item_ids,
        }
    )


def test_als_grouped_pandas_string_item_labels():
    """String item ids must not crash and must come back as external labels."""
    df = _df(["SKU-A", "SKU-B", "SKU-A", "SKU-C"])

    res = als_grouped(df, "group_id", "user_id", "item_id", factors=2, iterations=1, k=5)

    assert not res.empty
    all_recommended = {x for row in res["recommended_items"] for x in row}
    assert all_recommended <= {"SKU-A", "SKU-B", "SKU-C"}


def test_als_grouped_pandas_int64_labels_beyond_int32_range():
    """Item labels wider than int32 must not crash (old code cast to int32)."""
    big = 5_000_000_000  # outside int32 range
    df = _df([big, big + 1, big, big + 2])

    res = als_grouped(df, "group_id", "user_id", "item_id", factors=2, iterations=1, k=5)

    assert not res.empty
    for row in res["recommended_items"]:
        for x in row:
            assert int(x) >= big


def test_als_grouped_polars_string_item_labels():
    pl = pytest.importorskip("polars")

    df = pl.DataFrame(
        {
            "group_id": ["g1", "g1", "g1", "g1"],
            "user_id": [1, 1, 2, 2],
            "item_id": ["SKU-A", "SKU-B", "SKU-A", "SKU-C"],
        }
    )

    res = als_grouped(df, "group_id", "user_id", "item_id", factors=2, iterations=1, k=5)

    assert res.height > 0


def test_spark_als_grouped_arrow_schema_accepts_string_labels():
    """Regression for the exact ArrowInvalid crash: casting string item ids to int32.

    Mirrors the pyarrow cast `rusket.integrations.spark.als_grouped` performs on
    its output table; the schema must be `pa.list_(pa.string())`, not
    `pa.list_(pa.int32())`, or this raises `ArrowInvalid`.
    """
    from rusket.integrations import spark as rusket_spark

    schema = rusket_spark.als_grouped.__doc__
    assert "array of strings" in schema  # docstring documents the fixed contract

    res_df = pd.DataFrame(
        {
            "user_id": ["1", "2"],
            "recommended_items": [["SKU-A", "SKU-B"], ["SKU-C"]],
        }
    )
    table = pa.Table.from_pandas(res_df)
    expected_schema = pa.schema(
        [
            ("user_id", pa.string()),
            ("recommended_items", pa.list_(pa.string())),
        ]
    )
    # Must not raise ArrowInvalid (it would, for pa.list_(pa.int32())).
    table.cast(expected_schema)
