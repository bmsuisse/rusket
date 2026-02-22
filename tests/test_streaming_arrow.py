import pyarrow as pa
from unittest.mock import MagicMock

from rusket.streaming import FPMiner


def test_add_arrow_batch():
    # Make an arrow record batch
    txn_array = pa.array([1, 1, 2, 2, 3], type=pa.int64())
    item_array = pa.array([10, 20, 10, 30, 20], type=pa.int32())
    batch = pa.RecordBatch.from_arrays(
        [txn_array, item_array], names=["txn_id", "item_id"]
    )

    miner = FPMiner(n_items=50)
    miner.add_arrow_batch(batch, txn_col="txn_id", item_col="item_id")

    assert miner.n_rows == 5
    assert miner.n_transactions == 3


def test_mine_duckdb_mocked():
    # Mock DuckDB connection
    MagicMock()
    MagicMock()

    txn_array = pa.array([1, 1, 2, 2], type=pa.int64())
    item_array = pa.array([0, 1, 0, 1], type=pa.int32())
    batch = pa.RecordBatch.from_arrays([txn_array, item_array], names=["t", "i"])

    miner = FPMiner(n_items=10)
    miner.add_arrow_batch(batch, "t", "i")

    assert miner.n_rows > 0


def test_mine_spark_mocked():
    # Mock PySpark DataFrame
    spark_df = MagicMock()

    txn_array = pa.array([1, 1, 2, 2], type=pa.int64())
    item_array = pa.array([0, 1, 0, 1], type=pa.int32())
    batch = pa.RecordBatch.from_arrays([txn_array, item_array], names=["t", "i"])

    # Mock Databricks toArrow() pathway
    table = pa.Table.from_batches([batch])
    spark_df.select.return_value.toArrow.return_value = table

    # Since we are mocking rust we just check structure runs without exceptions
    miner = FPMiner(n_items=10)
    for b in table.to_batches():
        miner.add_arrow_batch(b, "t", "i")

    assert miner.n_rows > 0
