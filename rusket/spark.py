from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .mine import mine

if TYPE_CHECKING:
    import pandas as pd


def to_spark(spark_session: Any, df: Any) -> Any:
    """Convert a Pandas or Polars DataFrame into a PySpark DataFrame.

    Parameters
    ----------
    spark_session
        The active PySpark `SparkSession`.
    df
        The `pd.DataFrame` or `pl.DataFrame` to convert.

    Returns
    -------
    pyspark.sql.DataFrame
        The resulting PySpark DataFrame.
    """
    mod = getattr(df, "__module__", "") or ""
    if mod.startswith("polars"):
        df = df.to_pandas()  # PySpark does not natively accept Polars yet

    return spark_session.createDataFrame(df)


def mine_grouped(
    df: Any,
    group_col: str,
    min_support: float = 0.5,
    max_len: int | None = None,
    method: str = "auto",
    use_colnames: bool = True,
) -> Any:
    """Distribute Market Basket Analysis across PySpark partitions.

    This function groups a PySpark DataFrame by `group_col` and applies
    `rusket.mine` to each group concurrently across the cluster.

    It assumes the input PySpark DataFrame is formatted like a dense
    boolean matrix (One-Hot Encoded) per group, where rows are transactions.

    Parameters
    ----------
    df
        The input `pyspark.sql.DataFrame`.
    group_col
        The column to group by (e.g. `store_id`).
    min_support
        Minimum support threshold.
    max_len
        Maximum itemset length.
    method
        Algorithm to use: 'auto', 'fpgrowth', or 'eclat'.
    use_colnames
        If True, returns item names instead of column indices.
        Must be True for PySpark `applyInArrow` schema consistency.

    Returns
    -------
    pyspark.sql.DataFrame
        A PySpark DataFrame containing:
        - `group_col`
        - `support` (float)
        - `itemsets` (array of strings)
    """
    if not use_colnames:
        raise ValueError(
            "mine_grouped requires use_colnames=True to safely manage PySpark schemas across partitions."
        )

    # We defer the pyspark imports so we don't crash if pyspark is not installed
    import pyarrow as pa
    import pyspark.sql.types as T

    # 1. Define the output schema for the Spark UDF
    # PySpark Grouped Map UDF requires an exact schema
    schema = T.StructType(
        [
            T.StructField(group_col, T.StringType(), True),
            T.StructField("support", T.DoubleType(), True),
            T.StructField("itemsets", T.ArrayType(T.StringType()), True),
        ]
    )

    # 2. Define the grouped arrow execution function
    def _mine_group(table: pa.Table) -> pa.Table:
        """Runs within each Spark Task (Node) on an Arrow Table."""
        import polars as pl

        # Extract the group identity (safe to grab from row 0)
        group_id = str(table.column(group_col)[0].as_py())

        # Convert the Arrow Table to Polars for zero-copy
        pl_df = pl.from_arrow(table)

        # Drop the group column so we only pass the OHE matrix into Rust
        pl_matrix = pl_df.drop(group_col)

        # Call Rusket C-Extensions directly on the Arrow Buffers!
        # This will return a Pandas DataFrame
        result_pd = mine(
            pl_matrix,
            min_support=min_support,
            max_len=max_len,
            method=method,
            use_colnames=True,
        )

        # If no itemsets were found
        if len(result_pd) == 0:
            return pa.Table.from_batches(
                [],
                schema=pa.schema(
                    [
                        (group_col, pa.string()),
                        ("support", pa.float64()),
                        ("itemsets", pa.list_(pa.string())),
                    ]
                ),
            )

        # Re-attach the group identifier for PySpark tracking
        result_pd.insert(0, group_col, group_id)

        # Yield back as pyarrow table, ensuring we cast to the exact expected schema
        # PySpark expects `string` and not `large_string` which Polars/Pandas might create
        out_table = pa.Table.from_pandas(result_pd)
        expected_schema = pa.schema(
            [
                (group_col, pa.string()),
                ("support", pa.float64()),
                ("itemsets", pa.list_(pa.string())),
            ]
        )
        return out_table.cast(expected_schema)

    # 3. Apply the grouped Native Arrow UDF across the PySpark Cluster
    if hasattr(df.groupby(group_col), "applyInArrow"):
        return df.groupby(group_col).applyInArrow(_mine_group, schema=schema)
    else:
        # Fallback for PySpark < 3.4
        def _mine_group_pd(pdf: pd.DataFrame) -> pd.DataFrame:
            group_id = str(pdf[group_col].iloc[0])
            drop_df = pdf.drop(columns=[group_col])
            res = mine(
                drop_df,
                min_support=min_support,
                max_len=max_len,
                method=method,
                use_colnames=True,
            )
            res.insert(0, group_col, group_id)
            return res

        return df.groupby(group_col).applyInPandas(_mine_group_pd, schema=schema)
