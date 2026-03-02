from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from .mine import mine


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
    method: str = "fpgrowth",
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
        Algorithm to use: 'fpgrowth', or 'eclat'.
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
        raise ValueError("mine_grouped requires use_colnames=True to safely manage PySpark schemas across partitions.")

    # We defer the pyspark imports so we don't crash if pyspark is not installed
    from rusket._dependencies import import_optional_dependency

    pa = import_optional_dependency("pyarrow")
    from rusket._dependencies import import_optional_dependency

    T = import_optional_dependency("pyspark.sql.types", "pyspark")

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
        # Extract the group identity (safe to grab from row 0)
        group_id = str(table.column(group_col)[0].as_py())

        # Drop the group column using native PyArrow — input is Arrow, stay in Arrow
        col_idx = table.schema.get_field_index(group_col)
        matrix_pd = table.remove_column(col_idx).to_pandas()

        # Call Rusket C-Extensions via the pandas matrix
        result_pd = mine(
            matrix_pd,
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
        # PySpark expects `string` and not `large_string` which pandas might create
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


def prefixspan_grouped(
    df: Any,
    group_col: str,
    user_col: str,
    time_col: str,
    item_col: str,
    min_support: int = 1,
    max_len: int | None = None,
) -> Any:
    """Distribute Sequential Pattern Mining (PrefixSpan) across PySpark partitions.

    This function groups a PySpark DataFrame by `group_col` and applies
    `PrefixSpan.from_transactions` to each group concurrently across the cluster.

    Parameters
    ----------
    df
        The input `pyspark.sql.DataFrame`.
    group_col
        The column to group by (e.g. `store_id`).
    user_col
        The column identifying the sequence within each group (e.g., `user_id` or `session_id`).
    time_col
        The column used for ordering events within a sequence.
    item_col
        The column containing the items.
    min_support
        The minimum absolute support (number of sequences a pattern must appear in).
    max_len
        Maximum length of the sequential patterns to mine.

    Returns
    -------
    pyspark.sql.DataFrame
        A PySpark DataFrame containing:
        - `group_col`
        - `support` (long/int64)
        - `sequence` (array of strings)
    """
    from rusket._dependencies import import_optional_dependency

    pd = import_optional_dependency("pandas")
    from rusket._dependencies import import_optional_dependency

    pa = import_optional_dependency("pyarrow")
    from rusket._dependencies import import_optional_dependency

    T = import_optional_dependency("pyspark.sql.types", "pyspark")

    schema = T.StructType(
        [
            T.StructField(group_col, T.StringType(), True),
            T.StructField("support", T.LongType(), True),
            T.StructField("sequence", T.ArrayType(T.StringType()), True),
        ]
    )

    def _prefixspan_group(table: pa.Table) -> pa.Table:
        from rusket.prefixspan import PrefixSpan

        group_id = str(table.column(group_col)[0].as_py())

        # Stay in PyArrow for manipulation — input is Arrow, convert to pandas for the algorithm
        input_pd = table.to_pandas()

        model = PrefixSpan.from_transactions(
            data=input_pd,
            user_col=user_col,
            time_col=time_col,
            item_col=item_col,
            min_support=min_support,
            max_len=max_len,
        )
        result_pd = model.mine()

        # Ensure items in the sequences are cast to string for the array<string> schema
        if not result_pd.empty:
            result_pd["sequence"] = result_pd["sequence"].apply(lambda seq: [str(x) for x in seq])

        if len(result_pd) == 0:
            return pa.Table.from_batches(
                [],
                schema=pa.schema(
                    [
                        (group_col, pa.string()),
                        ("support", pa.int64()),
                        ("sequence", pa.list_(pa.string())),
                    ]
                ),
            )

        result_pd.insert(0, group_col, group_id)

        out_table = pa.Table.from_pandas(result_pd)
        expected_schema = pa.schema(
            [
                (group_col, pa.string()),
                ("support", pa.int64()),
                ("sequence", pa.list_(pa.string())),
            ]
        )
        return out_table.cast(expected_schema)

    if hasattr(df.groupby(group_col), "applyInArrow"):
        return df.groupby(group_col).applyInArrow(_prefixspan_group, schema=schema)
    else:

        def _prefixspan_group_pd(pdf: pd.DataFrame) -> pd.DataFrame:
            from rusket.prefixspan import PrefixSpan

            group_id = str(pdf[group_col].iloc[0])

            model = PrefixSpan.from_transactions(
                data=pdf,
                user_col=user_col,
                time_col=time_col,
                item_col=item_col,
                min_support=min_support,
                max_len=max_len,
            )
            res = model.mine()
            if not res.empty:
                res["sequence"] = res["sequence"].apply(lambda seq: [str(x) for x in seq])

            res.insert(0, group_col, group_id)
            return res

        return df.groupby(group_col).applyInPandas(_prefixspan_group_pd, schema=schema)


def hupm_grouped(
    df: Any,
    group_col: str,
    transaction_col: str,
    item_col: str,
    utility_col: str,
    min_utility: float,
    max_len: int | None = None,
) -> Any:
    """Distribute High-Utility Pattern Mining (HUPM) across PySpark partitions.

    This function groups a PySpark DataFrame by `group_col` and applies
    `HUPM.from_transactions` to each group concurrently across the cluster.

    Parameters
    ----------
    df
        The input `pyspark.sql.DataFrame`.
    group_col
        The column to group by (e.g. `store_id`).
    transaction_col
        The column identifying the transaction within each group.
    item_col
        The column containing the numeric item IDs.
    utility_col
        The column containing the numeric utility (e.g., profit) of the item in the transaction.
    min_utility
        The minimum total utility required to consider a pattern "high-utility".
    max_len
        Maximum length of the itemsets to mine.

    Returns
    -------
    pyspark.sql.DataFrame
        A PySpark DataFrame containing:
        - `group_col`
        - `utility` (double/float64)
        - `itemset` (array of longs/int64)
    """
    from rusket._dependencies import import_optional_dependency

    pa = import_optional_dependency("pyarrow")

    from rusket._dependencies import import_optional_dependency

    T = import_optional_dependency("pyspark.sql.types", "pyspark")

    schema = T.StructType(
        [
            T.StructField(group_col, T.StringType(), True),
            T.StructField("utility", T.DoubleType(), True),
            T.StructField("itemset", T.ArrayType(T.LongType()), True),
        ]
    )

    def _hupm_group(table: pa.Table) -> pa.Table:
        from rusket.hupm import HUPM

        group_id = str(table.column(group_col)[0].as_py())

        # Stay in PyArrow for manipulation — input is Arrow, convert to pandas for the algorithm
        input_pd = table.to_pandas()

        model = HUPM.from_transactions(
            data=input_pd,
            transaction_col=transaction_col,
            item_col=item_col,
            utility_col=utility_col,
            min_utility=min_utility,
            max_len=max_len,
        )
        result_pd = model.mine()

        # Ensure items in the sequences are cast to int64 for the array<long> schema
        if not result_pd.empty:
            result_pd["itemset"] = result_pd["itemset"].apply(lambda seq: [int(x) for x in seq])

        if len(result_pd) == 0:
            return pa.Table.from_batches(
                [],
                schema=pa.schema(
                    [
                        (group_col, pa.string()),
                        ("utility", pa.float64()),
                        ("itemset", pa.list_(pa.int64())),
                    ]
                ),
            )

        result_pd.insert(0, group_col, group_id)

        out_table = pa.Table.from_pandas(result_pd)
        expected_schema = pa.schema(
            [
                (group_col, pa.string()),
                ("utility", pa.float64()),
                ("itemset", pa.list_(pa.int64())),
            ]
        )
        return out_table.cast(expected_schema)

    if hasattr(df.groupby(group_col), "applyInArrow"):
        return df.groupby(group_col).applyInArrow(_hupm_group, schema=schema)
    else:

        def _hupm_group_pd(pdf: pd.DataFrame) -> pd.DataFrame:
            from rusket.hupm import HUPM

            group_id = str(pdf[group_col].iloc[0])

            model = HUPM.from_transactions(
                data=pdf,
                transaction_col=transaction_col,
                item_col=item_col,
                utility_col=utility_col,
                min_utility=min_utility,
                max_len=max_len,
            )
            res = model.mine()
            if not res.empty:
                res["itemset"] = res["itemset"].apply(lambda seq: [int(x) for x in seq])

            res.insert(0, group_col, group_id)
            return res

        return df.groupby(group_col).applyInPandas(_hupm_group_pd, schema=schema)


def rules_grouped(
    df: Any,
    group_col: str,
    num_itemsets: dict[Any, int] | int,
    metric: str = "confidence",
    min_threshold: float = 0.8,
) -> Any:
    """Distribute Association Rule Mining across PySpark partitions.

    This takes the frequent itemsets DataFrame (output of `mine_grouped`)
    and applies `association_rules` uniformly across the groups.

    Parameters
    ----------
    df
        The PySpark `DataFrame` containing frequent itemsets.
    group_col
        The column to group by.
    num_itemsets
        A dictionary mapping group IDs to their total transaction count,
        or a single integer if all groups have the same number of transactions.
    metric
        The metric to filter by (e.g. "confidence", "lift").
    min_threshold
        The minimal threshold for the evaluation metric.

    Returns
    -------
    pyspark.sql.DataFrame
        A DataFrame containing antecedents, consequents, and all rule metrics,
        prepended with the `group_col`.
    """
    from rusket._dependencies import import_optional_dependency

    pd = import_optional_dependency("pandas")
    from rusket._dependencies import import_optional_dependency

    pa = import_optional_dependency("pyarrow")
    from rusket._dependencies import import_optional_dependency

    T = import_optional_dependency("pyspark.sql.types", "pyspark")

    all_metrics = [
        "antecedent support",
        "consequent support",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction",
        "zhangs_metric",
        "jaccard",
        "certainty",
        "kulczynski",
    ]

    schema_fields = [
        T.StructField(group_col, T.StringType(), True),
        T.StructField("antecedents", T.ArrayType(T.StringType()), True),
        T.StructField("consequents", T.ArrayType(T.StringType()), True),
    ]
    for m in all_metrics:
        schema_fields.append(T.StructField(m, T.DoubleType(), True))

    schema = T.StructType(schema_fields)

    def _rules_group(table: pa.Table) -> pa.Table:
        from rusket.association_rules import association_rules

        group_id = str(table.column(group_col)[0].as_py())

        if isinstance(num_itemsets, dict):
            key_int = int(group_id) if group_id.isdigit() else group_id
            num_tx = int(num_itemsets.get(group_id, num_itemsets.get(key_int, 0)))  # type: ignore
        else:
            num_tx = int(num_itemsets)

        if num_tx is None:
            num_tx = 0

        # Input is Arrow — convert directly to pandas, no Polars hop needed
        pdf = table.to_pandas()

        res_df = association_rules(
            df=pdf,
            num_itemsets=num_tx,
            metric=metric,
            min_threshold=min_threshold,
            return_metrics=all_metrics,
        )

        if not res_df.empty:
            res_df["antecedents"] = res_df["antecedents"].apply(list)
            res_df["consequents"] = res_df["consequents"].apply(list)
        else:
            res_df = pd.DataFrame(columns=["antecedents", "consequents"] + all_metrics)  # type: ignore[reportArgumentType]

        res_df.insert(0, group_col, group_id)

        # Ensure correct PyArrow schema types
        schema_pa = [
            (group_col, pa.string()),
            ("antecedents", pa.list_(pa.string())),
            ("consequents", pa.list_(pa.string())),
        ]
        for m in all_metrics:
            schema_pa.append((m, pa.float64()))

        return pa.Table.from_pandas(res_df).cast(pa.schema(schema_pa))

    if hasattr(df.groupby(group_col), "applyInArrow"):
        return df.groupby(group_col).applyInArrow(_rules_group, schema=schema)
    else:

        def _rules_group_pd(pdf: pd.DataFrame) -> pd.DataFrame:
            from rusket.association_rules import association_rules

            group_id = str(pdf[group_col].iloc[0])

            if isinstance(num_itemsets, dict):
                # Try string key, then fallback to int key
                key_int = int(group_id) if group_id.isdigit() else group_id
                num_tx = int(num_itemsets.get(group_id, num_itemsets.get(key_int, 0)))  # type: ignore
            else:
                num_tx = int(num_itemsets)

            if num_tx is None:
                num_tx = 0

            res_df = association_rules(
                df=pdf,
                num_itemsets=num_tx,
                metric=metric,
                min_threshold=min_threshold,
                return_metrics=all_metrics,
            )
            if not res_df.empty:
                res_df["antecedents"] = res_df["antecedents"].apply(list)
                res_df["consequents"] = res_df["consequents"].apply(list)
            else:
                res_df = pd.DataFrame(columns=["antecedents", "consequents"] + all_metrics)  # type: ignore[reportArgumentType]

            res_df.insert(0, group_col, group_id)
            return res_df

        return df.groupby(group_col).applyInPandas(_rules_group_pd, schema=schema)


def recommend_batches(
    df: Any,
    model: Any,
    user_col: str = "user_id",
    k: int = 5,
) -> Any:
    """Distribute Batch Recommendations across PySpark partitions.

    This function uses `mapInArrow` to process partitions of users concurrently,
    applying a pre-fitted `Recommender` (or `ALS`) to each chunk.

    Parameters
    ----------
    df
        The PySpark `DataFrame` containing user histories (must contain `user_col`).
    model
        The pre-trained `Recommender` or `ALS` model instance to use for scoring.
    user_col
        The column identifying the user.
    k
        The number of top recommendations to return per user.

    Returns
    -------
    pyspark.sql.DataFrame
        A DataFrame with two columns:
        - `user_col`
        - `recommended_items` (array of longs/int64)
    """
    from rusket._dependencies import import_optional_dependency

    pd = import_optional_dependency("pandas")
    from rusket._dependencies import import_optional_dependency

    T = import_optional_dependency("pyspark.sql.types", "pyspark")

    # If it's a raw ALS model, wrap it
    try:
        from rusket.recommend import Recommender

        recommender = model if isinstance(model, Recommender) else Recommender(model=model)
    except Exception:
        recommender = model  # Trust the duck-typing

    schema = T.StructType(
        [
            T.StructField(user_col, T.StringType(), True),
            T.StructField("recommended_items", T.ArrayType(T.IntegerType()), True),
        ]
    )

    # Broadcast the recommender instead of relying on RDD closure serialization
    sc = df.sparkSession.sparkContext
    b_recommender = sc.broadcast(recommender)

    def _recommend_row(row):
        import numpy as np

        # row is a pyspark.sql.Row
        df_single = pd.DataFrame([row.asDict()])
        rec = b_recommender.value
        res = rec.predict_next_chunk(df_single, user_col=user_col, k=k)
        u_id = str(res.iloc[0][user_col])

        # Ensure native Python integers for Spark ArrayType
        seq = res.iloc[0]["recommended_items"]
        if isinstance(seq, np.ndarray):
            items = [int(x) for x in seq]
        else:
            items = [int(x) for x in list(seq)]

        return (u_id, items)

    # Using RDD map avoids the PyArrow ListType serialization bugs entirely
    rdd = df.rdd.map(_recommend_row)
    return df.sparkSession.createDataFrame(rdd, schema=schema)


def als_grouped(
    df: Any,
    group_col: str,
    user_col: str,
    item_col: str,
    rating_col: str | None = None,
    factors: int = 64,
    regularization: float = 0.01,
    alpha: float = 40.0,
    iterations: int = 15,
    k: int = 10,
    **kwargs: Any,
) -> Any:
    """Distribute ALS collaborative filtering across PySpark partitions.

    This function groups a PySpark DataFrame by `group_col` and applies
    `ALS` to each group concurrently across the cluster. After fitting the model,
    it returns the top `k` recommendations for each user in the group.

    Parameters
    ----------
    df
        The input `pyspark.sql.DataFrame`.
    group_col
        The column to group by (e.g. `store_id`).
    user_col
        The column identifying the user.
    item_col
        The column identifying the item.
    rating_col
        The column containing the rating or interaction weight. If None, assumes implicit feedback.
    factors
        Number of latent factors for ALS.
    regularization
        L2 regularization weight for ALS.
    alpha
        Confidence scaling for implicit ALS: ``confidence = 1 + alpha * r``.
    iterations
        Number of ALS outer iterations.
    k
        The number of top recommendations to return per user.
    **kwargs
        Additional arguments passed to `ALS` initialization (e.g., `cg_iters`, `use_cholesky`, `anderson_m`).

    Returns
    -------
    pyspark.sql.DataFrame
        A PySpark DataFrame containing:
        - `group_col`
        - `user_col`
        - `recommended_items` (array of ints)
    """
    from rusket._dependencies import import_optional_dependency

    pd = import_optional_dependency("pandas")
    from rusket._dependencies import import_optional_dependency

    pa = import_optional_dependency("pyarrow")
    from rusket._dependencies import import_optional_dependency

    T = import_optional_dependency("pyspark.sql.types", "pyspark")

    schema = T.StructType(
        [
            T.StructField(group_col, T.StringType(), True),
            T.StructField(user_col, T.StringType(), True),
            T.StructField("recommended_items", T.ArrayType(T.IntegerType()), True),
        ]
    )

    def _als_group(table: pa.Table) -> pa.Table:
        from rusket.als import ALS

        group_id = str(table.column(group_col)[0].as_py())

        # Input is Arrow — convert directly to pandas for the algorithm
        input_pd = table.to_pandas()

        try:
            import warnings

            warnings.simplefilter("ignore", DeprecationWarning)
            model = ALS(
                factors=factors,
                regularization=regularization,
                alpha=alpha,
                iterations=iterations,
                **kwargs,
            )
            # Use fit_transactions to fit mapping dictionaries and model
            model.fit_transactions(
                data=input_pd,
                user_col=user_col,
                item_col=item_col,
                rating_col=rating_col,
            )

            # Recommend for all unique users in this group partition.
            # We iterate over internal model indices (0..n_users-1) and map back
            # to external user IDs via model._user_labels.
            user_labels = model._user_labels or list(range(model._n_users))
            records = []
            for internal_idx in range(model._n_users):
                item_ids, _ = model.recommend_items(user_id=internal_idx, n=k)
                records.append(
                    {
                        user_col: str(user_labels[internal_idx]),
                        "recommended_items": [int(x) for x in item_ids],
                    }
                )

            res_df = pd.DataFrame(records, columns=[user_col, "recommended_items"])  # type: ignore[reportArgumentType]

        except Exception as e:
            raise RuntimeError(f"als_grouped worker failed for group {group_id!r}: {e}") from e

        if len(res_df) == 0:
            return pa.Table.from_batches(
                [],
                schema=pa.schema(
                    [
                        (group_col, pa.string()),
                        (user_col, pa.string()),
                        ("recommended_items", pa.list_(pa.int32())),
                    ]
                ),
            )

        res_df.insert(0, group_col, group_id)

        out_table = pa.Table.from_pandas(res_df)
        expected_schema = pa.schema(
            [
                (group_col, pa.string()),
                (user_col, pa.string()),
                ("recommended_items", pa.list_(pa.int32())),
            ]
        )
        return out_table.cast(expected_schema)

    if hasattr(df.groupby(group_col), "applyInArrow"):
        return df.groupby(group_col).applyInArrow(_als_group, schema=schema)
    else:

        def _als_group_pd(pdf: pd.DataFrame) -> pd.DataFrame:
            from rusket.als import ALS

            group_id = str(pdf[group_col].iloc[0])

            try:
                import warnings

                warnings.simplefilter("ignore", DeprecationWarning)
                model = ALS(
                    factors=factors,
                    regularization=regularization,
                    alpha=alpha,
                    iterations=iterations,
                    **kwargs,
                )
                model.fit_transactions(
                    data=pdf,
                    user_col=user_col,
                    item_col=item_col,
                    rating_col=rating_col,
                )

                # Iterate internal 0-based indices, map to external IDs via _user_labels
                user_labels = model._user_labels or list(range(model._n_users))
                records = []
                for internal_idx in range(model._n_users):
                    item_ids, _ = model.recommend_items(user_id=internal_idx, n=k)
                    records.append(
                        {
                            user_col: str(user_labels[internal_idx]),
                            "recommended_items": [int(x) for x in item_ids],
                        }
                    )

                res_df = pd.DataFrame(records, columns=[user_col, "recommended_items"])  # type: ignore[reportArgumentType]

            except Exception:
                res_df = pd.DataFrame(columns=[user_col, "recommended_items"])  # type: ignore[reportArgumentType]

            res_df.insert(0, group_col, group_id)
            return res_df

        return df.groupby(group_col).applyInPandas(_als_group_pd, schema=schema)
