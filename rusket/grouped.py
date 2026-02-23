from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def _is_spark(df: Any) -> bool:
    return getattr(type(df), "__module__", "").startswith("pyspark")


def _is_polars(df: Any) -> bool:
    return getattr(type(df), "__module__", "").startswith("polars")


def mine_grouped(
    df: Any,
    group_col: str,
    min_support: float = 0.5,
    max_len: int | None = None,
    method: str = "auto",
    use_colnames: bool = True,
) -> Any:
    """Distribute Market Basket Analysis across groups. Supports Polars, Pandas, and Spark."""
    if _is_spark(df):
        from .spark import mine_grouped as spark_mine_grouped

        return spark_mine_grouped(df, group_col, min_support, max_len, method, use_colnames)

    is_pl = _is_polars(df)
    if is_pl:
        import polars as pl

        res_dfs = []
        for name, group_df in df.group_by(group_col):
            if isinstance(name, tuple):
                name = name[0]
            drop_df = group_df.drop(group_col)
            from .mine import mine

            freq = mine(drop_df, min_support=min_support, max_len=max_len, method=method, use_colnames=use_colnames)
            if not freq.empty:
                freq.insert(0, group_col, name)
                res_dfs.append(pl.from_pandas(freq))
        if res_dfs:
            return pl.concat(res_dfs)
        return pl.DataFrame(schema={group_col: pl.Utf8, "support": pl.Float64, "itemsets": pl.List(pl.Utf8)})

    else:
        import pandas as pd

        res_dfs = []
        for name, group_df in df.groupby(group_col):
            if isinstance(name, tuple):
                name = name[0]
            drop_df = group_df.drop(columns=[group_col])
            from .mine import mine

            freq = mine(drop_df, min_support=min_support, max_len=max_len, method=method, use_colnames=use_colnames)
            if not freq.empty:
                freq.insert(0, group_col, name)
                res_dfs.append(freq)
        if res_dfs:
            return pd.concat(res_dfs, ignore_index=True)
        return pd.DataFrame(columns=[group_col, "support", "itemsets"])


def rules_grouped(
    df: Any,
    group_col: str,
    num_itemsets: dict[Any, int] | int,
    metric: str = "confidence",
    min_threshold: float = 0.8,
) -> Any:
    """Distribute Association Rule Mining across groups. Supports Polars, Pandas, and Spark."""
    if _is_spark(df):
        from .spark import rules_grouped as spark_rules_grouped

        return spark_rules_grouped(df, group_col, num_itemsets, metric, min_threshold)

    is_pl = _is_polars(df)
    if is_pl:
        import polars as pl

        res_dfs = []
        for name, group_df in df.group_by(group_col):
            if isinstance(name, tuple):
                name = name[0]
            n_tx = num_itemsets.get(name, 0) if isinstance(num_itemsets, dict) else num_itemsets
            from .association_rules import association_rules

            # Convert internally to pandas since association_rules expects pandas from mine natively under the hood
            res = association_rules(group_df.to_pandas(), num_itemsets=n_tx, metric=metric, min_threshold=min_threshold)
            if not res.empty:
                res.insert(0, group_col, name)
                res_dfs.append(pl.from_pandas(res))
        if res_dfs:
            return pl.concat(res_dfs)
        return pl.DataFrame(
            schema={
                group_col: pl.Utf8,
                "antecedents": pl.List(pl.Utf8),
                "consequents": pl.List(pl.Utf8),
                metric: pl.Float64,
            }
        )

    else:
        import pandas as pd

        res_dfs = []
        for name, group_df in df.groupby(group_col):
            if isinstance(name, tuple):
                name = name[0]
            n_tx = num_itemsets.get(name, 0) if isinstance(num_itemsets, dict) else num_itemsets
            from .association_rules import association_rules

            res = association_rules(group_df, num_itemsets=n_tx, metric=metric, min_threshold=min_threshold)
            if not res.empty:
                res.insert(0, group_col, name)
                res_dfs.append(res)
        if res_dfs:
            return pd.concat(res_dfs, ignore_index=True)
        return pd.DataFrame(columns=[group_col, "antecedents", "consequents", metric])


def prefixspan_grouped(
    df: Any,
    group_col: str,
    user_col: str,
    time_col: str,
    item_col: str,
    min_support: int = 1,
    max_len: int | None = None,
) -> Any:
    """Distribute Sequential Pattern Mining across groups. Supports Polars, Pandas, and Spark."""
    if _is_spark(df):
        from .spark import prefixspan_grouped as spark_prefixspan_grouped

        return spark_prefixspan_grouped(df, group_col, user_col, time_col, item_col, min_support, max_len)

    is_pl = _is_polars(df)
    if is_pl:
        import polars as pl

        res_dfs = []
        for name, group_df in df.group_by(group_col):
            if isinstance(name, tuple):
                name = name[0]
            from .prefixspan import PrefixSpan

            try:
                model = PrefixSpan.from_transactions(
                    group_df,
                    user_col=user_col,
                    time_col=time_col,
                    item_col=item_col,
                    min_support=min_support,
                    max_len=max_len,
                )
                res = model.mine()
                if not res.empty:
                    res.insert(0, group_col, name)
                    res_dfs.append(pl.from_pandas(res))
            except Exception:
                pass
        if res_dfs:
            return pl.concat(res_dfs)
        return pl.DataFrame(schema={group_col: pl.Utf8, "support": pl.Int64, "sequence": pl.List(pl.Utf8)})

    else:
        res_dfs = []
        for name, group_df in df.groupby(group_col):
            if isinstance(name, tuple):
                name = name[0]
            from .prefixspan import PrefixSpan

            try:
                model = PrefixSpan.from_transactions(
                    group_df,
                    user_col=user_col,
                    time_col=time_col,
                    item_col=item_col,
                    min_support=min_support,
                    max_len=max_len,
                )
                res = model.mine()
                if not res.empty:
                    res.insert(0, group_col, name)
                    res_dfs.append(res)
            except Exception:
                pass
        if res_dfs:
            return pd.concat(res_dfs, ignore_index=True)
        return pd.DataFrame(columns=[group_col, "support", "sequence"])


def hupm_grouped(
    df: Any,
    group_col: str,
    transaction_col: str,
    item_col: str,
    utility_col: str,
    min_utility: float,
    max_len: int | None = None,
) -> Any:
    """Distribute High-Utility Pattern Mining across groups. Supports Polars, Pandas, and Spark."""
    if _is_spark(df):
        from .spark import hupm_grouped as spark_hupm_grouped

        return spark_hupm_grouped(df, group_col, transaction_col, item_col, utility_col, min_utility, max_len)

    is_pl = _is_polars(df)
    if is_pl:
        import polars as pl

        res_dfs = []
        for name, group_df in df.group_by(group_col):
            if isinstance(name, tuple):
                name = name[0]
            from .hupm import HUPM

            try:
                model = HUPM.from_transactions(
                    group_df,
                    transaction_col=transaction_col,
                    item_col=item_col,
                    utility_col=utility_col,
                    min_utility=min_utility,
                    max_len=max_len,
                )
                res = model.mine()
                if not res.empty:
                    res.insert(0, group_col, name)
                    res_dfs.append(pl.from_pandas(res))
            except Exception:
                pass
        if res_dfs:
            return pl.concat(res_dfs)
        return pl.DataFrame(schema={group_col: pl.Utf8, "utility": pl.Float64, "itemset": pl.List(pl.Int64)})

    else:
        res_dfs = []
        for name, group_df in df.groupby(group_col):
            if isinstance(name, tuple):
                name = name[0]
            from .hupm import HUPM

            try:
                model = HUPM.from_transactions(
                    group_df,
                    transaction_col=transaction_col,
                    item_col=item_col,
                    utility_col=utility_col,
                    min_utility=min_utility,
                    max_len=max_len,
                )
                res = model.mine()
                if not res.empty:
                    res.insert(0, group_col, name)
                    res_dfs.append(res)
            except Exception:
                pass
        if res_dfs:
            return pd.concat(res_dfs, ignore_index=True)
        return pd.DataFrame(columns=[group_col, "utility", "itemset"])


def recommend_batches(
    df: Any,
    model: Any,
    user_col: str = "user_id",
    k: int = 5,
) -> Any:
    """Distribute Batch Recommendations across groups. Supports Polars, Pandas, and Spark."""
    if _is_spark(df):
        from .spark import recommend_batches as spark_recommend_batches

        return spark_recommend_batches(df, model, user_col, k)

    is_pl = _is_polars(df)

    # Model can be `Recommender` or `ALS`. Ensure we have predict_next_chunk
    try:
        from .recommend import Recommender

        recommender = model if isinstance(model, Recommender) else Recommender(model=model)
    except Exception:
        recommender = model  # Trust duck-typing

    try:
        if is_pl:
            res_df = recommender.predict_next_chunk(df.to_pandas(), user_col=user_col, k=k)
            import polars as pl

            return pl.from_pandas(res_df)
        else:
            import pandas as pd

            return recommender.predict_next_chunk(df, user_col=user_col, k=k)
    except Exception:
        if is_pl:
            import polars as pl

            return pl.DataFrame(schema={user_col: pl.Utf8, "recommended_items": pl.List(pl.Int64)})
        return pd.DataFrame(columns=[user_col, "recommended_items"])


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
    """Distribute ALS collaborative filtering across groups. Supports Polars, Pandas, and Spark."""
    if _is_spark(df):
        from .spark import als_grouped as spark_als_grouped

        return spark_als_grouped(
            df, group_col, user_col, item_col, rating_col, factors, regularization, alpha, iterations, k, **kwargs
        )

    is_pl = _is_polars(df)

    import warnings

    import pandas as pd

    from .als import ALS

    warnings.simplefilter("ignore", DeprecationWarning)

    res_dfs = []

    # Internal generic group handler
    def process_group(name, group_df):
        model = ALS(
            factors=factors,
            regularization=regularization,
            alpha=alpha,
            iterations=iterations,
            **kwargs,
        )
        model.fit_transactions(
            data=group_df,
            user_col=user_col,
            item_col=item_col,
            rating_col=rating_col,
        )

        user_labels = model._user_labels or list(range(model._n_users))
        records = []
        for internal_idx in range(model._n_users):
            try:
                item_ids, _ = model.recommend_items(user_id=internal_idx, n=k)
                records.append(
                    {
                        user_col: str(user_labels[internal_idx]),
                        "recommended_items": [int(x) for x in item_ids],
                    }
                )
            except Exception:
                pass

        res_df = pd.DataFrame(records, columns=[user_col, "recommended_items"])
        if not res_df.empty:
            res_df.insert(0, group_col, name)
            return res_df
        return None

    if is_pl:
        import polars as pl

        for name, group_df in df.group_by(group_col):
            if isinstance(name, tuple):
                name = name[0]
            res_df = process_group(name, group_df.to_pandas())
            if res_df is not None:
                res_dfs.append(pl.from_pandas(res_df))
        if res_dfs:
            return pl.concat(res_dfs)
        return pl.DataFrame(schema={group_col: pl.Utf8, user_col: pl.Utf8, "recommended_items": pl.List(pl.Int32)})
    else:
        for name, group_df in df.groupby(group_col):
            if isinstance(name, tuple):
                name = name[0]
            res_df = process_group(name, group_df)
            if res_df is not None:
                res_dfs.append(res_df)
        if res_dfs:
            return pd.concat(res_dfs, ignore_index=True)
        return pd.DataFrame(columns=[group_col, user_col, "recommended_items"])
