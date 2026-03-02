from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import _rusket as _rust

if TYPE_CHECKING:
    import pandas as pd

_ALL_METRICS = [
    "antecedent support",
    "consequent support",
    "support",
    "confidence",
    "lift",
    "representativity",
    "leverage",
    "conviction",
    "zhangs_metric",
    "jaccard",
    "certainty",
    "kulczynski",
]


def association_rules(
    df: pd.DataFrame | Any,
    num_itemsets: int | None = None,
    df_orig: pd.DataFrame | None = None,
    null_values: bool = False,
    metric: str = "confidence",
    min_threshold: float = 0.8,
    support_only: bool = False,
    return_metrics: list[str] = _ALL_METRICS,
) -> pd.DataFrame:
    from rusket._dependencies import import_optional_dependency

    pd = import_optional_dependency("pandas")

    # Convert non-Pandas DataFrames to Pandas for the internal operations,
    # the returned type is re-converted to Spark/Polars by RuleMinerMixin.
    if not isinstance(df, pd.DataFrame):
        try:
            if hasattr(df, "to_pandas"):
                df = df.to_pandas()
            elif hasattr(df, "toPandas"):
                df = df.toPandas()
            else:
                raise TypeError(f"Expected a pandas/polars/spark DataFrame, got {type(df)}")
        except Exception as e:
            raise TypeError(f"Expected a pandas/polars/spark DataFrame, got {type(df)}: {e}") from e

    if "support" not in df.columns:
        raise ValueError("The input DataFrame must contain a 'support' column")
    if "itemsets" not in df.columns:
        raise ValueError("The input DataFrame must contain an 'itemsets' column")
    if df.empty:
        raise ValueError("The input DataFrame `df` containing the frequent itemsets is empty.")

    if num_itemsets is None:
        # Try to read from DataFrame metadata (set automatically by fpgrowth/mine/eclat)
        if "num_itemsets" in df.attrs:
            num_itemsets = int(df.attrs["num_itemsets"])
        else:
            # Infer from max support: the most frequent 1-item set has support == count/n,
            # so n = round(max_support_of_singleton / max_support).  A simpler robust
            # approach: n = round(1 / min_positive_support) is unreliable.  Instead,
            # look for singles (len==1) and use their max support as an approximation.
            singles = df[df["itemsets"].apply(len) == 1]
            if not singles.empty:
                max_support: float = singles["support"].astype(float).max()  # type: ignore[assignment]
                num_itemsets = round(1.0 / max_support)
            else:
                # Last resort: use the single highest support row as proxy
                max_support = df["support"].astype(float).max()  # type: ignore[assignment]
                num_itemsets = round(1.0 / max_support)

    first_iset = next(iter(df["itemsets"]))
    has_string_labels = any(isinstance(x, str) for x in first_iset)

    if has_string_labels:
        all_labels = []
        seen = set()
        for iset in df["itemsets"]:
            for x in iset:
                s = str(x)
                if s not in seen:
                    seen.add(s)
                    all_labels.append(s)
        label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
        idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}

        def _to_int_list(iset: Any) -> list[int]:
            return sorted(label_to_idx[str(x)] for x in iset)
    else:
        idx_to_label = {}

        def _to_int_list(iset: Any) -> list[int]:
            return sorted(int(x) for x in iset)

    itemsets_raw = [_to_int_list(iset) for iset in df["itemsets"]]
    supports_raw = list(df["support"].astype(float))

    antecedents_raw, consequents_raw, metric_cols = _rust.association_rules_inner(
        itemsets_raw,
        supports_raw,
        num_itemsets,
        metric if not support_only else "support",
        min_threshold,
        support_only,
        list(return_metrics),
    )

    if not antecedents_raw:
        return pd.DataFrame(columns=pd.Index(["antecedents", "consequents"] + list(return_metrics)))

    if has_string_labels:
        ant_fs = [tuple(idx_to_label[i] for i in a) for a in antecedents_raw]
        con_fs = [tuple(idx_to_label[i] for i in c) for c in consequents_raw]
    else:
        ant_fs = [tuple(a) for a in antecedents_raw]
        con_fs = [tuple(c) for c in consequents_raw]

    result = pd.DataFrame({"antecedents": ant_fs, "consequents": con_fs})
    for col_name, col_vals in zip(return_metrics, metric_cols, strict=False):
        result[col_name] = col_vals

    return result
