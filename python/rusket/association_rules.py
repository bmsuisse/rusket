"""Association rule generation backed by Rust + PyO3."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import _rusket as _rust

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
    df: pd.DataFrame,
    num_itemsets: int,
    df_orig: Optional[pd.DataFrame] = None,
    null_values: bool = False,
    metric: str = "confidence",
    min_threshold: float = 0.8,
    support_only: bool = False,
    return_metrics: list[str] = _ALL_METRICS,
) -> pd.DataFrame:
    """Generate association rules from a DataFrame of frequent itemsets.

    Parameters
    ----------
    df:
        DataFrame with columns ``['support', 'itemsets']`` as returned by
        :func:`fpgrowth`.
    num_itemsets:
        Total number of transactions in the original dataset.
    df_orig:
        Original (non-binarised) DataFrame – only needed when
        ``null_values=True``.
    null_values:
        Whether null-value correction should be applied
        (not yet implemented in Rust path; falls back gracefully).
    metric:
        Metric to filter rules on.  One of:
        ``'support'``, ``'confidence'``, ``'lift'``, ``'leverage'``,
        ``'conviction'``, ``'zhangs_metric'``, ``'jaccard'``,
        ``'certainty'``, ``'kulczynski'``, ``'representativity'``,
        ``'antecedent support'``, ``'consequent support'``.
    min_threshold:
        Minimum value of *metric* to include a rule.
    support_only:
        If ``True``, only compute support and fill all other metrics with NaN.
    return_metrics:
        List of metric column names to include in the result.

    Returns
    -------
    pandas.DataFrame
        Columns: ``antecedents``, ``consequents``, and the requested metrics.

    Raises
    ------
    ValueError
        If required columns are missing, df is empty, or an unknown metric is
        given.
    KeyError
        If antecedent/consequent support information is missing and
        ``support_only=False``.
    """
    if "support" not in df.columns:
        raise ValueError(
            "DataFrame needs to contain a 'support' column"
        )
    if "itemsets" not in df.columns:
        raise ValueError(
            "DataFrame needs to contain an 'itemsets' column"
        )
    if df.shape[0] == 0:
        raise ValueError(
            "The DataFrame of frequent itemsets is empty."
        )

    if not support_only and metric not in _ALL_METRICS:
        raise ValueError(
            "Metric must be 'confidence' or 'lift', got '{}'".format(metric)
        )

    # ------------------------------------------------------------------ #
    # Normalise itemsets to sorted lists of integers
    # Detect whether itemsets contain string labels and build a mapping.
    # ------------------------------------------------------------------ #
    first_iset = next(iter(df["itemsets"]))
    has_string_labels = any(isinstance(x, str) for x in first_iset)

    if has_string_labels:
        # Collect all unique labels and assign stable integer indices
        all_labels: list[str] = []
        seen: set[str] = set()
        for iset in df["itemsets"]:
            for x in iset:
                s = str(x)
                if s not in seen:
                    seen.add(s)
                    all_labels.append(s)
        label_to_idx: dict[str, int] = {lbl: i for i, lbl in enumerate(all_labels)}
        idx_to_label: dict[int, str] = {i: lbl for lbl, i in label_to_idx.items()}

        def _to_int_list(iset: object) -> list[int]:  # type: ignore[misc]
            return sorted(label_to_idx[str(x)] for x in iset)  # type: ignore[union-attr]
    else:
        idx_to_label = {}

        def _to_int_list(iset: object) -> list[int]:  # type: ignore[misc]
            return sorted(
                int(x) if not isinstance(x, np.generic) else int(x)  # type: ignore[arg-type]
                for x in iset  # type: ignore[union-attr]
            )

    itemsets_raw = [_to_int_list(iset) for iset in df["itemsets"]]
    supports_raw = list(df["support"].astype(float))

    # ------------------------------------------------------------------ #
    # Call Rust
    # ------------------------------------------------------------------ #
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
        return pd.DataFrame(columns=["antecedents", "consequents"] + list(return_metrics))

    # ------------------------------------------------------------------ #
    # Reconstruct frozensets – map back to string labels when applicable
    # ------------------------------------------------------------------ #
    if has_string_labels:
        ant_fs = [frozenset(idx_to_label[i] for i in a) for a in antecedents_raw]
        con_fs = [frozenset(idx_to_label[i] for i in c) for c in consequents_raw]
    else:
        ant_fs = [frozenset(a) for a in antecedents_raw]
        con_fs = [frozenset(c) for c in consequents_raw]

    result = pd.DataFrame({"antecedents": ant_fs, "consequents": con_fs})
    for col_name, col_vals in zip(return_metrics, metric_cols):
        result[col_name] = col_vals

    return result
