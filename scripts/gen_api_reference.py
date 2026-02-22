#!/usr/bin/env python3
"""gen_api_reference.py — Auto-generate docs/api-reference.md from Python docstrings.

Run from the repository root:
    uv run python scripts/gen_api_reference.py

Introspects the live `rusket` package and converts all public symbols'
NumPy-style docstrings into MkDocs-compatible Markdown, then writes the
result to docs/api-reference.md.  The file is always overwritten so it
stays canonical.  Hand-edits will be lost — put permanent notes in the
docstrings instead.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Section-level order for the generated document
# ---------------------------------------------------------------------------

# (heading, list-of-(name, obj)) — populated later after import
SECTIONS: list[tuple[str, list[tuple[str, Any]]]] = []

# ---------------------------------------------------------------------------
# DocString parsing helpers
# ---------------------------------------------------------------------------

_NUMPY_SECTION_RE = re.compile(r"^([A-Za-z][A-Za-z ]+)\n[-]+\s*$", re.MULTILINE)


def _split_numpy_docstring(doc: str) -> dict[str, str]:
    """Return a dict  {section_name: section_body} for a NumPy-style docstring."""
    if not doc:
        return {}
    # Normalise indentation
    lines = doc.expandtabs(4).splitlines()
    if lines:
        indent = len(lines[0]) - len(lines[0].lstrip())
        lines = [line[indent:] if line.startswith(" " * indent) else line for line in lines]
    doc = "\n".join(lines)

    parts = _NUMPY_SECTION_RE.split(doc)
    # parts[0] = summary, then pairs of (section_name, section_body)
    result: dict[str, str] = {}
    result["Summary"] = parts[0].strip()
    for i in range(1, len(parts) - 1, 2):
        name = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        result[name] = body
    return result


def _parse_parameters_section(text: str) -> list[tuple[str, str, str]]:
    """Parse a NumPy Parameters / Returns / Raises section.

    Returns a list of (name, type, description) tuples.
    """
    if not text:
        return []
    rows: list[tuple[str, str, str]] = []
    current_name = ""
    current_type = ""
    current_desc_lines: list[str] = []

    for line in text.splitlines():
        # A new parameter entry starts with something that is NOT indented and
        # matches `name : type` or just `name`
        if line and not line.startswith(" "):
            if current_name:
                rows.append((current_name, current_type, " ".join(current_desc_lines).strip()))
            if " : " in line:
                current_name, current_type = line.split(" : ", 1)
            else:
                current_name = line.strip()
                current_type = ""
            current_name = current_name.strip()
            current_type = current_type.strip()
            current_desc_lines = []
        else:
            current_desc_lines.append(line.strip())

    if current_name:
        rows.append((current_name, current_type, " ".join(current_desc_lines).strip()))
    return rows


def _escape_pipes(s: str) -> str:
    return s.replace("|", "\\|")


def _params_to_table(rows: list[tuple[str, str, str]], cols: tuple[str, ...] = ("Parameter", "Type", "Description")) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(_escape_pipes(str(v)) for v in row) + " |")
    return "\n".join(lines)


def _format_signature(obj: Any) -> str:
    """Return a clean `name(params)` string."""
    try:
        sig = inspect.signature(obj)
        # Remove 'self' / 'cls'
        params = {
            k: v
            for k, v in sig.parameters.items()
            if k not in ("self", "cls")
        }
        new_sig = sig.replace(parameters=list(params.values()))
        return str(new_sig)
    except (ValueError, TypeError):
        return "()"


def _format_symbol(name: str, obj: Any, level: int = 2) -> str:
    """Render a single function or class into markdown."""
    heading = "#" * level
    lines: list[str] = []

    # ---- Heading ---------------------------------------------------------
    lines.append(f"{heading} `{name}`")
    lines.append("")

    doc = inspect.getdoc(obj) or ""
    sections = _split_numpy_docstring(doc)
    summary = sections.get("Summary", "")

    is_class = inspect.isclass(obj)

    # ---- Summary ---------------------------------------------------------
    if summary:
        lines.append(summary)
        lines.append("")

    # ---- Signature -------------------------------------------------------
    if is_class:
        # Show __init__ signature
        init = getattr(obj, "__init__", None)
        if init:
            sig = _format_signature(init)
        else:
            sig = "()"
        # Get module path
        mod = getattr(obj, "__module__", "rusket")
        lines.append(f"```python")
        lines.append(f"from {mod} import {name}")
        lines.append(f"")
        lines.append(f"{name}{sig}")
        lines.append(f"```")
    else:
        mod = getattr(obj, "__module__", "rusket")
        sig = _format_signature(obj)
        lines.append(f"```python")
        lines.append(f"from {mod} import {name}")
        lines.append(f"")
        lines.append(f"{name}{sig}")
        lines.append(f"```")
    lines.append("")

    # ---- Parameters / Attributes -----------------------------------------
    for sec_name in ("Parameters", "Attributes"):
        if sec_name in sections:
            rows = _parse_parameters_section(sections[sec_name])
            if rows:
                lines.append(f"**{sec_name}**")
                lines.append("")
                lines.append(_params_to_table(rows))
                lines.append("")

    # ---- Returns ---------------------------------------------------------
    if "Returns" in sections:
        rows = _parse_parameters_section(sections["Returns"])
        lines.append("**Returns**")
        lines.append("")
        if rows:
            lines.append(_params_to_table(rows, cols=("Type", "Description") if len(rows[0]) == 2 else ("Name", "Type", "Description")))
        else:
            # Free-form return description
            lines.append(sections["Returns"])
        lines.append("")

    # ---- Raises ----------------------------------------------------------
    if "Raises" in sections:
        rows = _parse_parameters_section(sections["Raises"])
        if rows:
            lines.append("**Raises**")
            lines.append("")
            lines.append(_params_to_table(rows, cols=("Exception", "Condition")))
            lines.append("")

    # ---- Notes / Warnings -----------------------------------------------
    for sec_name in ("Notes", "Warning", "Warnings"):
        if sec_name in sections:
            lines.append(f"> **{sec_name}**")
            lines.append(f"> {sections[sec_name]}")
            lines.append("")

    # ---- Examples --------------------------------------------------------
    if "Examples" in sections:
        lines.append("**Examples**")
        lines.append("")
        # Wrap bare code blocks if not already wrapped
        ex = sections["Examples"]
        if "```" not in ex:
            lines.append("```python")
            lines.append(ex)
            lines.append("```")
        else:
            lines.append(ex)
        lines.append("")

    # ---- Public methods for classes --------------------------------------
    if is_class:
        methods = _collect_class_methods(obj)
        if methods:
            for mname, mobj in methods:
                lines.append(_format_symbol(f"{name}.{mname}", mobj, level=level + 1))

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _collect_class_methods(cls: type) -> list[tuple[str, Any]]:
    """Return public non-dunder methods defined directly on the class (not inherited from base)."""
    skip_bases = {"object", "BaseModel", "Miner", "ImplicitRecommender", "RuleMinerMixin", "ABC"}
    own_methods: list[tuple[str, Any]] = []
    for name, val in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        # Only include methods that are explicitly defined on this class (not just inherited)
        if name not in cls.__dict__:
            continue
        own_methods.append((name, val))
    return own_methods


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def build_api_reference() -> str:
    """Import rusket and build the full API reference markdown string."""
    import rusket
    import rusket.spark as spark_mod
    import rusket.streaming as streaming_mod
    import rusket.analytics as analytics_mod
    import rusket.similarity as similarity_mod
    import rusket.export as export_mod
    import rusket.transactions as transactions_mod
    import rusket.recommend as recommend_mod
    import rusket.viz as viz_mod
    from rusket.model import RuleMinerMixin

    output_lines: list[str] = []

    output_lines.append("# API Reference")
    output_lines.append("")
    output_lines.append(
        "> This file is **auto-generated** by `scripts/gen_api_reference.py`.  "
        "Do not edit by hand — update the Python docstrings instead."
    )
    output_lines.append("")

    # ------------------------------------------------------------------
    # 1. Functional API (top-level convenience functions)
    # ------------------------------------------------------------------
    output_lines.append("## Functional API")
    output_lines.append("")
    output_lines.append("Convenience module-level functions.  For most use-cases these are the only entry points you need.")
    output_lines.append("")

    func_symbols: list[tuple[str, Any]] = [
        ("mine", rusket.mine),
        ("fpgrowth", rusket.fpgrowth),
        ("eclat", rusket.eclat),
        ("association_rules", rusket.association_rules),
        ("prefixspan", rusket.prefixspan),
        ("hupm", rusket.hupm),
        ("sequences_from_event_log", rusket.sequences_from_event_log),
        ("mine_hupm", rusket.mine_hupm),
        ("mine_duckdb", rusket.mine_duckdb),
        ("mine_spark", rusket.mine_spark),
        ("from_transactions", rusket.from_transactions),
        ("from_transactions_csr", rusket.from_transactions_csr),
        ("from_pandas", rusket.from_pandas),
        ("from_polars", rusket.from_polars),
        ("from_spark", rusket.from_spark),
    ]
    for sym_name, obj in func_symbols:
        output_lines.append(_format_symbol(sym_name, obj, level=3))

    # ------------------------------------------------------------------
    # 2. OOP Mining API
    # ------------------------------------------------------------------
    output_lines.append("## OOP Mining API")
    output_lines.append("")
    output_lines.append(
        "All mining classes share a common `Miner.from_transactions()` / `.mine()` interface. "
        "`FPGrowth`, `Eclat`, `AutoMiner`, and `HUPM` also inherit `RuleMinerMixin` which adds "
        "`.association_rules()` and `.recommend_items()` helpers."
    )
    output_lines.append("")

    oop_mining_symbols: list[tuple[str, Any]] = [
        ("FPGrowth", rusket.FPGrowth),
        ("Eclat", rusket.Eclat),
        ("AutoMiner", rusket.AutoMiner),
        ("PrefixSpan", rusket.PrefixSpan),
        ("HUPM", rusket.HUPM),
        ("FPMiner", rusket.FPMiner),
    ]
    for sym_name, obj in oop_mining_symbols:
        output_lines.append(_format_symbol(sym_name, obj, level=3))

    # ------------------------------------------------------------------
    # 2b. RuleMinerMixin — shared mixin inherited by all item-set miners
    # ------------------------------------------------------------------
    output_lines.append("## `RuleMinerMixin` — Shared Miner Interface")
    output_lines.append("")
    output_lines.append(
        "`FPGrowth`, `Eclat`, `AutoMiner`, and `HUPM` all inherit these methods from "
        "`RuleMinerMixin`.  You do not construct `RuleMinerMixin` directly."
    )
    output_lines.append("")

    mixin_methods: list[tuple[str, Any]] = [
        ("RuleMinerMixin.association_rules", RuleMinerMixin.association_rules),
        ("RuleMinerMixin.recommend_items", RuleMinerMixin.recommend_items),
        ("RuleMinerMixin._invalidate_rules_cache", RuleMinerMixin._invalidate_rules_cache),
    ]
    for sym_name, obj in mixin_methods:
        output_lines.append(_format_symbol(sym_name, obj, level=3))

    # ------------------------------------------------------------------
    # 3. Recommender / Collaborative Filtering
    # ------------------------------------------------------------------
    output_lines.append("## Recommenders")
    output_lines.append("")

    rec_symbols: list[tuple[str, Any]] = [
        ("ALS", rusket.ALS),
        ("BPR", rusket.BPR),
        ("Recommender", rusket.Recommender),
        ("NextBestAction", rusket.NextBestAction),
    ]
    for sym_name, obj in rec_symbols:
        output_lines.append(_format_symbol(sym_name, obj, level=3))

    # ------------------------------------------------------------------
    # 4. Analytics & Utilities
    # ------------------------------------------------------------------
    output_lines.append("## Analytics & Utilities")
    output_lines.append("")

    util_symbols: list[tuple[str, Any]] = [
        ("score_potential", rusket.score_potential),
        ("similar_items", rusket.similar_items),
        ("find_substitutes", rusket.find_substitutes),
        ("customer_saturation", rusket.customer_saturation),
        ("export_item_factors", rusket.export_item_factors),
    ]
    for sym_name, obj in util_symbols:
        output_lines.append(_format_symbol(sym_name, obj, level=3))

    # ------------------------------------------------------------------
    # 4b. Visualization (rusket.viz)
    # ------------------------------------------------------------------
    output_lines.append("## Visualization (`rusket.viz`)")
    output_lines.append("")
    output_lines.append(
        "Graph and visualization utilities.  Requires `networkx` (`pip install networkx`)."
    )
    output_lines.append("")

    viz_symbols: list[tuple[str, Any]] = [
        ("to_networkx", viz_mod.to_networkx),
    ]
    for sym_name, obj in viz_symbols:
        output_lines.append(_format_symbol(f"rusket.viz.{sym_name}", obj, level=3))

    # ------------------------------------------------------------------
    # 5. Distributed Spark API
    # ------------------------------------------------------------------
    output_lines.append("## Distributed Spark API (`rusket.spark`)")
    output_lines.append("")
    output_lines.append(
        "All functions in `rusket.spark` distribute computation across PySpark partitions "
        "using Apache Arrow (zero-copy) for maximum throughput."
    )
    output_lines.append("")

    spark_symbols: list[tuple[str, Any]] = [
        ("mine_grouped", spark_mod.mine_grouped),
        ("rules_grouped", spark_mod.rules_grouped),
        ("prefixspan_grouped", spark_mod.prefixspan_grouped),
        ("hupm_grouped", spark_mod.hupm_grouped),
        ("recommend_batches", spark_mod.recommend_batches),
        ("to_spark", spark_mod.to_spark),
    ]
    for sym_name, obj in spark_symbols:
        output_lines.append(_format_symbol(f"rusket.spark.{sym_name}", obj, level=3))

    return "\n".join(output_lines) + "\n"


def main() -> None:
    target = ROOT / "docs" / "api-reference.md"
    print("Generating API reference …")
    try:
        content = build_api_reference()
    except Exception as e:
        print(f"ERROR: Failed to generate API reference: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    target.write_text(content, encoding="utf-8")
    print(f"✔ Wrote {target.relative_to(ROOT)}  ({len(content):,} chars)")


if __name__ == "__main__":
    main()
