"""Consolidated backward-compatibility import table.

Replaces the many one-file-per-module shims that used to live directly
under ``rusket/`` (each doing ``sys.modules[__name__] = importlib.import_module(...)``).
This module holds the same old-name -> canonical-module-path mapping in a
single table and installs it into :data:`sys.modules` once, from
``rusket/__init__.py``, so that e.g. ``import rusket.als`` keeps resolving
to ``rusket.recommenders.als`` with identical behavior.
"""

import importlib.util as _importlib_util
import sys as _sys

#: old top-level ``rusket.<name>`` -> canonical module path.
LEGACY_MODULE_MAP: dict[str, str] = {
    "_config": "rusket._internal._config",
    "_compat": "rusket._internal._compat",
    "_dependencies": "rusket._internal._dependencies",
    "_core": "rusket._internal._core",
    "_validation": "rusket._internal._validation",
    "_embedding_mixin": "rusket._internal._embedding_mixin",
    "_type_utils": "rusket._internal._type_utils",
    "als": "rusket.recommenders.als",
    "bert4rec": "rusket.sequential.bert4rec",
    "association_rules": "rusket.miners.association_rules",
    "analytics": "rusket._internal.analytics",
    "ann": "rusket.export.ann",
    "bpr": "rusket.recommenders.bpr",
    "content_based": "rusket.recommenders.content_based",
    "ease": "rusket.recommenders.ease",
    "cuda": "rusket.integrations.cuda",
    "faiss_ann": "rusket.export.faiss_ann",
    "fin": "rusket.miners.fin",
    "fpmc": "rusket.sequential.fpmc",
    "fm": "rusket.recommenders.fm",
    "gpu": "rusket.integrations.gpu",
    "grouped": "rusket.integrations.grouped",
    "hybrid_embedding": "rusket.export.hybrid_embedding",
    "hybrid": "rusket.recommenders.hybrid",
    "item_knn": "rusket.recommenders.item_knn",
    "incremental_pca": "rusket.viz.incremental_pca",
    "lightgcn": "rusket.recommenders.lightgcn",
    "model_selection": "rusket.evaluation.model_selection",
    "lcm": "rusket.miners.lcm",
    "negfin": "rusket.miners.negfin",
    "optuna": "rusket.evaluation.optuna",
    "pipeline": "rusket.evaluation.pipeline",
    "nmf": "rusket.recommenders.nmf",
    "popularity": "rusket.recommenders.popularity",
    "rules": "rusket.recommenders.rules",
    "sasrec": "rusket.sequential.sasrec",
    "recommend": "rusket.recommenders.recommend",
    "similarity": "rusket._internal.similarity",
    "splitting": "rusket.evaluation.splitting",
    "svd": "rusket.recommenders.svd",
    "spark": "rusket.integrations.spark",
    "streaming": "rusket.miners.streaming",
    "typing": "rusket._internal.typing",
    "vector_export": "rusket.export.vector_export",
    "transactions": "rusket.miners.transactions",
    "user_knn": "rusket.recommenders.user_knn",
}


def install() -> None:
    """Register every legacy ``rusket.<old>`` name as an alias in ``sys.modules``.

    Also sets each old name as an attribute of the ``rusket`` package so
    that ``rusket.<old>`` resolves even without an explicit
    ``import rusket.<old>`` statement, matching the behavior of the
    individual shim files this replaces.

    A few old module basenames (e.g. ``association_rules``) collide with a
    same-named public function/class already exported from
    ``rusket/__init__.py``. In that case the existing package attribute
    (the function/class) takes precedence and is left untouched — only
    ``sys.modules['rusket.<old_name>']`` is registered, so
    ``import rusket.association_rules`` still resolves to the submodule
    while ``rusket.association_rules(...)`` keeps calling the function,
    exactly as before this consolidation.

    Most canonical targets (``rusket.recommenders.als`` and friends) are
    already imported eagerly by ``rusket/__init__.py`` itself, so aliasing
    them here is free. A handful (``integrations.spark``, ``integrations.cuda``,
    ``export.faiss_ann``, ``export.vector_export``, ``integrations.grouped``, …)
    are NOT imported by ``__init__.py`` on purpose, since they pull in heavy
    optional deps (pyspark, cupy/torch, faiss). For those we register an
    ``importlib.util.LazyLoader`` instead of importing them for real, so
    ``rusket.spark`` / ``import rusket.spark`` still resolve but the actual
    module body (and its heavy imports) only runs on first attribute access.
    """
    rusket_pkg = _sys.modules.get("rusket")
    for old_name, canonical in LEGACY_MODULE_MAP.items():
        full_name = f"rusket.{old_name}"
        if full_name in _sys.modules:
            continue

        already_loaded = _sys.modules.get(canonical)
        if already_loaded is not None:
            # Cheap case: __init__.py (or something it imported) already
            # loaded the canonical module — just alias it, no extra work.
            _sys.modules[full_name] = already_loaded
            if rusket_pkg is not None and not hasattr(rusket_pkg, old_name):
                setattr(rusket_pkg, old_name, already_loaded)
            continue

        # Not yet imported — defer via LazyLoader so `rusket.<old_name>` and
        # `import rusket.<old_name>` resolve, but the real (possibly heavy)
        # module body only executes on first attribute access.
        spec = _importlib_util.find_spec(canonical)
        if spec is None or spec.loader is None:
            continue
        spec.loader = _importlib_util.LazyLoader(spec.loader)
        module = _importlib_util.module_from_spec(spec)
        _sys.modules[full_name] = module
        _sys.modules.setdefault(canonical, module)
        spec.loader.exec_module(module)  # installs the lazy proxy, doesn't run the module yet

        # Registering modules directly in sys.modules (above) means Python's
        # import system finds them there and skips its usual "bind submodule
        # onto parent package" step. Without this, `rusket.<old_name>` (the
        # legacy top-level alias) AND `rusket.integrations.grouped`-style
        # attribute access on the real canonical parent package (as opposed to
        # `import rusket.integrations.grouped`, which works either way) would
        # raise AttributeError. Bind both ourselves, matching what a normal
        # import statement would have done.
        if rusket_pkg is not None:
            setattr(rusket_pkg, old_name, module)

        canonical_parent_name, _, canonical_base = canonical.rpartition(".")
        if canonical_parent_name:
            canonical_parent = _sys.modules.get(canonical_parent_name)
            if canonical_parent is None:
                # The intermediate package (e.g. ``rusket.integrations``) hasn't
                # been imported yet. These are thin, docstring-only ``__init__.py``
                # files — importing them for real is cheap and does not pull in
                # any heavy optional deps (unlike the leaf module itself, which
                # stays lazy via the LazyLoader above).
                import importlib as _importlib

                canonical_parent = _importlib.import_module(canonical_parent_name)
            setattr(canonical_parent, canonical_base, module)
