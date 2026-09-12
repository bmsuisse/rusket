"""Regression tests for the legacy module-alias shims in rusket/_shims.py.

Historically each old top-level module (``rusket.spark``, ``rusket.grouped``, …)
was its own file doing a plain ``sys.modules[__name__] = importlib.import_module(...)``
re-export. That let ordinary Python import machinery bind the submodule onto its
parent package. The consolidated ``install()`` in ``rusket/_shims.py`` registers
modules directly in ``sys.modules`` instead, which bypasses that binding step
unless done explicitly.
"""

import rusket


def test_legacy_top_level_alias_attribute_access():
    """rusket.<old_name> must resolve via plain attribute access, not just import."""
    assert rusket.grouped is rusket.integrations.grouped
    assert rusket.spark is rusket.integrations.spark
    assert rusket.cuda is rusket.integrations.cuda
    assert rusket.gpu is rusket.integrations.gpu


def test_canonical_submodule_attribute_access_after_import():
    """import rusket.integrations.grouped must also bind `grouped` onto `rusket.integrations`.

    This is the exact regression from the bug report: ordinary Python import
    followed by plain attribute access used to raise
    ``AttributeError: module 'rusket.integrations' has no attribute 'grouped'``.
    """
    import rusket.integrations.grouped  # noqa: F401

    assert hasattr(rusket.integrations, "grouped")
    assert rusket.integrations.grouped is rusket.grouped


def test_canonical_submodule_attribute_access_export_package():
    import rusket.export.faiss_ann  # noqa: F401
    import rusket.export.vector_export  # noqa: F401

    assert hasattr(rusket.export, "faiss_ann")
    assert hasattr(rusket.export, "vector_export")


def test_lazy_submodules_do_not_eagerly_import_heavy_deps():
    """Binding rusket.integrations.grouped onto its parent must not import pyspark."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, rusket, rusket.integrations.grouped; print('pyspark' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"
