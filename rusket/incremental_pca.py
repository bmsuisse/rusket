"""Backward-compatibility shim — canonical location: ``rusket.viz.incremental_pca``."""

import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.viz.incremental_pca")
_sys.modules[__name__] = _real
