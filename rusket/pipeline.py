"""Backward-compatibility shim — canonical location: ``rusket.evaluation.pipeline``."""

import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.evaluation.pipeline")
_sys.modules[__name__] = _real
