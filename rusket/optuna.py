"""Backward-compatibility shim — canonical location: ``rusket.evaluation.optuna``."""

import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.evaluation.optuna")
_sys.modules[__name__] = _real
