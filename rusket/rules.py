"""Backward-compatibility shim — canonical location: ``rusket.recommenders.rules``."""

import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.recommenders.rules")
_sys.modules[__name__] = _real
