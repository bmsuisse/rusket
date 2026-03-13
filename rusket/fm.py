"""Backward-compatibility shim — canonical location: ``rusket.recommenders.fm``."""

import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.recommenders.fm")
_sys.modules[__name__] = _real
