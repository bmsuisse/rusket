"""Backward-compatibility shim — canonical location: ``rusket.recommenders.content_based``."""

import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.recommenders.content_based")
_sys.modules[__name__] = _real
