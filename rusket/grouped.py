"""Backward-compatibility shim — canonical location: ``rusket.integrations.grouped``."""

import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.integrations.grouped")
_sys.modules[__name__] = _real
