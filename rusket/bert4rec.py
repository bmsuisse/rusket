"""Backward-compatibility shim — canonical location: ``rusket.sequential.bert4rec``."""

import importlib as _importlib
import sys as _sys

_real = _importlib.import_module("rusket.sequential.bert4rec")
_sys.modules[__name__] = _real
