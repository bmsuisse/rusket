"""Model hierarchy for rusket — re-exports all public classes for backward compatibility.

This package splits the original monolithic ``model.py`` into focused submodules:

- ``_mixins.py``: :class:`RuleMinerMixin`
- ``_base.py``:   :class:`BaseModel`, :func:`load_model`
- ``_miner.py``:  :class:`Miner`
- ``_recommender.py``: :class:`ImplicitRecommender`, :class:`SequentialRecommender`

All names are re-exported here so existing ``from rusket.model import …`` imports
continue to work without modification.
"""

from ._base import BaseModel, load_model
from ._miner import Miner
from ._mixins import RuleMinerMixin
from ._recommender import ImplicitRecommender, SequentialRecommender

__all__ = [
    "BaseModel",
    "ImplicitRecommender",
    "Miner",
    "RuleMinerMixin",
    "SequentialRecommender",
    "load_model",
]
