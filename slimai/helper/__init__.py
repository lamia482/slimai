"""
slimai.helper package entry.

Avoid eager importing optional dependencies during package bootstrap.
"""

import importlib
from typing import Any

__all__ = [
  "common",
  "help_build",
  "help_utils",
  "utils",
  "visuals",
  "checkpoint",
  "distributed",
  "gradient",
  "record",
  "structure",
]


def __getattr__(name: str) -> Any:
  module_map = {
    "common": "common",
    "help_build": "help_build",
    "help_utils": "help_utils",
    "utils": "utils",
    "visuals": "visuals",
    "checkpoint": "checkpoint",
    "distributed": "distributed",
    "gradient": "gradient",
    "record": "record",
    "structure": "structure",
  }
  if name in module_map:
    module = importlib.import_module(f".{module_map[name]}", __name__)
    globals()[name] = module
    return module
  raise AttributeError(f"module '{__name__}' has no attribute '{name}'")