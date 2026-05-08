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
  "Checkpoint",
  "DataSample",
  "Distributed",
  "Gradient",
  "Record",
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

  object_map = {
    "Checkpoint": ("checkpoint", "Checkpoint"),
    "DataSample": ("structure", "DataSample"),
    "Distributed": ("distributed", "Distributed"),
    "Gradient": ("gradient", "Gradient"),
    "Record": ("record", "Record"),
  }
  if name in object_map:
    module_name, object_name = object_map[name]
    module = importlib.import_module(f".{module_name}", __name__)
    obj = getattr(module, object_name)
    globals()[name] = obj
    return obj

  raise AttributeError(f"module '{__name__}' has no attribute '{name}'")