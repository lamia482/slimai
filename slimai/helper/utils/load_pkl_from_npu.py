import io
import pickle
from types import ModuleType
from typing import Any, Dict, Optional

import torch


_DEFAULT_MODULE_REMAP: Dict[str, str] = {
  # NumPy 2.x internal layout vs NumPy 1.x
  "numpy._core": "numpy.core",
}


class _RemappingUnpickler(pickle.Unpickler):
  def __init__(self, file, module_remap: Optional[Dict[str, str]] = None, **kwargs):
    super().__init__(file, **kwargs)
    self._module_remap = dict(_DEFAULT_MODULE_REMAP)
    if module_remap:
      self._module_remap.update(module_remap)

  def find_class(self, module: str, name: str):
    # Some pickles reference submodules (e.g. numpy._core.something). Remap both
    # exact module names and prefix matches.
    if module in self._module_remap:
      module = self._module_remap[module]
    else:
      for old, new in self._module_remap.items():
        if module.startswith(old + "."):
          module = new + module[len(old) :]
          break
    return super().find_class(module, name)


def _make_pickle_module(module_remap: Optional[Dict[str, str]] = None):
  """
  torch.load expects a 'pickle_module' with an Unpickler class.
  We provide a tiny shim so we can inject our module remapping.
  """

  class Unpickler(_RemappingUnpickler):
    def __init__(self, file, **kwargs):
      super().__init__(file, module_remap=module_remap, **kwargs)

  m = ModuleType("pickle_compat")
  m.Unpickler = Unpickler
  return m


def load_torch_pickle_compat(
  path: str,
  map_location: str = "cpu",
  *,
  weights_only: bool = False,
  module_remap: Optional[Dict[str, str]] = None,
  **pickle_load_args: Any,
):
  """
  Load a torch-saved pickle while remapping module paths for compatibility.

  WARNING: Like torch.load(weights_only=False), this unpickles arbitrary objects.
  Only use on trusted files.
  """
  pickle_module = _make_pickle_module(module_remap=module_remap)
  return torch.load(
    path,
    map_location=map_location,
    weights_only=weights_only,
    pickle_module=pickle_module,
    **pickle_load_args,
  )
