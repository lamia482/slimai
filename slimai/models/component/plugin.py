import sys
import os.path as osp
import importlib
import torch
from slimai.helper.help_build import MODELS


__all__ = [
  "Plugin", 
]

@MODELS.register_module()
class Plugin(torch.nn.Module):
  """
  A plugin module that loads a module from a path and method name.
  path: str, e.g. 
  1. "/root/dir/file.py:create"
  2. root.dir.file:create
  3. torchvision.models.resnet50

  Args:
    module: str, the module to load.
    layer_replacement: dict, the attribute of the module to replace
                       e.g. {"forward": "forward_new"} will replace the forward method
                       to Plugin(module=forward_new), this is useful when it's necessary to 
                       replace/drop some layers.
    **kwargs: the kwargs to pass to initialize the module.
  """
  def __init__(self, *, 
               module, 
               layer_replacement={}, 
               **kwargs):
    super().__init__()
    module_name, *method_name = module.rsplit(":", 1)
    if len(method_name) == 0: # torch.nn.Module
      module = MODELS.get(module)
      method_name = None
    elif module_name.endswith(".py"): # file.py:forward
      module_dir = osp.dirname(module_name)
      module_file = osp.basename(module_name)
      module_name = module_file[:-3]
      sys.path.insert(0, module_dir)
      module = importlib.import_module(module_name)
      sys.path.pop(0)
    else: # file:forward
      # Load from installed package
      module = importlib.import_module(module_name)
    
    if method_name is None:
      layer = module(**kwargs) # type: ignore
    else:
      assert (
        len(method_name) == 1
      ), f"Invalid method name: {method_name}"
      method_name = method_name[0]
      layer = getattr(module, method_name)(**kwargs)

    for name, replacement in layer_replacement.items():
      if L := getattr(layer, name, None):
        print(f"[Plugin] -> Replacing ({module_name}.{name}) from '{L.__class__.__name__}' to '{replacement}'")
        setattr(layer, name, Plugin(module=replacement))

    self.layer = layer
    return
  
  def forward(self, x):
    return self.layer(x)
