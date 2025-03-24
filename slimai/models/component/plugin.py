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
  path: str, e.g. "/root/dir/file.py:create"
  """
  def __init__(self, *, 
               module, 
               **kwargs):
    super().__init__()
    module_name, method_name = module.rsplit(":", 1)
    if module_name.endswith(".py"):      
      module_dir = osp.dirname(module_name)
      module_file = osp.basename(module_name)
      module_name = module_file[:-3]
      sys.path.insert(0, module_dir)
      module = importlib.import_module(module_name)
      sys.path.pop(0)
    else:
      # Load from installed package
      module = importlib.import_module(module_name)
    
    self.layer = getattr(module, method_name)(**kwargs)
    return
  
  def forward(self, x):
    return self.layer(x)
