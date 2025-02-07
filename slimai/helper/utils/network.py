from typing import Dict


class PytorchNetworkUtils(object):

  @classmethod
  def get_module(cls, module):
    if hasattr(module, "module"):
      module = module.module
    return module

  @classmethod
  def get_module_params(cls, module, grad_mode="all"):
    """Get parameters from PyTorch model, handling both DDP and non-DDP cases.
    
    Args:
      module: PyTorch model (nn.Module or DDP wrapped module)
      
    Returns:
      Parameters of the underlying model
    """
    assert (
      grad_mode in ["all", "trainable", "frozen", "unused"]
    ), "grad_mode must be one of ['all', 'trainable', 'frozen', 'unused']"

    if grad_mode == "all":
      condition = lambda x: True
    elif grad_mode == "trainable":
      condition = lambda x: x.requires_grad
    elif grad_mode == "frozen":
      condition = lambda x: not x.requires_grad
    elif grad_mode == "unused":
      condition = lambda x: x.grad is None

    params = []
    for _, param in module.named_parameters():
      if condition(param):
        params.append(param)
    return params
    
  @classmethod
  def convert_magnitude(cls, num, magnitude="auto"):
    assert magnitude in ["auto", "B", "M", "K", "digit", None], "Invalid magnitude"
    if magnitude == "auto":
      if num >= 1e9:
        return f"{num / 1e9:.2f}B"
      elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
      elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
      else:
        return f"{num}"
    elif magnitude == "B":
      return f"{num / 1e9:.2f}B"
    elif magnitude == "M":
      return f"{num / 1e6:.2f}M"
    elif magnitude == "K":
      return f"{num / 1e3:.2f}K"
    elif magnitude == "digit":
      return num
    else:
      return f"{num}"
  
  @classmethod
  def get_params_size(cls, module, grad_mode="all", magnitude="auto"):
    params = cls.get_module_params(module, grad_mode=grad_mode)
    num = sum(p.numel() for p in params)
    return cls.convert_magnitude(num, magnitude)
  
  @classmethod
  def freeze(cls, module):
    for param in cls.get_module_params(module):
      param.requires_grad = False
    return module

  @classmethod
  def unfreeze(cls, module):
    for param in cls.get_module_params(module):
      param.requires_grad = True
    return module
  