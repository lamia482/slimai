

class PytorchNetworkUtils(object):

  @classmethod
  def get_module_params(cls, module):
    """Get parameters from PyTorch model, handling both DDP and non-DDP cases.
    
    Args:
      module: PyTorch model (nn.Module or DDP wrapped module)
      
    Returns:
      Parameters of the underlying model
    """
    if hasattr(module, "module"):
      # Handle DistributedDataParallel case
      return module.module.parameters()
    # Handle regular nn.Module case
    return module.parameters()
    
  @classmethod
  def convert_magnitude(cls, num, magnitude="auto"):
    assert magnitude in ["auto", "B", "M", "K", None], "Invalid magnitude"
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
    else:
      return f"{num}"
  
  @classmethod
  def get_params_size(cls, module, grad_mode="all", magnitude="auto"):
    assert (
      grad_mode in ["all", "trainable", "frozen"]
    ), "grad_mode must be one of ['all', 'trainable', 'frozen']"

    if grad_mode == "all":
      condition = lambda x: True
    elif grad_mode == "trainable":
      condition = lambda x: x.requires_grad
    elif grad_mode == "frozen":
      condition = lambda x: not x.requires_grad

    num = sum(p.numel() for p in cls.get_module_params(module) if condition(p))
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
  