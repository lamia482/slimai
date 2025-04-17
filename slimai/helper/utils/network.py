import torch


class PytorchNetworkUtils(object):
  """A collection of utility functions for PyTorch networks."""
  
  @classmethod
  def desc(cls, module):
    all_param_size = cls.get_params_size(module, grad_mode="all")
    trainable_param_size = cls.get_params_size(module, grad_mode="trainable")
    all_param_num = cls.get_params_size(module, grad_mode="all", magnitude="digit")
    trainable_param_num = cls.get_params_size(module, grad_mode="trainable", magnitude="digit")
    return (f"Total {all_param_size} parameters, in which "
            f"{trainable_param_size} parameters are trainable({100*trainable_param_num/all_param_num:.2f}%)")

  @classmethod
  def get_module(cls, module):
    """Get module from module dict or module."""
    def _get_module(_module):
      if hasattr(_module, "module"):
        _module = _module.module
      return _module
    if isinstance(module, torch.nn.ModuleDict):
      return torch.nn.ModuleDict({k: _get_module(v) for k, v in module.items()})
    return _get_module(module)

  @classmethod
  def fix_weight(cls, weight, *, to_ddp, is_module_dict, ddp_prefix="module."):
    """fix weight name to be compatible with ddp"""
    is_already_ddp = ddp_prefix in list(weight.keys())[0]
    if to_ddp and not is_already_ddp:
      def update_key(k):
        names = k.split(".")
        start_idx = 1 if is_module_dict else 0
        return ".".join(names[:start_idx] + [ddp_prefix.replace(".", "")] + names[start_idx:])
      weight = {update_key(k): v for k, v in weight.items()}
    elif not to_ddp and is_already_ddp:
      weight = {k.replace(ddp_prefix, ""): v for k, v in weight.items()}
    return weight

  @classmethod
  def clip_gradients(cls, model, clip=None):
    norms = []
    for name, p in model.named_parameters():
      if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        norms.append(param_norm.item())
        clip_coef = 1 if clip is None else (clip / (param_norm + 1e-6))
        if clip_coef < 1:
          p.grad.data.mul_(clip_coef)
    return norms
  
  @classmethod
  def cancel_gradient(cls, module, current_epoch, first_n_epoch_freeze_last_layer):
    if current_epoch >= first_n_epoch_freeze_last_layer:
      return
    for param in cls.get_module_params(module, 
                                       grad_mode=lambda name, _: "last_layer" in name):
      param.grad = None
    return module

  @classmethod
  def get_module_params(cls, module, grad_mode="all"):
    assert (
      grad_mode in ["all", "trainable", "frozen", "unused", "grad"] or isinstance(grad_mode, callable)
    ), "grad_mode must be one of ['all', 'trainable', 'frozen', 'unused', 'grad'] or a callable, but got {}".format(grad_mode)

    if grad_mode == "all":
      condition = lambda _, x: True
    elif grad_mode == "trainable":
      condition = lambda _, x: x.requires_grad
    elif grad_mode == "frozen":
      condition = lambda _, x: not x.requires_grad
    elif grad_mode == "unused":
      condition = lambda _, x: x.grad is None
    elif grad_mode == "grad":
      condition = lambda _, x: x.grad is not None
    elif isinstance(grad_mode, callable):
      condition = grad_mode

    params = []
    for name, param in module.named_parameters():
      if condition(name, param):
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
  
  @classmethod
  def init_weights(cls, m):
    if isinstance(m, torch.nn.Linear):
      # Xavier/Glorot initialization for linear layers
      torch.nn.init.xavier_uniform_(m.weight)
      if m.bias is not None:
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Conv2d):
      # He/Kaiming initialization for conv layers with ReLU
      torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      if m.bias is not None:
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
      # Initialize BN and GN with ones for weights and zeros for bias
      torch.nn.init.constant_(m.weight, 1.0)
      torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, torch.nn.LayerNorm):
      # Initialize LN with ones for weights and zeros for bias
      torch.nn.init.constant_(m.weight, 1.0)
      torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, torch.nn.Embedding):
      # Truncated normal initialization for embeddings
      torch.nn.init.trunc_normal_(m.weight, std=0.02, a=-2, b=2)
    elif isinstance(m, torch.nn.Parameter):
      # Truncated normal initialization for parameters
      torch.nn.init.trunc_normal_(m.data, std=0.02, a=-2, b=2)
    elif isinstance(m, (torch.nn.ParameterList, torch.nn.ParameterDict)):
      # Recursively initialize parameters in containers
      for param in m.parameters():
        if param is not None:
          cls.init_weights(param)
    return