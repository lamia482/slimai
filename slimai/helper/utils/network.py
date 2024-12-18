

class PytorchNetworkUtils(object):
  
  @classmethod
  def get_params_size(cls, module):
    return sum(p.numel() for p in module.parameters())
  
  @classmethod
  def freeze(cls, module):
    for param in module.parameters():
      param.requires_grad = False
    return module

  @classmethod
  def unfreeze(cls, module):
    for param in module.parameters():
      param.requires_grad = True
    return module
