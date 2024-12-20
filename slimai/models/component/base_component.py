import torch
from abc import abstractmethod


class BaseComponent(torch.nn.Module):
  def __init__(self, pretrained_weight=None):
    super().__init__()
    self._init_weights(pretrained_weight=pretrained_weight)
    return

  @abstractmethod
  def _init_weights(self, pretrained_weight=None):
    if pretrained_weight is not None:
      self.load_state_dict(torch.load(pretrained_weight, map_location="cpu"), strict=False)
    return

  @abstractmethod
  def forward(self, *args, **kwargs):
    raise NotImplementedError
  