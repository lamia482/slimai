import torch
from typing import Dict
from slimai.helper.help_build import MODELS, build_loss


@MODELS.register_module()
class BasicClassificationLoss(torch.nn.Module):
  def __init__(self, 
               cls_loss=dict(
                  type="torch.nn.CrossEntropyLoss",
                  label_smoothing=0.1,
               ), **kwargs):
    super().__init__()
    # Initialize classification loss
    self.cls_loss = build_loss(cls_loss)
    self.extra_loss = {}
    for k, v in kwargs.items():
      self.extra_loss[k] = build_loss(v)
    return
  
  def forward(self, 
              logits: torch.Tensor, 
              targets: torch.Tensor) -> Dict[str, torch.Tensor]:
    # Compute classification loss
    cls_loss = self.cls_loss(logits, targets)
    result = dict(cls_loss=cls_loss)
    for k, v in self.extra_loss.items():
      result[k] = v(logits, targets)
    return result
