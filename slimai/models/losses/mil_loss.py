import torch
from typing import Dict
from slimai.helper.help_build import MODELS, build_loss


__all__ = [
  "MILLoss", 
]

@MODELS.register_module()
class MILLoss(torch.nn.Module):
  def __init__(self, 
               cls_loss=dict(
                  type="torch.nn.CrossEntropyLoss",
                  label_smoothing=0.1,
               )):
    super().__init__()
    # Initialize classification loss
    self.cls_loss = build_loss(cls_loss)
    return
  
  def forward(self, 
              logits: torch.Tensor, 
              targets: torch.Tensor) -> Dict[str, torch.Tensor]:
    # Compute classification loss
    cls_loss = self.cls_loss(logits, targets)
    return dict(cls_loss=cls_loss)
