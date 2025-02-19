from typing import Dict
import torch
from slimai.helper.help_build import MODELS, build_metric


@MODELS.register_module()
class DINOMetric(torch.nn.Module):
  def __init__(self, ):
    super().__init__()
    return

  @torch.no_grad()
  def forward(self, 
              logits: torch.Tensor, 
              targets: torch.Tensor) -> Dict[str, torch.Tensor]:
    return
