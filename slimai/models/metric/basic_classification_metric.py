from typing import Dict
import torch
from slimai.helper.help_build import MODELS, build_metric


@MODELS.register_module()
class BasicClassificationMetric(torch.nn.Module):
  def __init__(self, 
               acc=dict(
                  type="torchmetrics.Accuracy",
                  task="multiclass",
                ), 
               kappa=dict(
                  type="torchmetrics.CohenKappa",
                  task="multiclass",
                )
              ):
    super().__init__()
    self.acc = build_metric(acc)
    self.kappa = build_metric(kappa)

  @torch.no_grad()
  def forward(self, 
              logits: torch.Tensor, 
              targets: torch.Tensor) -> Dict[str, torch.Tensor]:
    scores = torch.softmax(logits, dim=1)
    return dict(
      acc=self.acc(scores, targets),
      kappa=self.kappa(scores, targets),
    )
