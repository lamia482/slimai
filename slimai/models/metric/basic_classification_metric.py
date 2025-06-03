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

  @torch.inference_mode()
  def forward(self, 
              output: Dict[str, torch.Tensor], 
              targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    softmax = output["softmax"]
    labels = targets["label"]
    return dict(
      acc=self.acc(softmax, labels),
      kappa=self.kappa(softmax, labels),
    )
