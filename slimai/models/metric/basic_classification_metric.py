from typing import Dict
import torch
from slimai.helper.help_build import MODELS, build_metric
from slimai.helper.structure import DataSample


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
              embedding_dict: Dict[str, torch.Tensor], 
              batch_info: DataSample) -> Dict[str, torch.Tensor]:
    logits = embedding_dict["head"]
    scores = torch.softmax(logits, dim=1)
    labels = batch_info.label
    return dict(
      acc=self.acc(scores, labels),
      kappa=self.kappa(scores, labels),
    )
