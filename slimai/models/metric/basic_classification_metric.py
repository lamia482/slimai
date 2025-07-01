from typing import Dict
import torch
from slimai.helper.help_build import MODELS, build_metric


@MODELS.register_module()
class BasicClassificationMetric(torch.nn.Module):
  def __init__(self, 
               acc=dict(
                  type="torchmetrics.Accuracy",
                  task="multiclass", 
                  num_classes=None,
                  sync_on_compute=False,
                ), 
                auc=dict(
                  type="torchmetrics.AUROC",
                  task="multiclass",
                  num_classes=None,
                  sync_on_compute=False,
                ), 
               kappa=dict(
                  type="torchmetrics.CohenKappa",
                  task="multiclass",
                  num_classes=None,
                  sync_on_compute=False,
                )
              ):
    super().__init__()
    self.acc = build_metric(acc)
    self.auc = build_metric(auc)
    self.kappa = build_metric(kappa)
    return

  @torch.inference_mode()
  def forward(self, 
              output: Dict[str, torch.Tensor], 
              targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    softmax = output["softmax"]
    labels = targets["label"]
    return dict(
      acc=self.acc(softmax, labels),
      auc=self.auc(softmax, labels),
      kappa=self.kappa(softmax, labels),
    )
