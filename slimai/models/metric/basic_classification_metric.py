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
                ),
               f1=None,
              ):
    super().__init__()
    self.acc = build_metric(acc)
    self.auc = build_metric(auc)
    self.kappa = build_metric(kappa)
    if f1 is None:
      f1 = dict(
        type="torchmetrics.F1Score",
        task="multiclass",
        average="macro",
        num_classes=acc.get("num_classes", None),
        sync_on_compute=False,
      )
    self.f1 = build_metric(f1)
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
      f1=self.f1(softmax, labels),
    )
