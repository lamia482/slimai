import torch
import torch.nn.functional as F
import torchvision
from slimai.helper.help_build import MODELS


@MODELS.register_module("FocalLoss")
class FocalLoss(torch.nn.Module):
  def __init__(self, num_classes, *, weight=None, alpha=0.25, gamma=2.0):
    super().__init__()
    self.num_classes = num_classes
    self.weight = torch.as_tensor(weight) if weight is not None else None
    self.alpha = alpha
    self.gamma = gamma
    return

  def forward(self, logits, targets, reduction="mean"):
    targets_one_hot = F.one_hot(targets, num_classes=self.num_classes+1
                               ).type_as(logits)[..., :-1]
    focal_loss = torchvision.ops.sigmoid_focal_loss(logits, targets_one_hot, reduction="none")
    if self.weight is not None:
      focal_loss = focal_loss * self.weight

    if reduction == "mean":
      focal_loss = focal_loss.mean()
    elif reduction == "sum":
      focal_loss = focal_loss.sum()
    elif reduction == "none":
      pass
    else:
      raise ValueError(f"Invalid reduction: {reduction}")

    return focal_loss