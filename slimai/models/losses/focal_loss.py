import torch
import torch.nn.functional as F
import torchvision
from slimai.helper.help_build import MODELS


@MODELS.register_module()
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


@MODELS.register_module()
class SoftmaxFocalLoss(torch.nn.Module):
  def __init__(self, num_classes, *, alpha=None, gamma=2.0, reduction='mean'):
    """
    Args:
      num_classes: int, number of classes
      alpha: 类别权重, shape [C] or float (balance positive and negative examples) or None
      gamma: 聚焦参数
      reduction: 'mean' or 'sum' or 'none'
    """
    super().__init__()
    self.num_classes = num_classes
    self.alpha = alpha
    self.gamma = gamma
    self.reduction = reduction

  def forward(self, logits, targets):
    """
    logits: [N, C] (raw logits, no softmax)
    targets: [N] (long tensor, values in [0, C])
    """
    # 数值稳定版 log_softmax
    log_probs = F.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)

    # One-hot encode targets
    one_hot = F.one_hot(targets, num_classes=self.num_classes).float()  # [N, C]

    # 计算 focal weight: (1 - p_t)^gamma
    focal_weight = (1 - probs) ** self.gamma

    # 应用 alpha（类别权重）
    if self.alpha is not None:
      if not isinstance(self.alpha, (tuple, list)):
        self.alpha = [self.alpha] * self.num_classes
      alpha_t = torch.tensor(self.alpha, device=logits.device, dtype=torch.float)[None, :]  # [1, C]
      focal_weight = alpha_t * focal_weight

    # Focal loss: -alpha * (1-p)^gamma * log(p)
    loss = -focal_weight * one_hot * log_probs

    # Reduce
    loss = loss.sum(dim=1)  # [N]
    if self.reduction == 'mean':
      return loss.mean()
    elif self.reduction == 'sum':
      return loss.sum()
    else:
      return loss