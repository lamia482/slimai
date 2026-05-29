import torch
from typing import Dict, List, Optional
from slimai.helper.help_build import MODELS, build_loss


__all__ = [
  "MILLoss",
  "HierarchicalMILLoss",
]

@MODELS.register_module()
class MILLoss(torch.nn.Module):
  def __init__(self, 
               atten_loss=False, 
               cls_loss=dict(
                 type="torch.nn.CrossEntropyLoss",
                 label_smoothing=0.1,
               )):
    super().__init__()
    self.atten_loss = atten_loss
    # Initialize classification loss
    self.cls_loss = build_loss(cls_loss)
    return
  
  def forward(self, 
              backbone: List[torch.Tensor], 
              atten_logits: List[torch.Tensor], 
              cls_logits: torch.Tensor, 
              cls_targets: torch.Tensor) -> Dict[str, torch.Tensor]:
    loss = dict()

    # Compute attention loss
    if self.atten_loss:
      attention_loss = 0
      for patch_embeds, patch_logits in zip(backbone, atten_logits):
        attention_targets = torch.cosine_similarity(patch_embeds[:, None, ...], patch_embeds[None, :, ...], dim=-1) # [N, N]
        patch_logits = patch_logits.sigmoid() # [N, 1]
        attention_logits = torch.cdist(patch_logits[..., None], patch_logits[..., None], p=2).abs() # [N, N]
        attention_logits = 1 - attention_logits
        attention_loss += self.cls_loss(attention_logits, attention_targets)
      loss["atten_loss"] = attention_loss / len(backbone)

    # Compute classification loss
    cls_loss = self.cls_loss(cls_logits, cls_targets)
    loss["cls_loss"] = cls_loss

    return loss


def _loss_cfg_with_num_classes(loss_cfg: dict, num_classes: int) -> dict:
  cfg = loss_cfg.copy()
  loss_type = str(cfg.get("type", ""))
  if loss_type in ("SoftmaxFocalLoss", "FocalLoss"):
    cfg["num_classes"] = int(num_classes)
  return cfg


@MODELS.register_module()
class HierarchicalMILLoss(torch.nn.Module):
  def __init__(
    self,
    *,
    primary_loss: Optional[dict] = None,
    secondary_loss: Optional[dict] = None,
    secondary_num_classes: Optional[Dict[str, int]] = None,
    consistency_loss_weight: float = 0.0,
    loss_weighting: str = "kendall",
    log_var_min: float = -5.0,
    log_var_max: float = 5.0,
  ):
    super().__init__()
    if primary_loss is None:
      primary_loss = dict(type="torch.nn.CrossEntropyLoss", label_smoothing=0.0)
    if secondary_loss is None:
      secondary_loss = dict(type="torch.nn.CrossEntropyLoss", label_smoothing=0.0)
    self.primary_loss = build_loss(primary_loss)
    secondary_loss_template = secondary_loss.copy()
    if isinstance(secondary_num_classes, dict) and len(secondary_num_classes) > 0:
      self.secondary_losses = torch.nn.ModuleDict({
        name: build_loss(_loss_cfg_with_num_classes(secondary_loss_template, num_classes))
        for name, num_classes in secondary_num_classes.items()
      })
      self.secondary_loss = None
    else:
      self.secondary_losses = None
      self.secondary_loss = build_loss(secondary_loss_template)
    self.consistency_loss_weight = float(consistency_loss_weight)
    self.loss_weighting = loss_weighting
    self.log_var_min = float(log_var_min)
    self.log_var_max = float(log_var_max)

    # Kendall uncertainty weighting parameters.
    self.log_var_primary = torch.nn.Parameter(torch.zeros(()))
    self.log_var_secondary = torch.nn.Parameter(torch.zeros(()))
    return

  def resolve_secondary_loss(self, parent_key: str) -> torch.nn.Module:
    if self.secondary_losses is not None:
      if parent_key not in self.secondary_losses:
        raise KeyError(f"secondary loss for parent_key={parent_key!r} is not registered")
      return self.secondary_losses[parent_key]
    if self.secondary_loss is None:
      raise ValueError("secondary loss is not configured")
    return self.secondary_loss

  def clamp_log_vars(self):
    with torch.no_grad():
      self.log_var_primary.clamp_(self.log_var_min, self.log_var_max)
      self.log_var_secondary.clamp_(self.log_var_min, self.log_var_max)
    return

  def forward(
    self,
    *,
    primary_loss_value: torch.Tensor,
    secondary_loss_value: torch.Tensor,
    secondary_valid_mask: torch.Tensor,
    consistency_loss: Optional[torch.Tensor] = None,
  ) -> Dict[str, torch.Tensor]:
    self.clamp_log_vars()
    if secondary_valid_mask.numel() > 0:
      secondary_valid_mask = secondary_valid_mask.to(device=primary_loss_value.device, dtype=torch.bool)
    primary_loss = primary_loss_value
    secondary_loss_raw = secondary_loss_value

    if self.loss_weighting == "kendall":
      primary_weight = torch.exp(-self.log_var_primary)
      secondary_weight = torch.exp(-self.log_var_secondary)
      weighted_loss = (
        primary_weight * primary_loss
        + secondary_weight * secondary_loss_raw
        + self.log_var_primary
        + self.log_var_secondary
      )
    elif self.loss_weighting == "fixed":
      primary_weight = torch.ones_like(primary_loss)
      secondary_weight = torch.ones_like(secondary_loss_raw)
      weighted_loss = primary_loss + secondary_loss_raw
    else:
      raise ValueError(f"Unsupported loss_weighting: {self.loss_weighting}")

    consistency_loss_value = (
      torch.zeros((), device=primary_loss.device, dtype=primary_loss.dtype)
      if consistency_loss is None
      else consistency_loss
    )
    total_loss = weighted_loss + self.consistency_loss_weight * consistency_loss_value
    return dict(
      loss=total_loss,
      composite_loss=total_loss.detach(),
      primary_loss=primary_loss.detach(),
      secondary_loss=secondary_loss_raw.detach(),
      consistency_loss=consistency_loss_value.detach(),
      kendall_primary_weight=primary_weight.detach(),
      kendall_secondary_weight=secondary_weight.detach(),
    )