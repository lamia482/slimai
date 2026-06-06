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
    marginal_loss: Optional[dict] = None,
    conditional_loss: Optional[dict] = None,
    secondary_num_classes: Optional[Dict[str, int]] = None,
    global_secondary_num_classes: Optional[int] = None,
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
    else:
      self.secondary_losses = None

    global_num = int(global_secondary_num_classes or 0)
    if marginal_loss is None:
      marginal_loss = secondary_loss.copy()
    if conditional_loss is None:
      conditional_loss = secondary_loss.copy()
    self.marginal_loss = build_loss(
      _loss_cfg_with_num_classes(marginal_loss, global_num) if global_num > 0 else marginal_loss
    )
    self.conditional_loss = build_loss(
      _loss_cfg_with_num_classes(conditional_loss, global_num) if global_num > 0 else conditional_loss
    )
    self.loss_weighting = loss_weighting
    self.log_var_min = float(log_var_min)
    self.log_var_max = float(log_var_max)

    self.log_var_primary = torch.nn.Parameter(torch.zeros(()))
    self.log_var_marginal = torch.nn.Parameter(torch.zeros(()))
    self.log_var_conditional = torch.nn.Parameter(torch.zeros(()))
    self.log_var_local_aux = torch.nn.Parameter(torch.zeros(()))
    return

  def resolve_secondary_loss(self, parent_key: str) -> torch.nn.Module:
    if self.secondary_losses is not None:
      if parent_key not in self.secondary_losses:
        raise KeyError(f"secondary loss for parent_key={parent_key!r} is not registered")
      return self.secondary_losses[parent_key]
    raise ValueError("secondary losses are not configured")

  def clamp_log_vars(self):
    with torch.no_grad():
      self.log_var_primary.clamp_(self.log_var_min, self.log_var_max)
      self.log_var_marginal.clamp_(self.log_var_min, self.log_var_max)
      self.log_var_conditional.clamp_(self.log_var_min, self.log_var_max)
      self.log_var_local_aux.clamp_(self.log_var_min, self.log_var_max)
    return

  def forward(
    self,
    *,
    primary_loss_value: torch.Tensor,
    marginal_loss_value: torch.Tensor,
    conditional_loss_value: torch.Tensor,
    local_aux_loss_value: torch.Tensor,
    local_aux_valid_mask: torch.Tensor,
  ) -> Dict[str, torch.Tensor]:
    self.clamp_log_vars()
    if local_aux_valid_mask.numel() > 0:
      local_aux_valid_mask = local_aux_valid_mask.to(
        device=primary_loss_value.device,
        dtype=torch.bool,
      )
    primary_loss = primary_loss_value
    marginal_loss = marginal_loss_value
    conditional_loss = conditional_loss_value
    local_aux_loss = local_aux_loss_value

    if self.loss_weighting == "kendall":
      primary_weight = torch.exp(-self.log_var_primary)
      marginal_weight = torch.exp(-self.log_var_marginal)
      conditional_weight = torch.exp(-self.log_var_conditional)
      local_aux_weight = torch.exp(-self.log_var_local_aux)
      weighted_loss = (
        primary_weight * primary_loss
        + marginal_weight * marginal_loss
        + conditional_weight * conditional_loss
        + local_aux_weight * local_aux_loss
        + self.log_var_primary
        + self.log_var_marginal
        + self.log_var_conditional
        + self.log_var_local_aux
      )
    elif self.loss_weighting == "fixed":
      primary_weight = torch.ones_like(primary_loss)
      marginal_weight = torch.ones_like(marginal_loss)
      conditional_weight = torch.ones_like(conditional_loss)
      local_aux_weight = torch.ones_like(local_aux_loss)
      weighted_loss = primary_loss + marginal_loss + conditional_loss + local_aux_loss
    else:
      raise ValueError(f"Unsupported loss_weighting: {self.loss_weighting}")

    return dict(
      loss=weighted_loss,
      composite_loss=weighted_loss.detach(),
      primary_loss=primary_loss.detach(),
      marginal_loss=marginal_loss.detach(),
      conditional_loss=conditional_loss.detach(),
      local_aux_loss=local_aux_loss.detach(),
      kendall_primary_weight=primary_weight.detach(),
      kendall_marginal_weight=marginal_weight.detach(),
      kendall_conditional_weight=conditional_weight.detach(),
      kendall_local_aux_weight=local_aux_weight.detach(),
    )
