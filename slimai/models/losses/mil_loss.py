import torch
from typing import Dict, List
from slimai.helper.help_build import MODELS, build_loss


__all__ = [
  "MILLoss", 
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