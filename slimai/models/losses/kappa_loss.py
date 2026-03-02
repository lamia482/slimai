import torch
import torch.nn.functional as F
from typing import Dict
from slimai.helper.help_build import MODELS, build_loss


@MODELS.register_module()
class KappaLoss(torch.nn.Module):
  def __init__(self, num_classes, scale=1.0, eps=1e-8):
    super().__init__()
    self.num_classes = num_classes
    self.scale = scale
    self.eps = eps
    return
  
  def forward(self, 
              logits: torch.Tensor, 
              targets: torch.Tensor) -> Dict[str, torch.Tensor]:
      probs = torch.softmax(logits, dim=1)  # [B, C]
      y_true = F.one_hot(targets, self.num_classes).float()  # [B, C]
      sum_probs = probs.sum(dim=0, keepdim=True)      # [1, C]
      sum_true = y_true.sum(dim=0, keepdim=True)      # [1, C]
      expected_matrix = torch.mm(sum_true.t(), sum_probs) / probs.size(0)  # [C, C]
      observed_matrix = torch.mm(y_true.t(), probs)  # [C, C]
      numerator = torch.sum(observed_matrix * torch.eye(self.num_classes, device=logits.device))
      denominator = torch.sum(expected_matrix * torch.eye(self.num_classes, device=logits.device))
      kappa = (numerator - denominator) / (probs.size(0) - denominator + self.eps)
      kappa_loss = 1.0 - kappa
      return kappa_loss * self.scale
    
