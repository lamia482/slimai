import torch
import torch.nn as nn

from slimai.helper.help_build import MODELS


@MODELS.register_module()
class LoRALinear(nn.Module):
  def __init__(
    self,
    linear_layer: nn.Linear,
    r: int,
    alpha: float = 1.0,
    dropout: float = 0.0,
  ):
    super().__init__()
    assert r > 0, f"`r` must be > 0, but got {r}"
    self.linear = linear_layer
    self.r = int(r)
    self.alpha = alpha
    self.scaling = alpha / r

    for param in self.linear.parameters():
      param.requires_grad = False

    self.in_features = linear_layer.in_features
    self.out_features = linear_layer.out_features
    self.lora_A = nn.Parameter(torch.zeros(self.r, self.in_features))
    self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))
    self.dropout = nn.Dropout(dropout)

    self.reset_parameters()
    return

  def reset_parameters(self):
    nn.init.normal_(self.lora_A, std=1e-2)
    nn.init.zeros_(self.lora_B)
    return

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    hidden = self.linear(x)
    lora_out = (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
    return hidden + lora_out

  def merge(self) -> nn.Linear:
    merged_weight = self.linear.weight + (self.lora_B @ self.lora_A) * self.scaling
    merged_bias = self.linear.bias
    merged = nn.Linear(
      self.linear.in_features,
      self.linear.out_features,
      bias=self.linear.bias is not None,
    )
    merged.weight.data = merged_weight
    if merged_bias is not None:
      merged.bias.data = merged_bias
    return merged


def apply_lora_to_vit(
  model: nn.Module,
  r: int = 8,
  alpha: float = 1.0,
  dropout: float = 0.0,
  target_keywords=None,
):
  if target_keywords is None:
    target_keywords = ["qkv", "fc1", "fc2", "proj"]

  target_layers = []
  for name, module in model.named_modules():
    if not isinstance(module, nn.Linear):
      continue
    if any(keyword in name for keyword in target_keywords):
      target_layers.append((name, module))

  named_module_dict = dict(model.named_modules())
  for name, module in target_layers:
    parent_name = ".".join(name.split(".")[:-1])
    child_name = name.split(".")[-1]
    parent = model if parent_name == "" else named_module_dict[parent_name]
    setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
  return model

