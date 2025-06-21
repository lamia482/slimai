from typing import Optional
import torch
from slimai.helper.help_build import MODELS


@MODELS.register_module()
class MLP(torch.nn.Module):
  def __init__(self, 
               *, 
               input_dim,
               output_dim, 
               hidden_dim=2048, 
               bottleneck_dim=256, 
               n_layer=3, 
               act="relu", 
               norm=None, 
               dropout=None):
    super().__init__()
    assert (
      n_layer >= 1
    ), "n_layer must be greater than or equal to 1, but got {}".format(n_layer)

    self.act = self.get_act(act)
    self.norm = torch.nn.ModuleDict({
      str(hidden_dim): self.get_norm(norm)(hidden_dim),
      str(bottleneck_dim): self.get_norm(norm)(bottleneck_dim),
    })
    self.dropout = self.get_dropout(dropout)
    self.n_layer = n_layer

    if n_layer == 1:
      layers = [self.get_linear(input_dim, output_dim)]
    elif n_layer == 2:
      layers = [
        self.get_linear(input_dim, bottleneck_dim), 
        self.get_linear(bottleneck_dim, output_dim)
      ]
    else: # n_layer >= 3
      layers = [
        self.get_linear(input_dim, hidden_dim),
        *[self.get_linear(hidden_dim, hidden_dim) for _ in range(n_layer - 3)],
        self.get_linear(hidden_dim, bottleneck_dim),
        self.get_linear(bottleneck_dim, output_dim)
      ]

    self.layers = torch.nn.ModuleList(layers)
    return
  
  def forward(self, x):
    for i, layer in enumerate(self.layers):
      x = layer(x)
      
      if i < self.n_layer - 1:
        norm = self.norm[str(x.size(-1))]
        x = self.dropout(self.act(norm(x)))

    return x
  
  def get_act(self, act: Optional[str]) -> torch.nn.Module:
    act_dict = {
      None: torch.nn.Identity(),
      "relu": torch.nn.ReLU(),
      "gelu": torch.nn.GELU(),
      "tanh": torch.nn.Tanh(),
    }
    return act_dict[act]
  
  def get_norm(self, norm: Optional[str]) -> torch.nn.Module:
    norm_dict = {
      None: lambda nfeat: torch.nn.Identity(),
      "layer_norm": lambda nfeat: torch.nn.LayerNorm(nfeat),
      "batch_norm_1d": lambda nfeat: torch.nn.BatchNorm1d(nfeat), 
      "batch_norm_2d": lambda nfeat: torch.nn.BatchNorm2d(nfeat), 
    }
    return norm_dict[norm]
  
  def get_dropout(self, dropout: Optional[float]) -> torch.nn.Module:
    if dropout is None:
      return torch.nn.Identity()
    return torch.nn.Dropout(dropout)
  
  def get_linear(self, in_features, out_features) -> torch.nn.Module:
    return torch.nn.Linear(in_features, out_features)
  