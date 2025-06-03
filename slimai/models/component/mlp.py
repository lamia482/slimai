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
               n_layer=1, 
               act="relu", 
               norm="batch_norm",
               dropout=0.0):
    super().__init__()
    assert (
      n_layer >= 1
    ), "n_layer must be greater than or equal to 1, but got {}".format(n_layer)

    kwargs = dict(act=act, norm=norm, dropout=dropout)
    self.mlp = torch.nn.Sequential(
      self.create_layer(input_dim, (bottleneck_dim if n_layer == 1 else hidden_dim), **kwargs), # type: ignore
      *[self.create_layer(hidden_dim, hidden_dim, **kwargs) for _ in range(0, n_layer-2)], # type: ignore
      *[self.create_layer(hidden_dim, bottleneck_dim, **kwargs) for _ in range(max(0, n_layer-2), n_layer-1)], # type: ignore
    )
    self.last_layer = self.create_layer(bottleneck_dim, output_dim, act=None, norm=None, dropout=dropout)
    return
  
  def forward(self, x):
    x = self.mlp(x)
    x = torch.nn.functional.normalize(x, p=2, dim=-1)
    x = self.last_layer(x)
    return x
  
  def create_layer(self, in_features, out_features, 
                   act: Optional[str] = "relu", norm: Optional[str] = None, 
                   dropout: float = 0.0) -> torch.nn.Sequential:
    act_dict = {
      None: torch.nn.Identity(),
      "relu": torch.nn.ReLU(),
      "gelu": torch.nn.GELU(),
      "tanh": torch.nn.Tanh(),
      "sigmoid": torch.nn.Sigmoid(),
    }
    assert (
      act in act_dict
    ), "act must be one of {}, but got {}".format(act_dict.keys(), act)

    norm_dict = {
      None: lambda nfeat: torch.nn.Identity(),
      "layer_norm": lambda nfeat: torch.nn.LayerNorm(nfeat),
      "batch_norm_1d": lambda nfeat: torch.nn.BatchNorm1d(nfeat), 
      "batch_norm_2d": lambda nfeat: torch.nn.BatchNorm2d(nfeat), 
      
    }
    assert (
      norm in norm_dict
    ), "norm must be one of {}, but got {}".format(norm_dict.keys(), norm)

    return torch.nn.Sequential(
      torch.nn.Linear(in_features, out_features),
      act_dict[act],
      norm_dict[norm](out_features), 
      torch.nn.Dropout(dropout),
    )