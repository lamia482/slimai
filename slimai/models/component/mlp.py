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
               norm=None,
               dropout=0.0):
    super().__init__()
    assert (
      n_layer >= 1
    ), "n_layer must be greater than or equal to 1, but got {}".format(n_layer)

    kwargs = dict(act=act, norm=norm, dropout=dropout)
    self.mlp = torch.nn.Sequential(
      self.create_layer(input_dim, (bottleneck_dim if n_layer == 1 else hidden_dim), **kwargs),
      *[self.create_layer(hidden_dim, hidden_dim, **kwargs) for _ in range(0, n_layer-2)],
      *[self.create_layer(hidden_dim, bottleneck_dim, **kwargs) for _ in range(max(0, n_layer-2), n_layer-1)],
      self.create_layer(bottleneck_dim, output_dim, act=None, norm=None, dropout=dropout),
    )
    return
  
  def forward(self, x):
    return self.mlp(x)
  
  def create_layer(self, in_features, out_features, act="relu", norm=None, dropout=0.0):
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
      None: torch.nn.Identity(),
      "layer_norm": torch.nn.LayerNorm,
      "batch_norm": torch.nn.BatchNorm2d,
    }
    assert (
      norm in norm_dict
    ), "norm must be one of {}, but got {}".format(norm_dict.keys(), norm)

    return torch.nn.Sequential(
      torch.nn.Linear(in_features, out_features),
      act_dict[act],
      norm_dict[norm], 
      torch.nn.Dropout(dropout),
    )