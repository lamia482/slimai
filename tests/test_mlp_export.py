import torch

from slimai.models.component.mlp import MLP


def test_mlp_export_matches_forward():
  mlp = MLP(
    input_dim=32,
    output_dim=4,
    hidden_dim=64,
    bottleneck_dim=16,
    n_layer=3,
    act="relu",
    norm="layer_norm",
    dropout=0.0,
  )
  mlp.eval()
  sample = torch.randn(2, 32)
  export_module = mlp.export_model()
  assert torch.allclose(export_module(sample), mlp(sample), atol=1e-5)
