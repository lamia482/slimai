import torch

from slimai.models.feature.lora import LoRALinear


def test_lora_export_matches_forward():
  linear = torch.nn.Linear(16, 8)
  lora = LoRALinear(linear, r=4, alpha=1.0, dropout=0.0)
  lora.eval()
  x = torch.randn(3, 16)
  export_module = lora.export_model()
  assert torch.allclose(export_module(x), lora(x), atol=1e-5)
