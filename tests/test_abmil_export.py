import torch

from slimai.models.component.abmil import ABMIL


def test_abmil_export_matches_single_bag_forward():
  abmil = ABMIL(
    input_dim=32,
    hidden_dim=16,
    attention="gated",
    dropout=0.0,
    attention_temperature=1.0,
  )
  abmil.eval()
  embedding = torch.randn(10, 32)
  export_module = abmil.export_model()
  bag_export, attn_export = export_module(embedding)
  bag_train, attn_train = abmil([embedding])
  assert torch.allclose(bag_export, bag_train[0], atol=1e-5)
  assert torch.allclose(attn_export, attn_train[0], atol=1e-5)
