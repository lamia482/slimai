import torch

from slimai.helper.features import extract as feature_extract
from slimai.models.arch.mil import MIL
from slimai.models.component.abmil import ABMIL
from slimai.models.component.mlp import MLP


def _fake_uni_builder(cache_dir: str):
  del cache_dir
  encoder = torch.nn.Sequential(
    torch.nn.AdaptiveAvgPool2d(1),
    torch.nn.Flatten(),
    torch.nn.Linear(3, 32),
  )
  encoder.eval()
  return encoder, (lambda tensor: tensor), 32


def test_mil_export_artifacts_dual_onnx(monkeypatch):
  monkeypatch.setitem(feature_extract.PATCH_ENCODER_BUILDERS, "UNI", _fake_uni_builder)
  arch = MIL(
    backbone=dict(type="PatchEncoderBackbone", encoder_name="UNI", cache_dir="/tmp"),
    neck=dict(type="ABMIL", input_dim=32, hidden_dim=16, attention="gated", dropout=0.0),
    head=dict(type="MLP", input_dim=32, output_dim=3, n_layer=2, dropout=0.0),
    freeze_backbone=True,
  )
  artifacts = arch.export_artifacts()
  patch_encoder = artifacts["patch_encoder"]
  slide_encoder = artifacts["slide_encoder"]
  patches = torch.randn(5, 3, 8, 8)
  embedding = patch_encoder(patches)
  outputs = slide_encoder(embedding)
  assert embedding.shape == (5, 32)
  assert len(outputs) == 4


def test_mil_export_rejects_identity_backbone(monkeypatch):
  monkeypatch.setitem(feature_extract.PATCH_ENCODER_BUILDERS, "UNI", _fake_uni_builder)
  arch = MIL(
    backbone=dict(type="torch.nn.Identity"),
    neck=dict(type="ABMIL", input_dim=32, hidden_dim=16, attention="gated", dropout=0.0),
    head=dict(type="MLP", input_dim=32, output_dim=3, n_layer=2, dropout=0.0),
    freeze_backbone=True,
  )
  try:
    arch.export_artifacts()
    raised = False
  except ValueError:
    raised = True
  assert raised
