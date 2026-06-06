import torch
from pathlib import Path
from mmengine.config import ConfigDict

from slimai.export.onnx_core import export_onnx
from slimai.export.validate import run_export_validation
from slimai.helper.features import extract as feature_extract
from slimai.models.arch.mil import HierarchicalMIL, SLIDE_ENCODER_OUTPUT_NAMES_HIERARCHICAL


def _fake_uni_builder(cache_dir: str):
  del cache_dir
  encoder = torch.nn.Sequential(
    torch.nn.AdaptiveAvgPool2d(1),
    torch.nn.Flatten(),
    torch.nn.Linear(3, 32),
  )
  encoder.eval()
  return encoder, (lambda tensor: tensor), 32


def test_export_validation_writes_html_and_calibration(tmp_path, monkeypatch):
  monkeypatch.setitem(feature_extract.PATCH_ENCODER_BUILDERS, "UNI", _fake_uni_builder)
  arch = HierarchicalMIL(
    backbone=dict(type="PatchEncoderBackbone", encoder_name="UNI", cache_dir="/tmp"),
    neck=dict(type="ABMIL", input_dim=32, hidden_dim=16, attention="gated", dropout=0.0),
    primary_head=dict(type="MLP", input_dim=32, output_dim=1, n_layer=2, dropout=0.0),
    marginal_head=dict(type="MLP", input_dim=32, output_dim=2, n_layer=2, dropout=0.0),
    secondary_heads=dict(h0=dict(type="MLP", input_dim=32, output_dim=2, n_layer=2, dropout=0.0)),
    primary_head_keys=["h0"],
    secondary_global_parent_idx=[0, 0],
    secondary_global_local_idx=[0, 1],
    solver=ConfigDict(type="torch.optim.SGD", lr=0.01),
    freeze_backbone=True,
  )
  artifacts = arch.export_artifacts()
  patch_encoder = artifacts["patch_encoder"].eval()
  slide_encoder = artifacts["slide_encoder"].eval()
  preprocess = artifacts.get("preprocess", {"input_size": [8, 8]})
  input_size = 8
  patch_onnx = export_onnx(
    patch_encoder,
    torch.randn(16, 3, input_size, input_size),
    tmp_path / "patch_encoder.onnx",
    input_names=["patch_tensor"],
    output_names=["embedding_arr"],
    dynamic_axes={"patch_tensor": {0: "num_patches"}, "embedding_arr": {0: "num_patches"}},
    device="cpu",
  )
  slide_onnx = export_onnx(
    slide_encoder,
    torch.randn(16, 32),
    tmp_path / "slide_encoder.onnx",
    input_names=["embedding_arr"],
    output_names=SLIDE_ENCODER_OUTPUT_NAMES_HIERARCHICAL,
    dynamic_axes={"embedding_arr": {0: "num_patches"}, "attention_weights": {0: "num_patches"}},
    device="cpu",
  )

  class _Cfg:
    TEST_LOADER = {}
    EXTERNAL_TEST_LOADERS = {}

  report = run_export_validation(
    patch_encoder=patch_encoder,
    slide_encoder=slide_encoder,
    patch_onnx_path=patch_onnx,
    slide_onnx_path=slide_onnx,
    slide_output_names=SLIDE_ENCODER_OUTPUT_NAMES_HIERARCHICAL,
    is_hierarchical=True,
    model_type="HierarchicalMIL",
    output_dir=tmp_path,
    cfg=_Cfg(),
    preprocess=preprocess,
    embedding_dim=32,
    input_size=input_size,
    num_trials=2,
    skip_test_eval=True,
    show_progress=False,
  )
  assert (tmp_path / "validation_main.html").exists()
  assert (tmp_path / "calibration_v3_trial0.pkl").exists()
  assert (tmp_path / "calibration_v3_trial0.md").exists()
  assert "timing" in report and "phases" in report["timing"]
  assert report["summary"]["num_trials"] == 2
