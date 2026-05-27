from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from mmengine.model.utils import revert_sync_batchnorm

from slimai.export.bundle import extract_taxonomy, load_training_bundle
from slimai.export.manifest import write_export_manifest
from slimai.export.onnx_core import export_onnx
from slimai.export.validate import run_export_validation
from slimai.helper import help_utils
from slimai.models.arch.mil import (
  HierarchicalMIL,
  MIL,
  SLIDE_ENCODER_INPUT_NAME,
  SLIDE_ENCODER_OUTPUT_NAMES_HIERARCHICAL,
  SLIDE_ENCODER_OUTPUT_NAMES_MIL,
)

PATCH_ENCODER_INPUT_NAME = "patch_tensor"
PATCH_ENCODER_OUTPUT_NAME = "embedding_arr"


class Exporter:
  """Export MIL / HierarchicalMIL checkpoints to patch_encoder + slide_encoder ONNX."""

  def __init__(
    self,
    config_path: str,
    ckpt_path: str,
    *,
    cache_dir: Optional[str] = None,
    disable_log: bool = False,
  ):
    self.config_path = config_path
    self.ckpt_path = ckpt_path
    self.cache_dir = cache_dir
    self.disable_log = disable_log
    self.cfg, self.arch, self.ckpt = load_training_bundle(config_path, ckpt_path)
    if not isinstance(self.arch, (MIL, HierarchicalMIL)):
      raise NotImplementedError(
        f"Export supports MIL and HierarchicalMIL only, got {type(self.arch).__name__}"
      )
    self.slide_output_names = (
      SLIDE_ENCODER_OUTPUT_NAMES_HIERARCHICAL
      if isinstance(self.arch, HierarchicalMIL)
      else SLIDE_ENCODER_OUTPUT_NAMES_MIL
    )
    self.artifacts = self.arch.export_artifacts(cache_dir=cache_dir, cfg=self.cfg)
    self.patch_encoder = revert_sync_batchnorm(self.artifacts["patch_encoder"]).eval()
    self.slide_encoder = revert_sync_batchnorm(self.artifacts["slide_encoder"]).eval()
    self.preprocess = self.artifacts.get("preprocess", {})
    help_utils.print_log("Export bundle loaded successfully", disable_log=self.disable_log)
    return

  def export(
    self,
    output_dir: str,
    *,
    opset_version: int = 17,
    device: str = "cpu",
    skip_validation: bool = False,
    validate_embedding_path: Optional[str] = None,
    max_patches: int = 32,
  ) -> Dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    embedding_dim = int(getattr(self.cfg, "EMBEDDING_DIM", 1024))
    input_size = int(self.preprocess.get("input_size", [224, 224])[-1])
    num_patches = min(max_patches, 8)

    patch_dummy = torch.randn(num_patches, 3, input_size, input_size, device=device)
    embedding_dummy = torch.randn(num_patches, embedding_dim, device=device)

    patch_onnx = export_onnx(
      self.patch_encoder,
      patch_dummy,
      output_path / "patch_encoder.onnx",
      input_names=[PATCH_ENCODER_INPUT_NAME],
      output_names=[PATCH_ENCODER_OUTPUT_NAME],
      dynamic_axes={
        PATCH_ENCODER_INPUT_NAME: {0: "num_patches"},
        PATCH_ENCODER_OUTPUT_NAME: {0: "num_patches"},
      },
      opset_version=opset_version,
      device=device,
    )
    slide_onnx = export_onnx(
      self.slide_encoder,
      embedding_dummy,
      output_path / "slide_encoder.onnx",
      input_names=[SLIDE_ENCODER_INPUT_NAME],
      output_names=self.slide_output_names,
      dynamic_axes={
        SLIDE_ENCODER_INPUT_NAME: {0: "num_patches"},
        "attention_weights": {0: "num_patches"},
      },
      opset_version=opset_version,
      device=device,
    )

    taxonomy = extract_taxonomy(self.cfg)
    write_export_manifest(
      output_path,
      config_path=self.config_path,
      ckpt_path=self.ckpt_path,
      preprocess=self.preprocess,
      taxonomy=taxonomy,
      patch_encoder_spec=dict(
        encoder_name=self.preprocess.get("encoder_name", getattr(self.cfg, "PATCH_ENCODER_NAME", "")),
        inputs=[{"name": PATCH_ENCODER_INPUT_NAME, "shape": ["N", 3, input_size, input_size]}],
        outputs=[{"name": PATCH_ENCODER_OUTPUT_NAME, "shape": ["N", embedding_dim]}],
      ),
      slide_encoder_spec=dict(
        model_type=type(self.arch).__name__,
        inputs=[{"name": SLIDE_ENCODER_INPUT_NAME, "shape": ["N", embedding_dim]}],
        outputs=[{"name": name} for name in self.slide_output_names],
      ),
    )

    if not skip_validation:
      run_export_validation(
        patch_encoder=self.patch_encoder,
        slide_encoder=self.slide_encoder,
        output_dir=output_path,
        embedding_dim=embedding_dim,
        input_size=input_size,
        validate_embedding_path=validate_embedding_path,
        embedding_key=str(getattr(self.cfg, "EMBEDDING_KEY", "embedding")),
        embedding_magnification=int(getattr(self.cfg, "EMBEDDING_MAGNIFICATION", 20)),
        h5_embedding_key=str(getattr(self.cfg, "EXTERNAL_H5_EMBEDDING_KEY", "UNI_feature_np")),
        max_patches=max_patches,
      )

    help_utils.print_log(f"Exported ONNX artifacts to {output_path}", disable_log=self.disable_log)
    return dict(
      patch_encoder=patch_onnx,
      slide_encoder=slide_onnx,
      manifest=output_path / "export_manifest.json",
      preprocess=output_path / "preprocess.json",
    )
