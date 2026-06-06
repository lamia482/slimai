from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from mmengine.model.utils import revert_sync_batchnorm

from slimai.export.bundle import extract_taxonomy, load_training_bundle
from slimai.export.label_catalog import attach_label_catalog_to_taxonomy
from slimai.export.manifest import write_export_manifest
from slimai.export.onnx_core import export_onnx
from slimai.export.validate import run_export_validation
from slimai.export.validate_progress import validation_progress
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
    self.is_hierarchical = isinstance(self.arch, HierarchicalMIL)
    self.slide_output_names = (
      SLIDE_ENCODER_OUTPUT_NAMES_HIERARCHICAL
      if self.is_hierarchical
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
    max_patches: int = 32,
    num_trials: int = 3,
    batch_min: int = 8,
    batch_max: int = 32,
    seed: int = 10482,
    deterministic_repeats: int = 3,
    deterministic_tol: float = 1e-8,
    parity_max_tol: float = 5e-5,
    parity_mean_tol: float = 5e-6,
    metrics_tol: float = 5e-4,
    skip_test_eval: bool = False,
    reference_work_dir: Optional[str] = None,
    skip_reference_compare: bool = True,
    ort_provider: str = "CPUExecutionProvider",
    show_progress: Optional[bool] = None,
  ) -> Dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if show_progress is None:
      show_progress = not self.disable_log

    embedding_dim = int(getattr(self.cfg, "EMBEDDING_DIM", 1024))
    input_size = int(self.preprocess.get("input_size", [224, 224])[-1])
    trace_patches = 16

    patch_dummy = torch.randn(trace_patches, 3, input_size, input_size, device=device)
    embedding_dummy = torch.randn(trace_patches, embedding_dim, device=device)

    export_steps = ["patch_encoder", "slide_encoder"]
    step_iter = validation_progress(
      export_steps,
      total=len(export_steps),
      desc="Export ONNX",
      unit="step",
      enabled=show_progress,
      leave=False,
    )
    patch_onnx = None
    slide_onnx = None
    for step in step_iter:
      if step == "patch_encoder":
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
      else:
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
    taxonomy = attach_label_catalog_to_taxonomy(
      taxonomy,
      self.cfg,
      output_dir=output_path,
    )
    export_meta = dict(
      opset_version=opset_version,
      device=device,
      schema_version="hier_v2" if self.is_hierarchical else "mil_v1",
      slide_output_names=list(self.slide_output_names),
    )
    if self.is_hierarchical:
      export_meta["num_secondary_classes"] = int(getattr(self.arch, "global_secondary_num_classes", 0))
      splits = []
      cursor = 0
      for head_name in self.arch.primary_head_keys:
        head_cfg = self.arch.secondary_heads_cfg.get(head_name, {})
        width = int(head_cfg.get("output_dim", 0))
        splits.append(dict(head=head_name, start=cursor, end=cursor + width))
        cursor += width
      export_meta["secondary_flat_splits"] = splits
    write_export_manifest(
      output_path,
      config_path=self.config_path,
      ckpt_path=self.ckpt_path,
      preprocess=self.preprocess,
      taxonomy=taxonomy,
      export_meta=export_meta,
      validation_batch_range=[batch_min, batch_max],
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
        patch_onnx_path=patch_onnx,
        slide_onnx_path=slide_onnx,
        slide_output_names=self.slide_output_names,
        is_hierarchical=self.is_hierarchical,
        model_type=type(self.arch).__name__,
        output_dir=output_path,
        cfg=self.cfg,
        preprocess=self.preprocess,
        embedding_dim=embedding_dim,
        input_size=input_size,
        config_path=self.config_path,
        ckpt_path=self.ckpt_path,
        num_trials=num_trials,
        batch_min=batch_min,
        batch_max=batch_max,
        seed=seed,
        deterministic_repeats=deterministic_repeats,
        deterministic_tol=deterministic_tol,
        parity_max_tol=parity_max_tol,
        parity_mean_tol=parity_mean_tol,
        metrics_tol=metrics_tol,
        skip_test_eval=skip_test_eval,
        reference_work_dir=Path(reference_work_dir) if reference_work_dir else None,
        skip_reference_compare=skip_reference_compare,
        ort_provider=ort_provider,
        show_progress=show_progress,
        disable_log=self.disable_log,
        onnx_meta=dict(opset_version=opset_version, device=device),
      )

    help_utils.print_log(f"Exported ONNX artifacts to {output_path}", disable_log=self.disable_log)
    return dict(
      patch_encoder=patch_onnx,
      slide_encoder=slide_onnx,
      manifest=output_path / "export_manifest.json",
      validation_report=output_path / "validation_main.html",
      calibration_pkl=output_path / "calibration_v3_trial0.pkl",
      calibration_md=output_path / "calibration_v3_trial0.md",
    )
