from __future__ import annotations

from typing import Any, Dict

import torch
from timm.data.config import resolve_data_config

from slimai.helper.help_build import MODELS

from .extract import PATCH_ENCODER_BUILDERS


__all__ = [
  "PatchEncoderBackbone",
]


class _PatchEncoderExport(torch.nn.Module):
  """ONNX-friendly patch encoder: ``patch_tensor [N,3,H,W] -> embedding_arr [N,K]``."""

  def __init__(self, source: "PatchEncoderBackbone"):
    super().__init__()
    self.backbone = source
    return

  def forward(self, patch_tensor: torch.Tensor) -> torch.Tensor:
    return self.backbone(patch_tensor)


@MODELS.register_module()
class PatchEncoderBackbone(torch.nn.Module):
  """Patch-level feature encoder registered in ``MODEL`` configs."""

  def __init__(
    self,
    *,
    encoder_name: str = "UNI",
    cache_dir: str = "/.slimai/cache/huggingface/hub",
    **kwargs,
  ):
    super().__init__()
    if encoder_name not in PATCH_ENCODER_BUILDERS:
      available = sorted(PATCH_ENCODER_BUILDERS.keys())
      raise ValueError(
        f"Unsupported patch encoder: {encoder_name}. Available: {available}"
      )
    encoder, transform, feature_dim = PATCH_ENCODER_BUILDERS[encoder_name](cache_dir)
    self.encoder_name = encoder_name
    self.cache_dir = cache_dir
    self.feature_dim = int(feature_dim)
    self.encoder = encoder
    self._transform = transform
    self._data_config = resolve_data_config(
      getattr(encoder, "pretrained_cfg", {}),
      model=encoder,
    )
    return

  def forward(self, patch_tensor: torch.Tensor) -> torch.Tensor:
    """Encode patches ``[N,3,H,W]`` (already preprocessed) to ``[N,K]``."""
    return self.encoder(patch_tensor)

  def export_model(self) -> torch.nn.Module:
    return _PatchEncoderExport(self).eval()

  def export_preprocess(self) -> Dict[str, Any]:
    cfg = dict(self._data_config)
    input_size = cfg.get("input_size", (3, 224, 224))
    if isinstance(input_size, (list, tuple)) and len(input_size) == 3:
      _, height, width = input_size
    else:
      height, width = 224, 224
    return dict(
      encoder_name=self.encoder_name,
      input_size=[int(height), int(width)],
      mean=list(cfg.get("mean", [0.485, 0.456, 0.406])),
      std=list(cfg.get("std", [0.229, 0.224, 0.225])),
      interpolation=str(cfg.get("interpolation", "bicubic")),
      crop_pct=float(cfg.get("crop_pct", 1.0)),
    )
