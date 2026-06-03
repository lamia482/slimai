from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def write_export_manifest(
  output_dir: Path,
  *,
  config_path: str,
  ckpt_path: str,
  preprocess: Dict[str, Any],
  taxonomy: Dict[str, Any],
  patch_encoder_spec: Dict[str, Any],
  slide_encoder_spec: Dict[str, Any],
) -> Path:
  manifest = dict(
    created_at=datetime.now(timezone.utc).isoformat(),
    config_path=str(config_path),
    ckpt_path=str(ckpt_path),
    weight_source="checkpoint",
    pipeline=["patch_encoder", "slide_encoder"],
    patch_encoder=dict(
      file="patch_encoder.onnx",
      preprocess_file="preprocess.json",
      weight_source="checkpoint",
      **patch_encoder_spec,
    ),
    slide_encoder=dict(
      file="slide_encoder.onnx",
      weight_source="checkpoint",
      **slide_encoder_spec,
    ),
    taxonomy=taxonomy,
  )
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  preprocess_path = output_dir / "preprocess.json"
  preprocess_path.write_text(json.dumps(preprocess, ensure_ascii=False, indent=2), encoding="utf-8")
  manifest_path = output_dir / "export_manifest.json"
  manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
  return manifest_path
