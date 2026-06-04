from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def file_md5(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
  digest = hashlib.md5()
  with Path(path).open("rb") as handle:
    while True:
      chunk = handle.read(chunk_size)
      if not chunk:
        break
      digest.update(chunk)
  return digest.hexdigest()


def write_export_manifest(
  output_dir: Path,
  *,
  config_path: str,
  ckpt_path: str,
  preprocess: Dict[str, Any],
  taxonomy: Dict[str, Any],
  patch_encoder_spec: Dict[str, Any],
  slide_encoder_spec: Dict[str, Any],
  export_meta: Optional[Dict[str, Any]] = None,
  validation_batch_range: Optional[list] = None,
) -> Path:
  validation_batch_range = validation_batch_range or [8, 32]
  output_dir = Path(output_dir)
  patch_file = "patch_encoder.onnx"
  slide_file = "slide_encoder.onnx"
  patch_path = output_dir / patch_file
  slide_path = output_dir / slide_file

  patch_body = dict(
    weight_source="checkpoint",
    preprocess=dict(preprocess),
    **patch_encoder_spec,
  )
  patch_encoder: Dict[str, Any] = dict(file=patch_file, **patch_body)
  if patch_path.is_file():
    patch_encoder = dict(file=patch_file, md5=file_md5(patch_path), **patch_body)

  slide_body = dict(weight_source="checkpoint", **slide_encoder_spec)
  slide_encoder: Dict[str, Any] = dict(file=slide_file, **slide_body)
  if slide_path.is_file():
    slide_encoder = dict(file=slide_file, md5=file_md5(slide_path), **slide_body)

  manifest = dict(
    created_at=datetime.now(timezone.utc).isoformat(),
    config_path=str(config_path),
    ckpt_path=str(ckpt_path),
    weight_source="checkpoint",
    export=dict(export_meta or {}),
    pipeline=["patch_encoder", "slide_encoder"],
    patch_encoder=patch_encoder,
    slide_encoder=slide_encoder,
    taxonomy=taxonomy,
    dynamic_batch=dict(
      axis=0,
      name="num_patches",
      validation_range=list(validation_batch_range),
    ),
  )
  output_dir.mkdir(parents=True, exist_ok=True)
  manifest_path = output_dir / "export_manifest.json"
  manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
  return manifest_path
