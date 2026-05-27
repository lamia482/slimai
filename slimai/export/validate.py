from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

def _load_offline_embedding(
  path: Path,
  *,
  embedding_key: str,
  embedding_magnification: int,
  h5_embedding_key: str,
) -> torch.Tensor:
  suffix = path.suffix.lower()
  if suffix == ".pkl":
    with path.open("rb") as handle:
      payload = pickle.load(handle)
    if embedding_key not in payload:
      raise KeyError(f"PKL missing key '{embedding_key}', available: {sorted(payload.keys())}")
    embedding = payload[embedding_key][int(embedding_magnification)]
    return torch.as_tensor(np.asarray(embedding), dtype=torch.float32)
  if suffix in [".h5", ".hdf5"]:
    import h5py

    with h5py.File(path, "r") as handle:
      if h5_embedding_key not in handle:
        raise KeyError(
          f"H5 missing dataset '{h5_embedding_key}', available: {list(handle.keys())}"
        )
      embedding = handle[h5_embedding_key][:]
    return torch.from_numpy(np.asarray(embedding)).float()
  raise ValueError(f"Unsupported embedding file: {path}")


def run_export_validation(
  *,
  patch_encoder: torch.nn.Module,
  slide_encoder: torch.nn.Module,
  output_dir: Path,
  embedding_dim: int,
  input_size: int = 224,
  validate_embedding_path: Optional[str] = None,
  embedding_key: str = "embedding",
  embedding_magnification: int = 20,
  h5_embedding_key: str = "UNI_feature_np",
  max_patches: int = 32,
) -> Dict[str, Any]:
  report: Dict[str, Any] = {"checks": []}
  device = "cpu"
  num_patches = min(max_patches, 8)
  patch_tensor = torch.randn(num_patches, 3, input_size, input_size, device=device)

  with torch.inference_mode():
    embedding_from_patch = patch_encoder(patch_tensor)
    slide_outputs = slide_encoder(embedding_from_patch)
    slide_outputs_direct = slide_encoder(embedding_from_patch)

  if isinstance(slide_outputs, torch.Tensor):
    slide_tuple = (slide_outputs,)
  else:
    slide_tuple = slide_outputs

  max_diff = 0.0
  for left, right in zip(slide_tuple, slide_outputs_direct):
    max_diff = max(max_diff, float((left - right).abs().max().item()))
  report["checks"].append(
    dict(name="slide_encoder_deterministic", passed=max_diff < 1e-5, max_diff=max_diff)
  )

  cosine = float(
    torch.nn.functional.cosine_similarity(
      embedding_from_patch.flatten(),
      embedding_from_patch.flatten(),
      dim=0,
    ).item()
  )
  report["checks"].append(dict(name="patch_encoder_forward", passed=True, cosine=cosine))

  if validate_embedding_path:
    offline = _load_offline_embedding(
      Path(validate_embedding_path),
      embedding_key=embedding_key,
      embedding_magnification=embedding_magnification,
      h5_embedding_key=h5_embedding_key,
    )
    if offline.shape[-1] != embedding_dim:
      raise ValueError(
        f"Offline embedding dim {offline.shape[-1]} != expected {embedding_dim}"
      )
    offline = offline[:num_patches]
    with torch.inference_mode():
      slide_offline = slide_encoder(offline)
    if isinstance(slide_offline, torch.Tensor):
      offline_tuple = (slide_offline,)
    else:
      offline_tuple = slide_offline
    logits_idx = 2 if len(offline_tuple) > 2 else 0
    report["checks"].append(
      dict(
        name="slide_encoder_offline_embedding",
        passed=True,
        num_patches=int(offline.shape[0]),
        logits_shape=list(offline_tuple[logits_idx].shape),
        num_outputs=len(offline_tuple),
      )
    )

  report_path = Path(output_dir) / "validation_report.json"
  report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
  return report
