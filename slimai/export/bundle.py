from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from mmengine.config import Config

from slimai.helper import help_build
from slimai.models.arch.base_arch import BaseArch


def check_state_dict_compat(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
  model_state = model.state_dict()
  model_keys = set(model_state.keys())
  ckpt_keys = set(state_dict.keys())
  missing = sorted(model_keys - ckpt_keys)
  unexpected = sorted(ckpt_keys - model_keys)
  shape_mismatch = []
  for key in sorted(model_keys & ckpt_keys):
    if tuple(model_state[key].shape) != tuple(state_dict[key].shape):
      shape_mismatch.append(
        f"{key}: model {tuple(model_state[key].shape)} vs ckpt {tuple(state_dict[key].shape)}"
      )
  messages = []
  if missing:
    messages.append(f"missing keys ({len(missing)}): {missing[:8]}{'...' if len(missing) > 8 else ''}")
  if unexpected:
    messages.append(f"unexpected keys ({len(unexpected)}): {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
  if shape_mismatch:
    messages.append(f"shape mismatch: {shape_mismatch[:4]}{'...' if len(shape_mismatch) > 4 else ''}")
  if messages:
    raise ValueError("Checkpoint is incompatible with config-built model. " + "; ".join(messages))
  return


def build_secondary_head_keys(cfg: Config) -> list[str]:
  """Return global secondary class names aligned with flat secondary outputs."""
  if hasattr(cfg, "SECONDARY_HEAD_KEYS"):
    return list(getattr(cfg, "SECONDARY_HEAD_KEYS"))

  primary_keys = getattr(cfg, "PRIMARY_HEAD_KEYS", None) or []
  canonical_local = getattr(cfg, "SECONDARY_CANONICAL_LOCAL_MAPPING", None)
  if primary_keys and isinstance(canonical_local, dict) and len(canonical_local) > 0:
    names: list[str] = []
    for primary_name in primary_keys:
      local_mapping = canonical_local.get(primary_name, {})
      if not isinstance(local_mapping, dict):
        continue
      ordered = sorted(local_mapping.items(), key=lambda item: int(item[1]))
      names.extend(str(name) for name, _ in ordered)
    if len(names) > 0:
      return names

  secondary_mapping = getattr(cfg, "SECONDARY_LABEL_MAPPING", None)
  if not isinstance(secondary_mapping, dict) or len(secondary_mapping) == 0:
    return []

  max_index = -1
  for index in secondary_mapping.values():
    if isinstance(index, int):
      max_index = max(max_index, int(index))
  if max_index < 0:
    return []

  names = [str(i) for i in range(max_index + 1)]
  for name, index in secondary_mapping.items():
    if isinstance(index, int) and 0 <= int(index) < len(names):
      names[int(index)] = str(name)
  return names


def extract_taxonomy(cfg: Config) -> Dict[str, Any]:
  taxonomy = {}
  for key in (
    "PRIMARY_HEAD_KEYS",
    "SECONDARY_GLOBAL_PARENT_IDX",
    "SECONDARY_GLOBAL_LOCAL_IDX",
    "NUM_CLASSES",
    "EMBEDDING_DIM",
    "PATCH_ENCODER_NAME",
    "TARGET_NAME",
  ):
    if hasattr(cfg, key):
      taxonomy[key] = getattr(cfg, key)

  secondary_head_keys = build_secondary_head_keys(cfg)
  if len(secondary_head_keys) > 0:
    taxonomy["SECONDARY_HEAD_KEYS"] = secondary_head_keys
    taxonomy["NUM_SECONDARY_CLASSES"] = len(secondary_head_keys)

  if hasattr(cfg, "SECONDARY_CANONICAL_LOCAL_MAPPING"):
    taxonomy["SECONDARY_CANONICAL_LOCAL_MAPPING"] = getattr(cfg, "SECONDARY_CANONICAL_LOCAL_MAPPING")

  return taxonomy


def load_training_bundle(
  config_path: str,
  ckpt_path: str,
) -> Tuple[Config, BaseArch, Dict[str, Any]]:
  config_file = Path(config_path)
  if not config_file.is_file():
    raise FileNotFoundError(f"Config not found: {config_path}")

  ckpt_file = Path(ckpt_path)
  if not ckpt_file.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

  cfg = Config.fromfile(str(config_file))
  arch: BaseArch = help_build.build_model(cfg.MODEL) # type: ignore
  ckpt = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
  if "weight" not in ckpt:
    raise KeyError(
      f"Checkpoint must contain 'weight' key, got keys: {sorted(ckpt.keys())}"
    )
  weight = ckpt["weight"]
  check_state_dict_compat(arch.model, weight)
  arch.load_state_dict(weight, strict=True)
  return cfg, arch, ckpt
