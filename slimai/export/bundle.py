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
