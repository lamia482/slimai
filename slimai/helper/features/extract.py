from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import timm
import torch
from loguru import logger
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from tqdm import tqdm

from .pipeline import PatchDataset


def _get_accelerator() -> str:
  if torch.cuda.is_available():
    return "cuda"
  return "cpu"


@dataclass(frozen=True)
class FeatureExtractor:
  name: str
  patch_encoder: torch.nn.Module
  transform: Callable
  feature_dim: int
  device: torch.device
  slide_encoder: Optional[torch.nn.Module] = None


def _build_uni_encoder(cache_dir: str) -> Tuple[torch.nn.Module, Callable, int]:
  model = timm.create_model(  # type: ignore
    "hf_hub:MahmoodLab/UNI",
    pretrained=True,
    init_values=1e-5,
    dynamic_img_size=True,
    cache_dir=cache_dir,
  )
  transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
  model.eval()
  return model, transform, int(model.num_features) # type: ignore


def _build_uni2_encoder(cache_dir: str) -> Tuple[torch.nn.Module, Callable, int]:
  model = timm.create_model(  # type: ignore
    "hf_hub:MahmoodLab/UNI2-h",
    pretrained=True,
    img_size=224,
    patch_size=14,
    depth=24,
    num_heads=24,
    init_values=1e-5,
    embed_dim=1536,
    mlp_ratio=2.66667 * 2,
    num_classes=0,
    no_embed_class=True,
    mlp_layer=timm.layers.SwiGLUPacked,  # type: ignore
    act_layer=torch.nn.SiLU,
    reg_tokens=8,
    dynamic_img_size=True,
    cache_dir=cache_dir,
  )
  transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
  model.eval()
  return model, transform, int(model.num_features) # type: ignore


def _build_not_implemented_encoder(name: str, _: str) -> Tuple[torch.nn.Module, Callable, int]:
  raise NotImplementedError(
    f"Encoder '{name}' is reserved but not implemented yet. "
    "Please register its builder in slimai.helper.features.extract."
  )


PATCH_ENCODER_BUILDERS: Dict[str, Callable[[str], Tuple[torch.nn.Module, Callable, int]]] = {
  "UNI": _build_uni_encoder,
  "UNI2": _build_uni2_encoder,
  "CONCH": lambda cache_dir: _build_not_implemented_encoder("CONCH", cache_dir),
  "CONCHV1_5": lambda cache_dir: _build_not_implemented_encoder("CONCHV1_5", cache_dir),
}


def register_patch_encoder(
  name: str,
  builder: Callable[[str], Tuple[torch.nn.Module, Callable, int]],
) -> None:
  PATCH_ENCODER_BUILDERS[name] = builder
  return


def build_feature_extractor(
  patch_encoder_name: str,
  slide_encoder_name: Optional[str],
  *,
  device_id: int,
  cache_dir: str = "/.slimai/cache/huggingface/hub",
) -> FeatureExtractor:
  if patch_encoder_name not in PATCH_ENCODER_BUILDERS:
    available = sorted(PATCH_ENCODER_BUILDERS.keys())
    raise ValueError(f"Unsupported patch encoder: {patch_encoder_name}. Available: {available}")

  model, transform, feature_dim = PATCH_ENCODER_BUILDERS[patch_encoder_name](cache_dir)
  accelerator = _get_accelerator()
  if accelerator == "cpu":
    device = torch.device("cpu")
  else:
    device = torch.device(f"{accelerator}:{device_id}")

  patch_encoder = model.to(device).eval()
  slide_encoder = None

  if slide_encoder_name is not None:
    logger.warning(
      "slide encoder '{}' is not implemented yet. "
      "The current run will use patch-level features only.",
      slide_encoder_name,
    )

  return FeatureExtractor(
    name=patch_encoder_name,
    patch_encoder=patch_encoder,
    transform=transform,
    feature_dim=feature_dim,
    device=device,
    slide_encoder=slide_encoder,
  )


def infer_patch_features(
  extractor: FeatureExtractor,
  *,
  wsi_file: str,
  region_np: np.ndarray,
  read_scale: float,
  patch_size: int,
  batch_size: int,
  num_workers: int,
  to_gray: bool,
  show_progress: bool = True,
) -> np.ndarray:
  if region_np.size == 0:
    return np.zeros((0, extractor.feature_dim), dtype=np.float32)

  dataset = PatchDataset(
    wsi_file=wsi_file,
    coords=region_np,
    scale=read_scale,
    patch_size=patch_size,
    transform=extractor.transform,
    to_gray=to_gray,
  )
  loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
  )

  outputs = []
  desc = f"Infer<{Path(wsi_file).stem}>[{extractor.name}] on {extractor.device}"
  for batch in tqdm(loader, desc=desc, leave=False, disable=not show_progress):
    batch = batch.to(extractor.device)
    with torch.inference_mode():
      output = extractor.patch_encoder(batch).cpu()
    outputs.append(output)

  if len(outputs) == 0:
    return np.zeros((0, extractor.feature_dim), dtype=np.float32)

  if extractor.slide_encoder is not None:
    with torch.inference_mode():
      slide_output = extractor.slide_encoder(torch.cat(outputs, dim=0).to(extractor.device)).cpu()
    outputs.append(slide_output)

  return torch.cat(outputs, dim=0).numpy().astype(np.float32)
