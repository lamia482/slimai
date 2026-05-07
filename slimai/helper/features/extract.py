from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import timm
import torch
from loguru import logger
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from tqdm import tqdm
from .pipeline import PatchDataset


def _ensure_npu_runtime() -> None:
  try:
    import torch_npu # type: ignore # noqa: F401
  except Exception as exc:
    raise RuntimeError("NPU requested but torch_npu is unavailable.") from exc
  if not hasattr(torch, "npu"):
    raise RuntimeError("NPU requested but torch.npu is unavailable.")
  return


@dataclass(frozen=True)
class FeatureExtractor:
  name: str
  patch_encoder: torch.nn.Module
  transform: Callable
  feature_dim: int
  device: torch.device
  slide_encoder: Optional[torch.nn.Module] = None


def _to_bool(value: object, default: bool = False) -> bool:
  if isinstance(value, bool):
    return value
  if isinstance(value, (int, np.integer)):
    return bool(int(value))
  if isinstance(value, str):
    lowered = value.strip().lower()
    if lowered in ["1", "true", "yes", "y", "on"]:
      return True
    if lowered in ["0", "false", "no", "n", "off"]:
      return False
  return default


class MutablePatchDataset(PatchDataset):
  def __init__(
    self,
    *,
    shared_state: Any,
    transform: Callable,
  ):
    super().__init__(
      wsi_file="",
      coords=np.zeros((0, 5), dtype=np.float32),
      scale=1.0,
      patch_size=224,
      transform=transform,
      to_gray=False,
    )
    self._shared_state = shared_state
    self._active_task_id = -1
    self._coords_path = ""
    return

  def _refresh_from_shared_state(self) -> None:
    task_id = int(self._shared_state.get("task_id", -1))
    if task_id == self._active_task_id:
      return
    coords_path = str(self._shared_state.get("coords_path", ""))
    if coords_path == "":
      self.coords = np.zeros((0, 5), dtype=np.float32)
    else:
      self.coords = np.load(coords_path)
    self.wsi_file = str(self._shared_state.get("wsi_file", ""))
    self.scale = float(self._shared_state.get("scale", 1.0))
    self.patch_size = int(self._shared_state.get("patch_size", 224))
    self.to_gray = _to_bool(self._shared_state.get("to_gray", False), default=False)
    self.reader = None
    self._coords_path = coords_path
    self._active_task_id = task_id
    return

  def __len__(self) -> int:
    self._refresh_from_shared_state()
    return int(self.coords.shape[0])

  def __getitem__(self, index: int):
    self._refresh_from_shared_state()
    return super().__getitem__(index)


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
  accelerator: str = "cpu",
  cache_dir: str = "/.slimai/cache/huggingface/hub",
) -> FeatureExtractor:
  if patch_encoder_name not in PATCH_ENCODER_BUILDERS:
    available = sorted(PATCH_ENCODER_BUILDERS.keys())
    raise ValueError(f"Unsupported patch encoder: {patch_encoder_name}. Available: {available}")

  model, transform, feature_dim = PATCH_ENCODER_BUILDERS[patch_encoder_name](cache_dir)
  accelerator_name = accelerator.strip().lower()
  if accelerator_name == "cpu":
    device = torch.device("cpu")
  elif accelerator_name == "npu":
    _ensure_npu_runtime()
    device = torch.device(f"npu:{device_id}")
  elif accelerator_name == "cuda":
    if not torch.cuda.is_available():
      raise RuntimeError("CUDA requested but torch.cuda is unavailable.")
    device = torch.device(f"cuda:{device_id}")
  else:
    raise ValueError(f"Unsupported accelerator: {accelerator}")

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
  return_metrics: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]] | np.ndarray:
  start_total = time.perf_counter()
  if region_np.size == 0:
    empty = np.zeros((0, extractor.feature_dim), dtype=np.float32)
    metrics = {
      "patch_count": 0.0,
      "batch_count": 0.0,
      "dataloader_setup_sec": 0.0,
      "first_batch_latency_sec": 0.0,
      "infer_loop_sec": 0.0,
      "avg_batch_sec": 0.0,
      "throughput_patch_per_sec": 0.0,
      "total_sec": max(time.perf_counter() - start_total, 0.0),
    }
    return (empty, metrics) if return_metrics else empty

  dataset = PatchDataset(
    wsi_file=wsi_file,
    coords=region_np,
    scale=read_scale,
    patch_size=patch_size,
    transform=extractor.transform,
    to_gray=to_gray,
  )
  loader_setup_start = time.perf_counter()
  loader_kwargs = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
  )
  if num_workers > 0:
    loader_kwargs.update(
      multiprocessing_context="spawn",
      persistent_workers=True,
      prefetch_factor=2,
    )
  loader = DataLoader(dataset, **loader_kwargs)
  dataloader_setup_sec = max(time.perf_counter() - loader_setup_start, 0.0)

  outputs = []
  desc = f"Infer<{Path(wsi_file).stem}>[{extractor.name}] on {extractor.device}"
  infer_loop_start = time.perf_counter()
  first_batch_latency_sec: Optional[float] = None
  infer_batch_total_sec = 0.0
  batch_count = 0
  patch_count = 0
  for batch in tqdm(loader, desc=desc, leave=False, disable=not show_progress):
    batch_start = time.perf_counter()
    if first_batch_latency_sec is None:
      first_batch_latency_sec = max(batch_start - infer_loop_start, 0.0)
    batch = batch.to(extractor.device)
    with torch.inference_mode():
      output = extractor.patch_encoder(batch).cpu()
    outputs.append(output)
    infer_batch_total_sec += max(time.perf_counter() - batch_start, 0.0)
    batch_count += 1
    patch_count += int(batch.shape[0])
  infer_loop_sec = max(time.perf_counter() - infer_loop_start, 0.0)

  if len(outputs) == 0:
    feature_np = np.zeros((0, extractor.feature_dim), dtype=np.float32)
  else:
    if extractor.slide_encoder is not None:
      with torch.inference_mode():
        slide_output = extractor.slide_encoder(torch.cat(outputs, dim=0).to(extractor.device)).cpu()
      outputs.append(slide_output)

    feature_np = torch.cat(outputs, dim=0).numpy().astype(np.float32)

  metrics = {
    "patch_count": float(patch_count),
    "batch_count": float(batch_count),
    "dataloader_setup_sec": dataloader_setup_sec,
    "first_batch_latency_sec": max(first_batch_latency_sec or 0.0, 0.0),
    "infer_loop_sec": infer_loop_sec,
    "avg_batch_sec": (infer_batch_total_sec / batch_count) if batch_count > 0 else 0.0,
    "throughput_patch_per_sec": (patch_count / infer_loop_sec) if infer_loop_sec > 1e-9 else 0.0,
    "total_sec": max(time.perf_counter() - start_total, 0.0),
  }
  return (feature_np, metrics) if return_metrics else feature_np


def infer_patch_features_with_loader(
  extractor: FeatureExtractor,
  *,
  loader: DataLoader,
  wsi_file: str,
  show_progress: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
  start_total = time.perf_counter()
  outputs = []
  desc = f"Infer<{Path(wsi_file).stem}>[{extractor.name}] on {extractor.device}"
  infer_loop_start = time.perf_counter()
  first_batch_latency_sec: Optional[float] = None
  infer_batch_total_sec = 0.0
  batch_count = 0
  patch_count = 0
  for batch in tqdm(loader, desc=desc, leave=False, disable=not show_progress):
    batch_start = time.perf_counter()
    if first_batch_latency_sec is None:
      first_batch_latency_sec = max(batch_start - infer_loop_start, 0.0)
    batch = batch.to(extractor.device)
    with torch.inference_mode():
      output = extractor.patch_encoder(batch).cpu()
    outputs.append(output)
    infer_batch_total_sec += max(time.perf_counter() - batch_start, 0.0)
    batch_count += 1
    patch_count += int(batch.shape[0])
  infer_loop_sec = max(time.perf_counter() - infer_loop_start, 0.0)

  if len(outputs) == 0:
    feature_np = np.zeros((0, extractor.feature_dim), dtype=np.float32)
  else:
    feature_np = torch.cat(outputs, dim=0).numpy().astype(np.float32)

  metrics = {
    "patch_count": float(patch_count),
    "batch_count": float(batch_count),
    "dataloader_setup_sec": 0.0,
    "first_batch_latency_sec": max(first_batch_latency_sec or 0.0, 0.0),
    "infer_loop_sec": infer_loop_sec,
    "avg_batch_sec": (infer_batch_total_sec / batch_count) if batch_count > 0 else 0.0,
    "throughput_patch_per_sec": (patch_count / infer_loop_sec) if infer_loop_sec > 1e-9 else 0.0,
    "total_sec": max(time.perf_counter() - start_total, 0.0),
  }
  return feature_np, metrics
