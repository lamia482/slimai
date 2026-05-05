from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from loguru import logger

try:
  from .extract import FeatureExtractor, infer_patch_features
except Exception:
  from extract import FeatureExtractor, infer_patch_features  # type: ignore


def _is_npu_available() -> bool:
  try:
    import torch_npu # type: ignore # noqa: F401
  except Exception:
    return False
  if not hasattr(torch, "npu"):
    return False
  try:
    return bool(torch.npu.is_available()) # type: ignore[attr-defined]
  except Exception:
    return False


def _npu_count() -> int:
  if not _is_npu_available():
    return 0
  try:
    return int(torch.npu.device_count()) # type: ignore[attr-defined]
  except Exception:
    return 0


def get_available_devices() -> Dict[str, List[int]]:
  available: Dict[str, List[int]] = {"cpu": [0]}
  if _is_npu_available():
    available["npu"] = list(range(_npu_count()))
  if torch.cuda.is_available():
    available["cuda"] = list(range(torch.cuda.device_count()))
  return available


def resolve_accelerator(accelerator: str) -> Tuple[str, Dict[str, List[int]]]:
  requested = accelerator.strip().lower()
  available = get_available_devices()
  if requested not in ["auto", "npu", "cuda", "cpu"]:
    raise ValueError(f"Unsupported --accelerator value: {accelerator}")
  if requested == "auto":
    if len(available.get("npu", [])) > 0:
      return "npu", available
    if len(available.get("cuda", [])) > 0:
      return "cuda", available
    return "cpu", available
  if requested == "cpu":
    return "cpu", available
  if len(available.get(requested, [])) == 0:
    raise ValueError(f"Accelerator '{requested}' is unavailable. detected={available}")
  return requested, available


def _parse_devices_str(devices_str: Optional[str]) -> List[int]:
  if devices_str is None:
    return []
  parts = [part.strip() for part in devices_str.split(",") if part.strip() != ""]
  if len(parts) == 0:
    return []
  parsed: List[int] = []
  for part in parts:
    if not part.isdigit():
      raise ValueError(f"Invalid --devices value: {devices_str}")
    parsed.append(int(part))
  return parsed


def _map_visible_devices(requested: List[int], env_name: str) -> Optional[List[int]]:
  raw = os.environ.get(env_name, "")
  parts = [part.strip() for part in raw.split(",") if part.strip() != ""]
  if len(parts) == 0:
    return None
  if not all(part.isdigit() for part in parts):
    return None
  physical_to_logical = {int(physical): logical for logical, physical in enumerate(parts)}
  if not all(device_id in physical_to_logical for device_id in requested):
    return None
  return [physical_to_logical[device_id] for device_id in requested]


def parse_devices(devices_str: Optional[str], accelerator: str) -> List[int]:
  if accelerator == "cpu":
    return [0]

  requested = _parse_devices_str(devices_str)
  if accelerator == "cuda":
    visible_count = torch.cuda.device_count()
    if len(requested) == 0:
      return list(range(visible_count))
    if all(0 <= device_id < visible_count for device_id in requested):
      return requested
    mapped = _map_visible_devices(requested, "CUDA_VISIBLE_DEVICES")
    if mapped is not None:
      return mapped
    raise ValueError(
      "Invalid --devices '{}' for cuda. visible logical range=[0..{}], CUDA_VISIBLE_DEVICES='{}'".format(
        devices_str,
        max(visible_count - 1, 0),
        os.environ.get("CUDA_VISIBLE_DEVICES", "<empty>"),
      )
    )

  if accelerator == "npu":
    visible_count = _npu_count()
    if len(requested) == 0:
      return list(range(visible_count))
    if all(0 <= device_id < visible_count for device_id in requested):
      return requested
    mapped = _map_visible_devices(requested, "ASCEND_VISIBLE_DEVICES")
    if mapped is not None:
      return mapped
    raise ValueError(
      "Invalid --devices '{}' for npu. visible logical range=[0..{}], ASCEND_VISIBLE_DEVICES='{}'".format(
        devices_str,
        max(visible_count - 1, 0),
        os.environ.get("ASCEND_VISIBLE_DEVICES", "<empty>"),
      )
    )

  raise ValueError(f"Unsupported resolved accelerator: {accelerator}")


def log_device_resolution(
  *,
  requested_accelerator: str,
  resolved_accelerator: str,
  available_devices: Dict[str, List[int]],
  requested_devices: Optional[str],
  resolved_devices: Sequence[int],
) -> None:
  logger.info(
    "Accelerator requested='{}' resolved='{}' available={} requested_devices='{}' resolved_devices={}",
    requested_accelerator,
    resolved_accelerator,
    available_devices,
    requested_devices if requested_devices is not None else "",
    list(resolved_devices),
  )
  return


@dataclass(frozen=True)
class ChunkTask:
  wsi_file: str
  chunk_index: int
  region_chunk: np.ndarray
  extractor: FeatureExtractor


@dataclass(frozen=True)
class InferenceOptions:
  read_scale: float
  patch_size: int
  batch_size: int
  num_workers: int
  to_gray: bool
  show_progress: bool = True


def build_chunk_tasks(
  *,
  wsi_file: str,
  region_np: np.ndarray,
  devices: Sequence[int],
  model_engines: Dict[int, FeatureExtractor],
) -> List[ChunkTask]:
  if len(devices) == 0:
    raise ValueError("No devices provided.")

  chunks = [chunk for chunk in np.array_split(region_np, len(devices)) if len(chunk) > 0]
  tasks: List[ChunkTask] = []
  for chunk_index, region_chunk in enumerate(chunks):
    device_id = devices[chunk_index % len(devices)]
    tasks.append(
      ChunkTask(
        wsi_file=wsi_file,
        chunk_index=chunk_index,
        region_chunk=region_chunk,
        extractor=model_engines[device_id],
      )
    )
  return tasks


def _worker_run(task: ChunkTask, options: InferenceOptions) -> np.ndarray:
  return infer_patch_features(
    task.extractor,
    wsi_file=task.wsi_file,
    region_np=task.region_chunk,
    read_scale=options.read_scale,
    patch_size=options.patch_size,
    batch_size=options.batch_size,
    num_workers=options.num_workers,
    to_gray=options.to_gray,
    show_progress=options.show_progress,
  )


def run_chunk_tasks(
  *,
  tasks: Sequence[ChunkTask],
  options: InferenceOptions,
  max_workers: Optional[int] = None,
) -> np.ndarray:
  if len(tasks) == 0:
    return np.zeros((0, 0), dtype=np.float32)

  if max_workers is None:
    max_workers = len(tasks)

  futures = []
  with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    for task in tasks:
      futures.append(executor.submit(_worker_run, task, options))

  indexed_results = []
  for task, future in zip(tasks, futures):
    indexed_results.append((task.chunk_index, future.result()))

  indexed_results.sort(key=lambda item: item[0])
  features = [item[1] for item in indexed_results if item[1].size > 0]
  if len(features) == 0:
    return np.zeros((0, tasks[0].extractor.feature_dim), dtype=np.float32)
  return np.concatenate(features, axis=0).astype(np.float32)
