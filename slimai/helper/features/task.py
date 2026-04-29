from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .extract import FeatureExtractor, infer_patch_features


def _get_accelerator() -> str:
  if torch.cuda.is_available():
    return "cuda"
  return "cpu"


def parse_devices(devices_str: Optional[str]) -> List[int]:
  accelerator = _get_accelerator()
  if accelerator == "cpu":
    return [0]

  if devices_str is None:
    devices_str = ""

  parts = [p.strip() for p in devices_str.split(",") if p.strip() != ""]
  if len(parts) == 0:
    return list(range(torch.cuda.device_count()))

  requested = []
  for part in parts:
    if not part.isdigit():
      raise ValueError(f"Invalid --devices value: {devices_str}")
    requested.append(int(part))

  visible_count = torch.cuda.device_count()
  if all(0 <= device_id < visible_count for device_id in requested):
    return requested

  # Support physical GPU ids when CUDA_VISIBLE_DEVICES is set, e.g. "6,7".
  # In this case torch uses logical ids [0, 1], so map physical -> logical.
  cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  cvd_parts = [p.strip() for p in cvd.split(",") if p.strip() != ""]
  if len(cvd_parts) > 0 and all(part.isdigit() for part in cvd_parts):
    physical_to_logical = {int(physical): logical for logical, physical in enumerate(cvd_parts)}
    if all(device_id in physical_to_logical for device_id in requested):
      return [physical_to_logical[device_id] for device_id in requested]

  raise ValueError(
    "Invalid --devices value '{}'. Visible logical devices are [0..{}]. "
    "If CUDA_VISIBLE_DEVICES is set (current='{}'), pass logical ids (e.g. 0,1) "
    "or matching physical ids from CUDA_VISIBLE_DEVICES.".format(
      devices_str,
      max(visible_count - 1, 0),
      cvd if cvd != "" else "<empty>",
    )
  )


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
