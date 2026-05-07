from __future__ import annotations

from dataclasses import dataclass
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue as queue_module
import tempfile
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from .extract import (
  MutablePatchDataset,
  build_feature_extractor,
  infer_patch_features,
  infer_patch_features_with_loader,
)


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
  encoder_name: str
  slide_encoder_name: Optional[str]
  device_id: int
  accelerator: str


@dataclass(frozen=True)
class InferenceOptions:
  read_scale: float
  patch_size: int
  batch_size: int
  num_workers: int
  to_gray: bool
  show_progress: bool = True


@dataclass(frozen=True)
class PersistentWorkerPoolConfig:
  encoder_name: str
  slide_encoder_name: Optional[str]
  accelerator: str
  devices: Tuple[int, ...]
  options: InferenceOptions


def build_chunk_tasks(
  *,
  wsi_file: str,
  encoder_name: str,
  slide_encoder_name: Optional[str],
  region_np: np.ndarray,
  devices: Sequence[int],
  accelerator: str,
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
        encoder_name=encoder_name,
        slide_encoder_name=slide_encoder_name,
        device_id=device_id,
        accelerator=accelerator,
      )
    )
  return tasks


def _persistent_worker_loop(
  *,
  device_id: int,
  config: PersistentWorkerPoolConfig,
  request_queue: mp.Queue,
  result_queue: mp.Queue,
  launch_time: float,
) -> None:
  startup_start = time.perf_counter()
  try:
    build_start = time.perf_counter()
    extractor = build_feature_extractor(
      config.encoder_name,
      config.slide_encoder_name,
      device_id=device_id,
      accelerator=config.accelerator,
    )
    build_feature_extractor_sec = max(time.perf_counter() - build_start, 0.0)

    manager = mp.Manager()
    shared_state = manager.dict(
      {
        "task_id": 0,
        "wsi_file": "",
        "coords_path": "",
        "scale": float(config.options.read_scale),
        "patch_size": int(config.options.patch_size),
        "to_gray": bool(config.options.to_gray),
      }
    )
    dataloader_setup_start = time.perf_counter()
    dataset = MutablePatchDataset(
      shared_state=shared_state,
      transform=extractor.transform,
    )
    loader_kwargs = dict(
      batch_size=config.options.batch_size,
      num_workers=config.options.num_workers,
      shuffle=False,
      drop_last=False,
    )
    if config.options.num_workers > 0:
      loader_kwargs.update(
        multiprocessing_context="spawn",
        persistent_workers=True,
        prefetch_factor=2,
      )
    loader = DataLoader(dataset, **loader_kwargs)
    dataloader_setup_sec = max(time.perf_counter() - dataloader_setup_start, 0.0)
    spawn_to_ready_sec = max(time.perf_counter() - launch_time, 0.0)
    result_queue.put(
      {
        "kind": "startup",
        "device_id": int(device_id),
        "pid": os.getpid(),
        "spawn_to_ready_sec": spawn_to_ready_sec,
        "build_feature_extractor_sec": build_feature_extractor_sec,
        "dataloader_setup_sec": dataloader_setup_sec,
        "worker_startup_total_sec": max(time.perf_counter() - startup_start, 0.0),
      }
    )

    while True:
      command = request_queue.get()
      kind = str(command.get("kind", ""))
      if kind == "stop":
        break
      if kind != "task":
        continue

      task_token = str(command["task_token"])
      chunk_index = int(command["chunk_index"])
      wsi_file = str(command["wsi_file"])
      coords_path = str(command["coords_path"])
      to_gray = bool(command["to_gray"])
      shared_state["task_id"] = int(shared_state.get("task_id", 0)) + 1
      shared_state["wsi_file"] = wsi_file
      shared_state["coords_path"] = coords_path
      shared_state["scale"] = float(config.options.read_scale)
      shared_state["patch_size"] = int(config.options.patch_size)
      shared_state["to_gray"] = bool(to_gray)

      worker_task_start = time.perf_counter()
      try:
        feature_np, infer_metrics = infer_patch_features_with_loader(
          extractor,
          loader=loader,
          wsi_file=wsi_file,
          show_progress=config.options.show_progress,
        )
        output_path = str(command["output_path"])
        save_start = time.perf_counter()
        np.save(output_path, feature_np.astype(np.float32))
        save_npy_sec = max(time.perf_counter() - save_start, 0.0)
        metrics: Dict[str, Any] = {
          "chunk_index": chunk_index,
          "device_id": int(device_id),
          "accelerator": config.accelerator,
          "encoder_name": config.encoder_name,
          "input_patch_count": int(command["patch_count"]),
          "spawn_to_start_sec": 0.0,
          "build_feature_extractor_sec": 0.0,
          "infer_patch_features_sec": max(time.perf_counter() - worker_task_start, 0.0),
          "save_npy_sec": save_npy_sec,
          "worker_total_sec": max(time.perf_counter() - worker_task_start, 0.0),
        }
        metrics.update(infer_metrics)
        result_queue.put(
          {
            "kind": "result",
            "task_token": task_token,
            "chunk_index": chunk_index,
            "output_path": output_path,
            "metrics": metrics,
          }
        )
      except BaseException as exc:
        result_queue.put(
          {
            "kind": "error",
            "task_token": task_token,
            "chunk_index": chunk_index,
            "device_id": int(device_id),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
          }
        )
  except BaseException as exc:
    result_queue.put(
      {
        "kind": "startup_error",
        "device_id": int(device_id),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
      }
    )
  return


class PersistentWorkerPool:
  def __init__(self, config: PersistentWorkerPoolConfig):
    self.config = config
    self.context = mp.get_context("spawn")
    self.request_queues: Dict[int, mp.Queue] = {}
    self.result_queue: mp.Queue = self.context.Queue()
    self.processes: Dict[int, mp.Process] = {}
    self.startup_metrics: Dict[int, Dict[str, Any]] = {}
    self._tmp_dir_obj = tempfile.TemporaryDirectory(prefix="slimai_persistent_worker_pool_")
    self._tmp_dir = Path(self._tmp_dir_obj.name)
    self._task_counter = 0
    self._started = False
    return

  def _start(self) -> None:
    if self._started:
      return
    for device_id in self.config.devices:
      device_queue: mp.Queue = self.context.Queue()
      launch_time = time.perf_counter()
      process = self.context.Process(
        target=_persistent_worker_loop,
        kwargs={
          "device_id": int(device_id),
          "config": self.config,
          "request_queue": device_queue,
          "result_queue": self.result_queue,
          "launch_time": launch_time,
        },
      )
      process.start()
      self.request_queues[int(device_id)] = device_queue
      self.processes[int(device_id)] = process

    pending_devices = set(int(device) for device in self.config.devices)
    timeout_sec = 300.0
    deadline = time.perf_counter() + timeout_sec
    while len(pending_devices) > 0:
      if time.perf_counter() > deadline:
        raise RuntimeError(f"Persistent worker startup timeout. pending_devices={sorted(pending_devices)}")
      try:
        message = self.result_queue.get(timeout=5.0)
      except queue_module.Empty:
        continue
      kind = str(message.get("kind", ""))
      if kind == "startup":
        device_id = int(message["device_id"])
        self.startup_metrics[device_id] = message
        pending_devices.discard(device_id)
      elif kind == "startup_error":
        raise RuntimeError(
          "Persistent worker startup failed device={} error_type={} error={}".format(
            message.get("device_id"),
            message.get("error_type"),
            message.get("error"),
          )
        )
    self._started = True
    return

  def close(self) -> None:
    for queue in self.request_queues.values():
      try:
        queue.put({"kind": "stop"})
      except Exception:
        pass
    for process in self.processes.values():
      process.join(timeout=10.0)
      if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
    self.request_queues.clear()
    self.processes.clear()
    if self._tmp_dir_obj is not None:
      self._tmp_dir_obj.cleanup()
    return

  def run_tasks(
    self,
    tasks: Sequence[ChunkTask],
    *,
    max_workers: int,
    options: InferenceOptions,
  ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    _ = options
    self._start()
    if len(tasks) == 0:
      return np.zeros((0, 0), dtype=np.float32), []
    if max_workers <= 0:
      max_workers = len(tasks)
    max_workers = max(1, min(int(max_workers), len(tasks)))

    indexed_results: List[Tuple[int, np.ndarray]] = []
    chunk_metrics: List[Dict[str, Any]] = []
    for start in range(0, len(tasks), max_workers):
      batch = list(tasks[start:start + max_workers])
      task_by_token: Dict[str, ChunkTask] = {}
      for task in batch:
        self._task_counter += 1
        task_token = f"task-{self._task_counter}-{task.chunk_index}"
        coords_path = self._tmp_dir / f"{task_token}.coords.npy"
        output_path = self._tmp_dir / f"{task_token}.out.npy"
        np.save(coords_path.as_posix(), task.region_chunk.astype(np.float32))
        self.request_queues[int(task.device_id)].put(
          {
            "kind": "task",
            "task_token": task_token,
            "chunk_index": int(task.chunk_index),
            "wsi_file": task.wsi_file,
            "coords_path": coords_path.as_posix(),
            "output_path": output_path.as_posix(),
            "patch_count": int(task.region_chunk.shape[0]),
            "to_gray": bool(options.to_gray),
          }
        )
        task_by_token[task_token] = task

      received = 0
      while received < len(batch):
        message = self.result_queue.get(timeout=3600.0)
        kind = str(message.get("kind", ""))
        if kind in ["startup", "startup_error"]:
          if kind == "startup":
            self.startup_metrics[int(message["device_id"])] = message
            continue
          raise RuntimeError(
            "Persistent worker startup failed device={} error_type={} error={}".format(
              message.get("device_id"),
              message.get("error_type"),
              message.get("error"),
            )
          )
        task_token = str(message.get("task_token", ""))
        if task_token not in task_by_token:
          continue
        task = task_by_token[task_token]
        if kind == "error":
          raise RuntimeError(
            "Persistent worker failed encoder={} device={}:{} chunk={} error_type={} error={}".format(
              task.encoder_name,
              task.accelerator,
              task.device_id,
              task.chunk_index,
              message.get("error_type"),
              message.get("error"),
            )
          )
        if kind != "result":
          continue
        output_path = Path(str(message["output_path"]))
        if not output_path.exists():
          raise RuntimeError(
            "Persistent worker produced no output encoder={} device={}:{} chunk={}".format(
              task.encoder_name,
              task.accelerator,
              task.device_id,
              task.chunk_index,
            )
          )
        indexed_results.append((task.chunk_index, np.load(output_path.as_posix())))
        chunk_metrics.append(dict(message.get("metrics", {})))
        try:
          output_path.unlink(missing_ok=True)
          coords_path = self._tmp_dir / f"{task_token}.coords.npy"
          coords_path.unlink(missing_ok=True)
        except Exception:
          pass
        received += 1

    indexed_results.sort(key=lambda item: item[0])
    features = [item[1] for item in indexed_results if item[1].size > 0]
    if len(features) == 0:
      return np.zeros((0, 0), dtype=np.float32), chunk_metrics
    return np.concatenate(features, axis=0).astype(np.float32), chunk_metrics


def create_persistent_worker_pool(
  *,
  encoder_name: str,
  slide_encoder_name: Optional[str],
  accelerator: str,
  devices: Sequence[int],
  options: InferenceOptions,
) -> PersistentWorkerPool:
  config = PersistentWorkerPoolConfig(
    encoder_name=encoder_name,
    slide_encoder_name=slide_encoder_name,
    accelerator=accelerator,
    devices=tuple(int(device) for device in devices),
    options=options,
  )
  return PersistentWorkerPool(config)


def _worker_run(task: ChunkTask, options: InferenceOptions) -> Tuple[np.ndarray, Dict[str, Any]]:
  build_start = time.perf_counter()
  extractor = build_feature_extractor(
    task.encoder_name,
    task.slide_encoder_name,
    device_id=task.device_id,
    accelerator=task.accelerator,
  )
  build_feature_extractor_sec = max(time.perf_counter() - build_start, 0.0)
  infer_start = time.perf_counter()
  feature_np, infer_metrics = infer_patch_features(
    extractor,
    wsi_file=task.wsi_file,
    region_np=task.region_chunk,
    read_scale=options.read_scale,
    patch_size=options.patch_size,
    batch_size=options.batch_size,
    num_workers=options.num_workers,
    to_gray=options.to_gray,
    show_progress=options.show_progress,
    return_metrics=True,
  )
  infer_patch_features_sec = max(time.perf_counter() - infer_start, 0.0)
  metrics: Dict[str, Any] = {
    "chunk_index": int(task.chunk_index),
    "device_id": int(task.device_id),
    "accelerator": task.accelerator,
    "encoder_name": task.encoder_name,
    "input_patch_count": int(task.region_chunk.shape[0]),
    "build_feature_extractor_sec": build_feature_extractor_sec,
    "infer_patch_features_sec": infer_patch_features_sec,
  }
  metrics.update(infer_metrics)
  return feature_np, metrics


def _worker_run_to_file(
  task: ChunkTask,
  options: InferenceOptions,
  launch_time: float,
  output_path: str,
  metrics_path: str,
  error_path: str,
) -> None:
  worker_start = time.perf_counter()
  spawn_to_start_sec = max(worker_start - launch_time, 0.0)
  try:
    logger.info(
      "Inference worker start encoder={} device={}:{} chunk={} patches={} pid={} spawn_to_start_sec={:.3f}",
      task.encoder_name,
      task.accelerator,
      task.device_id,
      task.chunk_index,
      int(task.region_chunk.shape[0]),
      os.getpid(),
      spawn_to_start_sec,
    )
    feature_np, metrics = _worker_run(task, options)
    save_start = time.perf_counter()
    np.save(output_path, feature_np.astype(np.float32))
    save_npy_sec = max(time.perf_counter() - save_start, 0.0)
    metrics["save_npy_sec"] = save_npy_sec
    metrics["spawn_to_start_sec"] = spawn_to_start_sec
    metrics["worker_total_sec"] = max(time.perf_counter() - worker_start, 0.0)
    Path(metrics_path).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
      "Inference worker done encoder={} device={}:{} chunk={} features={} pid={} worker_total_sec={:.3f} throughput={:.2f}patch/s",
      task.encoder_name,
      task.accelerator,
      task.device_id,
      task.chunk_index,
      tuple(feature_np.shape),
      os.getpid(),
      float(metrics.get("worker_total_sec", 0.0)),
      float(metrics.get("throughput_patch_per_sec", 0.0)),
    )
  except BaseException as exc:
    payload = {
      "error_type": type(exc).__name__,
      "error": str(exc),
      "traceback": traceback.format_exc(),
      "encoder_name": task.encoder_name,
      "accelerator": task.accelerator,
      "device_id": task.device_id,
      "chunk_index": task.chunk_index,
      "wsi_file": task.wsi_file,
      "pid": os.getpid(),
      "spawn_to_start_sec": spawn_to_start_sec,
    }
    Path(error_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    raise
  return


def _raise_worker_error(task: ChunkTask, process: mp.Process, error_path: Path) -> None:
  if error_path.exists():
    try:
      payload = json.loads(error_path.read_text(encoding="utf-8"))
      detail = f"{payload.get('error_type')}: {payload.get('error')}"
    except Exception:
      detail = error_path.read_text(encoding="utf-8", errors="replace")
  else:
    detail = "no error payload written"
  raise RuntimeError(
    "Inference worker failed encoder={} device={}:{} chunk={} pid={} exitcode={} detail={}".format(
      task.encoder_name,
      task.accelerator,
      task.device_id,
      task.chunk_index,
      process.pid,
      process.exitcode,
      detail,
    )
  )


def _run_chunk_tasks_ephemeral(
  *,
  tasks: Sequence[ChunkTask],
  options: InferenceOptions,
  max_workers: Optional[int] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
  if len(tasks) == 0:
    return np.zeros((0, 0), dtype=np.float32), []

  if max_workers is None or max_workers <= 0:
    max_workers = len(tasks)
  max_workers = max(1, min(int(max_workers), len(tasks)))

  context = mp.get_context("spawn")
  indexed_results = []
  chunk_metrics: List[Dict[str, Any]] = []
  with tempfile.TemporaryDirectory(prefix="slimai_feature_chunks_") as tmp_dir:
    tmp_path = Path(tmp_dir)
    for start in range(0, len(tasks), max_workers):
      active = []
      for task in tasks[start:start + max_workers]:
        output_path = tmp_path / f"chunk_{task.chunk_index}.npy"
        metrics_path = tmp_path / f"chunk_{task.chunk_index}.metrics.json"
        error_path = tmp_path / f"chunk_{task.chunk_index}.error.json"
        launch_time = time.perf_counter()
        process = context.Process(
          target=_worker_run_to_file,
          args=(
            task,
            options,
            launch_time,
            output_path.as_posix(),
            metrics_path.as_posix(),
            error_path.as_posix(),
          ),
        )
        process.start()
        active.append((task, process, output_path, metrics_path, error_path))

      for task, process, output_path, metrics_path, error_path in active:
        process.join()
        if process.exitcode != 0:
          _raise_worker_error(task, process, error_path)
        if not output_path.exists():
          raise RuntimeError(
            "Inference worker produced no output encoder={} device={}:{} chunk={} pid={}".format(
              task.encoder_name,
              task.accelerator,
              task.device_id,
              task.chunk_index,
              process.pid,
            )
          )
        indexed_results.append((task.chunk_index, np.load(output_path.as_posix())))
        if metrics_path.exists():
          payload = json.loads(metrics_path.read_text(encoding="utf-8"))
          chunk_metrics.append(payload)

  indexed_results.sort(key=lambda item: item[0])
  features = [item[1] for item in indexed_results if item[1].size > 0]
  if len(features) == 0:
    return np.zeros((0, 0), dtype=np.float32), chunk_metrics
  return np.concatenate(features, axis=0).astype(np.float32), chunk_metrics


def run_chunk_tasks(
  *,
  tasks: Sequence[ChunkTask],
  options: InferenceOptions,
  max_workers: Optional[int] = None,
  persistent_pool: Optional[PersistentWorkerPool] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
  if persistent_pool is not None:
    max_workers_value = len(tasks) if max_workers is None else int(max_workers)
    return persistent_pool.run_tasks(tasks, max_workers=max_workers_value, options=options)
  return _run_chunk_tasks_ephemeral(tasks=tasks, options=options, max_workers=max_workers)
