from __future__ import annotations

import hashlib
import importlib
import json
import os
import os.path as osp
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import signal
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from sdk.reader import get_reader_by_ext

try:
  from .extract import FeatureExtractor, build_feature_extractor
  from .pipeline import get_tissue_region
  from .task import (
    InferenceOptions,
    build_chunk_tasks,
    log_device_resolution,
    parse_devices,
    resolve_accelerator,
    run_chunk_tasks,
  )
except Exception:
  from extract import FeatureExtractor, build_feature_extractor  # type: ignore
  from pipeline import get_tissue_region  # type: ignore
  from task import (  # type: ignore
    InferenceOptions,
    build_chunk_tasks,
    log_device_resolution,
    parse_devices,
    resolve_accelerator,
    run_chunk_tasks,
  )


@dataclass(frozen=True)
class CreateFeatureConfig:
  input_file: str
  wsi_col: str = "wsi_file"
  patch_encoder: Optional[str] = None
  patch_encoders: Optional[str] = None
  slide_encoder: Optional[str] = None
  tag: str = "default"
  out_dir: str = "."
  devices: Optional[str] = None
  accelerator: str = "auto"
  max_futs: int = 0
  batch_size: int = 4
  num_workers: int = 2
  patch_size: int = 224
  stride_size: int = 192
  read_scale: float = 20
  operate_scale: float = 1.25
  to_gray: bool = False
  skip_existing: bool = True
  min_tissue_ratio: float = 0.05
  tissue_shrink: str = "tissue"
  output: Optional[str] = None
  preflight_only: bool = False
  manifest_only: bool = False
  limit: int = 0


@dataclass(frozen=True)
class ManifestItem:
  wsi_file: str
  relative_path: str
  expected_h5_path: str
  suffix: str
  size: int
  mtime_ns: int


@dataclass
class RunRuntime:
  requested_accelerator: str
  resolved_accelerator: str
  available_devices: Dict[str, List[int]]
  resolved_devices: List[int]
  input_mode: str
  stop_requested: bool = False


def _now_iso() -> str:
  return datetime.now(timezone.utc).isoformat()


def _extract_runtime_params(config: CreateFeatureConfig, encoder_list: Sequence[str]) -> Dict[str, object]:
  return {
    "encoder_list": list(encoder_list),
    "embedding_tag": config.tag,
    "read_scale": float(config.read_scale),
    "operate_scale": float(config.operate_scale),
    "patch_size": int(config.patch_size),
    "stride_size": int(config.stride_size),
    "to_gray": bool(config.to_gray),
    "min_tissue_ratio": float(config.min_tissue_ratio),
    "tissue_shrink": str(config.tissue_shrink),
  }


def _short_traceback(exc: BaseException) -> str:
  return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def _append_run_log(run_log: Path, message: str) -> None:
  run_log.parent.mkdir(parents=True, exist_ok=True)
  with open(run_log, "a", encoding="utf-8") as fp:
    fp.write(f"{_now_iso()} {message}\n")
  return


def _write_json(path: Path, payload: Dict[str, object]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w", encoding="utf-8") as fp:
    json.dump(payload, fp, ensure_ascii=False, indent=2)
  return


def _append_jsonl(path: Path, payload: Dict[str, object]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "a", encoding="utf-8") as fp:
    fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
  return


def _is_supported_wsi(path: Path) -> bool:
  suffix = path.suffix.lower()
  if suffix == "":
    return False
  try:
    get_reader_by_ext(suffix, strict=True)
    return True
  except Exception:
    return False


def _parse_encoder_list(config: CreateFeatureConfig) -> List[str]:
  if config.patch_encoders is not None and config.patch_encoders.strip() != "":
    encoders = [item.strip() for item in config.patch_encoders.split(",") if item.strip() != ""]
  elif config.patch_encoder is not None and config.patch_encoder.strip() != "":
    encoders = [config.patch_encoder.strip()]
  else:
    raise ValueError("Either --patch-encoder or --patch-encoders is required.")
  deduped: List[str] = []
  seen = set()
  for encoder in encoders:
    if encoder in seen:
      continue
    seen.add(encoder)
    deduped.append(encoder)
  return deduped


def _resolve_output_h5(
  *,
  out_dir: Path,
  wsi_file: str,
  relative_path: str,
  embedding_tag: str,
  preserve_structure: bool,
) -> str:
  file_name = f"{Path(wsi_file).stem}.wsi_feat_{embedding_tag}.h5"
  if not preserve_structure:
    return (out_dir / file_name).as_posix()
  rel_parent = Path(relative_path).parent
  return (out_dir / "embeddings" / rel_parent / file_name).as_posix()


def _read_input_pairs(config: CreateFeatureConfig) -> Tuple[List[Tuple[str, str]], str, bool]:
  input_path = Path(config.input_file)
  if input_path.is_dir():
    pairs: List[Tuple[str, str]] = []
    for path in sorted(input_path.rglob("*")):
      if not path.is_file():
        continue
      if not _is_supported_wsi(path):
        continue
      pairs.append((path.as_posix(), path.relative_to(input_path).as_posix()))
    return pairs, "dir", True

  if _is_supported_wsi(input_path):
    return [(input_path.as_posix(), input_path.name)], "single", False

  suffix = input_path.suffix.lower()
  if suffix not in [".xlsx", ".xls"]:
    raise ValueError(f"Unsupported --input-file suffix: {suffix}")
  *sheet_name, wsi_col_name = config.wsi_col.rsplit(":", 1)
  sheet_name_value = None if len(sheet_name) == 0 else sheet_name[0]
  df = pd.read_excel(input_path, sheet_name=sheet_name_value)
  if wsi_col_name not in df.columns:
    raise ValueError(f"Column not found in input file: {wsi_col_name}. available={list(df.columns)}")
  base_dir = input_path.parent
  pairs = []
  for value in df[wsi_col_name].tolist():
    if value is None:
      continue
    item = str(value).strip()
    if item == "" or item.lower() == "nan":
      continue
    raw_path = Path(item)
    resolved = raw_path if raw_path.is_absolute() else (base_dir / raw_path).resolve()
    rel = raw_path.as_posix() if not raw_path.is_absolute() else resolved.name
    pairs.append((resolved.as_posix(), rel))
  return pairs, "excel", False


def _file_md5(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
  digest = hashlib.md5()
  with open(path, "rb") as fp:
    while True:
      chunk = fp.read(chunk_size)
      if not chunk:
        break
      digest.update(chunk)
  return digest.hexdigest()


def _build_manifest(
  pairs: Sequence[Tuple[str, str]],
  *,
  out_dir: Path,
  embedding_tag: str,
  preserve_structure: bool,
) -> List[ManifestItem]:
  manifest: List[ManifestItem] = []
  for wsi_file, relative_path in pairs:
    path = Path(wsi_file)
    size = -1
    mtime_ns = -1
    if path.exists():
      stat_obj = path.stat()
      size = int(stat_obj.st_size)
      mtime_ns = int(stat_obj.st_mtime_ns)
    expected_h5_path = _resolve_output_h5(
      out_dir=out_dir,
      wsi_file=wsi_file,
      relative_path=relative_path,
      embedding_tag=embedding_tag,
      preserve_structure=preserve_structure,
    )
    manifest.append(
      ManifestItem(
        wsi_file=wsi_file,
        relative_path=relative_path,
        expected_h5_path=expected_h5_path,
        suffix=path.suffix.lower(),
        size=size,
        mtime_ns=mtime_ns,
      )
    )
  return manifest


def _check_manifest_collisions(manifest: Sequence[ManifestItem]) -> None:
  reverse_map: Dict[str, List[str]] = {}
  for item in manifest:
    reverse_map.setdefault(item.expected_h5_path, []).append(item.wsi_file)
  collisions = {k: v for k, v in reverse_map.items() if len(v) > 1}
  if len(collisions) > 0:
    sample_path = next(iter(collisions.keys()))
    raise ValueError(f"Manifest output collision detected: {sample_path} <- {collisions[sample_path]}")
  return


def _write_manifest(path: Path, manifest: Sequence[ManifestItem]) -> None:
  if path.exists():
    path.unlink()
  for item in manifest:
    _append_jsonl(path, asdict(item))
  return


def _append_record_to_xlsx(
  output_file: str,
  *,
  wsi_file: str,
  wsi_md5: str,
  h5_path: str,
  patch_num: int,
) -> None:
  output_path = Path(output_file)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  columns = ["wsi_file", "wsi_md5", "h5_path", "patch_num"]
  if output_path.exists():
    df = pd.read_excel(output_path)
    for col in columns:
      if col not in df.columns:
        df[col] = ""
    df = df[columns]
  else:
    df = pd.DataFrame(columns=columns)  # type: ignore
  df.loc[len(df)] = [wsi_file, wsi_md5, h5_path, int(patch_num)]
  df.to_excel(output_path, index=False)
  return


def _normalize_encoder_list(value: object) -> List[str]:
  if isinstance(value, np.ndarray):
    return [str(x) for x in value.tolist()]
  if isinstance(value, (list, tuple)):
    return [str(x) for x in value]
  if value is None:
    return []
  return [str(value)]


def _validate_h5_cache(
  path: Path,
  *,
  runtime_params: Dict[str, object],
  wsi_file: str,
  wsi_md5: str,
) -> bool:
  if not path.exists():
    return False
  try:
    with h5py.File(path.as_posix(), "r") as fp:
      attrs = fp.attrs
      if str(attrs.get("schema_version", "")) != "slimai_embedding_v1":
        return False
      if str(attrs.get("wsi_file", "")) != wsi_file:
        return False
      if str(attrs.get("wsi_md5", "")) != wsi_md5:
        return False
      if int(attrs.get("patch_size", -1)) != int(runtime_params["patch_size"]):
        return False
      if int(attrs.get("stride_size", -1)) != int(runtime_params["stride_size"]):
        return False
      if float(attrs.get("read_scale", -1.0)) != float(runtime_params["read_scale"]):
        return False
      if float(attrs.get("operate_scale", -1.0)) != float(runtime_params["operate_scale"]):
        return False
      if bool(attrs.get("to_gray", False)) != bool(runtime_params["to_gray"]):
        return False
      if float(attrs.get("min_tissue_ratio", -1.0)) != float(runtime_params["min_tissue_ratio"]):
        return False
      if str(attrs.get("tissue_shrink", "")) != str(runtime_params["tissue_shrink"]):
        return False
      if str(attrs.get("embedding_tag", "")) != str(runtime_params["embedding_tag"]):
        return False
      cached_encoders = sorted(_normalize_encoder_list(attrs.get("encoder_list")))
      expected_encoders = sorted([str(x) for x in runtime_params["encoder_list"]])  # type: ignore[index]
      if cached_encoders != expected_encoders:
        return False
      if "region_np" not in fp:
        return False
      patch_num = int(attrs.get("patch_num", -1))
      if patch_num < 0:
        return False
      if int(fp["region_np"].shape[0]) != patch_num:
        return False
      for encoder_name in expected_encoders:
        feature_name = f"{encoder_name}_feature_np"
        if feature_name not in fp:
          return False
        if int(fp[feature_name].shape[0]) != patch_num:
          return False
  except Exception:
    return False
  return True


def _write_h5_atomic(
  path: Path,
  *,
  wsi_file: str,
  wsi_md5: str,
  runtime_params: Dict[str, object],
  tissue: Optional[np.ndarray],
  region_np: np.ndarray,
  attention_np: Optional[np.ndarray],
  encoder_feature_dict: Dict[str, np.ndarray],
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
  string_dtype = h5py.string_dtype(encoding="utf-8")
  with h5py.File(tmp_path.as_posix(), "w") as fp:
    fp.attrs["schema_version"] = "slimai_embedding_v1"
    fp.attrs["wsi_file"] = wsi_file
    fp.attrs["wsi_md5"] = wsi_md5
    fp.attrs["patch_num"] = int(region_np.shape[0])
    fp.attrs["embedding_tag"] = runtime_params["embedding_tag"]
    fp.attrs["read_scale"] = runtime_params["read_scale"]
    fp.attrs["operate_scale"] = runtime_params["operate_scale"]
    fp.attrs["patch_size"] = runtime_params["patch_size"]
    fp.attrs["stride_size"] = runtime_params["stride_size"]
    fp.attrs["to_gray"] = runtime_params["to_gray"]
    fp.attrs["min_tissue_ratio"] = runtime_params["min_tissue_ratio"]
    fp.attrs["tissue_shrink"] = runtime_params["tissue_shrink"]
    encoder_list = list(runtime_params["encoder_list"])  # type: ignore[arg-type]
    fp.attrs.create("encoder_list", np.asarray(encoder_list, dtype=object), dtype=string_dtype)
    fp.attrs.create("patch_encoder_list", np.asarray(encoder_list, dtype=object), dtype=string_dtype)
    if tissue is None:
      fp.create_dataset("tissue", data=np.zeros((0,), dtype=np.uint8), dtype=np.uint8)
    else:
      fp.create_dataset("tissue", data=tissue, dtype=np.uint8)
    fp.create_dataset("region_np", data=region_np.astype(np.float32), dtype=np.float32)
    if attention_np is None:
      fp.create_dataset("attention_np", data=np.zeros((0,), dtype=np.float32), dtype=np.float32)
    else:
      fp.create_dataset("attention_np", data=attention_np.astype(np.float32), dtype=np.float32)
    for encoder_name in encoder_list:
      feature_name = f"{encoder_name}_feature_np"
      feature_np = encoder_feature_dict.get(encoder_name)
      if feature_np is None:
        raise ValueError(f"Missing features for encoder: {encoder_name}")
      fp.create_dataset(feature_name, data=feature_np.astype(np.float32), dtype=np.float32)
  os.replace(tmp_path.as_posix(), path.as_posix())
  return


def _check_runtime_dependencies() -> Dict[str, Dict[str, object]]:
  modules = [
    "torch",
    "timm",
    "h5py",
    "pandas",
    "openpyxl",
    "cv2",
    "sdk.reader",
  ]
  results: Dict[str, Dict[str, object]] = {}
  for module_name in modules:
    try:
      module = importlib.import_module(module_name)
      results[module_name] = {
        "ok": True,
        "version": str(getattr(module, "__version__", "unknown")),
      }
    except Exception as exc:
      results[module_name] = {
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
      }
  try:
    import torch_npu # type: ignore
    results["torch_npu"] = {"ok": True, "version": str(getattr(torch_npu, "__version__", "unknown"))}
  except Exception as exc:
    results["torch_npu"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
  return results


def _write_config_snapshot(
  path: Path,
  *,
  config: CreateFeatureConfig,
  runtime: RunRuntime,
  manifest_path: Path,
) -> None:
  payload: Dict[str, object] = asdict(config)
  payload["written_at"] = _now_iso()
  payload["manifest_path"] = manifest_path.as_posix()
  payload["requested_accelerator"] = runtime.requested_accelerator
  payload["resolved_accelerator"] = runtime.resolved_accelerator
  payload["available_devices"] = runtime.available_devices
  payload["resolved_devices"] = runtime.resolved_devices
  payload["torch_version"] = str(torch.__version__)
  try:
    import torch_npu # type: ignore
    payload["torch_npu_version"] = str(getattr(torch_npu, "__version__", "unknown"))
  except Exception:
    payload["torch_npu_version"] = None
  _write_json(path, payload)
  return


def _check_existing_config_or_raise(path: Path, config: CreateFeatureConfig) -> None:
  if not path.exists():
    return
  try:
    old_payload = json.loads(path.read_text(encoding="utf-8"))
  except Exception:
    return
  critical_keys = [
    "tag",
    "patch_encoder",
    "patch_encoders",
    "accelerator",
    "devices",
    "patch_size",
    "stride_size",
    "read_scale",
    "operate_scale",
    "to_gray",
    "min_tissue_ratio",
    "tissue_shrink",
  ]
  new_payload = asdict(config)
  mismatch = []
  for key in critical_keys:
    if old_payload.get(key) != new_payload.get(key):
      mismatch.append(key)
  if len(mismatch) > 0:
    raise ValueError(
      "Existing config.json has mismatched keys {}. "
      "Please confirm and use a new --out-dir or align parameters.".format(mismatch)
    )
  return


def _write_preflight(
  path: Path,
  *,
  config: CreateFeatureConfig,
  runtime: RunRuntime,
  manifest: Sequence[ManifestItem],
  dependency_check: Dict[str, Dict[str, object]],
  model_check: Dict[str, object],
) -> None:
  suffix_count: Dict[str, int] = {}
  for item in manifest:
    suffix_count[item.suffix] = suffix_count.get(item.suffix, 0) + 1
  statvfs = os.statvfs(path.parent.as_posix())
  payload = {
    "created_at": _now_iso(),
    "input_file": config.input_file,
    "input_mode": runtime.input_mode,
    "manifest_count": len(manifest),
    "suffix_count": suffix_count,
    "requested_accelerator": runtime.requested_accelerator,
    "resolved_accelerator": runtime.resolved_accelerator,
    "available_devices": runtime.available_devices,
    "requested_devices": config.devices,
    "resolved_devices": runtime.resolved_devices,
    "torch_version": str(torch.__version__),
    "dependency_check": dependency_check,
    "model_check": model_check,
    "disk_free_bytes": int(statvfs.f_bavail * statvfs.f_frsize),
    "collision_checked": True,
  }
  _write_json(path, payload)
  return


def _cleanup_stale_tmp_h5(out_dir: Path, run_log: Path) -> int:
  count = 0
  for path in out_dir.rglob("*.tmp.*"):
    if path.is_file():
      path.unlink(missing_ok=True)
      count += 1
  _append_run_log(run_log, f"stale_temp_cleanup count={count}")
  return count


def _acquire_run_lock(lock_path: Path) -> None:
  if lock_path.exists():
    try:
      payload = json.loads(lock_path.read_text(encoding="utf-8"))
      pid = int(payload.get("pid", -1))
      if pid > 0:
        os.kill(pid, 0)
        raise RuntimeError(f"Another run is active with pid={pid}, lock={lock_path.as_posix()}")
    except ProcessLookupError:
      pass
    except json.JSONDecodeError:
      pass
    except PermissionError:
      raise RuntimeError(f"Cannot validate run lock: {lock_path.as_posix()}")
  _write_json(
    lock_path,
    {
      "pid": os.getpid(),
      "created_at": _now_iso(),
      "command": "create.py",
    },
  )
  return


def _setup_sigint(runtime: RunRuntime, run_log: Path):
  previous = signal.getsignal(signal.SIGINT)

  def _handler(signum, frame):
    _ = signum
    _ = frame
    if runtime.stop_requested:
      return
    runtime.stop_requested = True
    _append_run_log(run_log, "signal=SIGINT stop_requested=1")
    logger.warning("SIGINT received. Stop after current WSI.")
    return

  signal.signal(signal.SIGINT, _handler)
  return previous


def _build_encoder_engines(
  *,
  encoder_list: Sequence[str],
  devices: Sequence[int],
  slide_encoder_name: Optional[str],
  accelerator: str,
) -> Dict[str, Dict[int, FeatureExtractor]]:
  model_engines: Dict[str, Dict[int, FeatureExtractor]] = {}
  for encoder_name in encoder_list:
    model_engines[encoder_name] = {}
    for device_id in devices:
      model_engines[encoder_name][device_id] = build_feature_extractor(
        encoder_name,
        slide_encoder_name,
        device_id=device_id,
        accelerator=accelerator,
      )
  return model_engines


def _safe_model_precheck(
  *,
  encoder_name: str,
  resolved_accelerator: str,
  device_id: int,
  slide_encoder_name: Optional[str],
) -> Dict[str, object]:
  try:
    extractor = build_feature_extractor(
      encoder_name,
      slide_encoder_name,
      device_id=device_id,
      accelerator=resolved_accelerator,
    )
    return {
      "ok": True,
      "encoder": encoder_name,
      "device": str(extractor.device),
      "feature_dim": int(extractor.feature_dim),
    }
  except Exception as exc:
    return {
      "ok": False,
      "encoder": encoder_name,
      "error_type": type(exc).__name__,
      "error": str(exc),
    }


def main(config: CreateFeatureConfig) -> None:
  out_dir = Path(config.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  embeddings_dir = out_dir / "embeddings"
  embeddings_dir.mkdir(parents=True, exist_ok=True)

  run_log = out_dir / "run.log"
  success_jsonl = out_dir / "success.jsonl"
  failed_jsonl = out_dir / "failed.jsonl"
  lock_path = out_dir / "run.lock"
  config_path = out_dir / "config.json"
  preflight_path = out_dir / "preflight.json"
  manifest_path = out_dir / "manifest.jsonl"

  _acquire_run_lock(lock_path)
  _cleanup_stale_tmp_h5(out_dir, run_log)

  try:
    encoder_list = _parse_encoder_list(config)
    input_pairs, input_mode, preserve_structure = _read_input_pairs(config)
    if len(input_pairs) == 0:
      raise ValueError("No wsi files found in input.")

    manifest = _build_manifest(
      input_pairs,
      out_dir=out_dir,
      embedding_tag=config.tag,
      preserve_structure=preserve_structure,
    )
    _check_manifest_collisions(manifest)
    if config.limit > 0:
      manifest = manifest[:config.limit]
    _write_manifest(manifest_path, manifest)

    resolved_accelerator, available_devices = resolve_accelerator(config.accelerator)
    resolved_devices = parse_devices(config.devices, resolved_accelerator)
    if len(resolved_devices) == 0:
      raise ValueError("No devices resolved. Use --devices 0 or --devices 0,1,...")

    runtime = RunRuntime(
      requested_accelerator=config.accelerator,
      resolved_accelerator=resolved_accelerator,
      available_devices=available_devices,
      resolved_devices=resolved_devices,
      input_mode=input_mode,
    )
    log_device_resolution(
      requested_accelerator=runtime.requested_accelerator,
      resolved_accelerator=runtime.resolved_accelerator,
      available_devices=runtime.available_devices,
      requested_devices=config.devices,
      resolved_devices=runtime.resolved_devices,
    )

    dependency_check = _check_runtime_dependencies()
    model_check = _safe_model_precheck(
      encoder_name=encoder_list[0],
      resolved_accelerator=resolved_accelerator,
      device_id=resolved_devices[0],
      slide_encoder_name=config.slide_encoder,
    )
    _write_preflight(
      preflight_path,
      config=config,
      runtime=runtime,
      manifest=manifest,
      dependency_check=dependency_check,
      model_check=model_check,
    )

    if config.preflight_only:
      _append_run_log(run_log, "preflight_only=1 exit_without_inference=1")
      return

    _check_existing_config_or_raise(config_path, config)
    _write_config_snapshot(
      config_path,
      config=config,
      runtime=runtime,
      manifest_path=manifest_path,
    )

    if config.manifest_only:
      _append_run_log(run_log, "manifest_only=1 exit_without_inference=1")
      return

    model_engines = _build_encoder_engines(
      encoder_list=encoder_list,
      devices=resolved_devices,
      slide_encoder_name=config.slide_encoder,
      accelerator=resolved_accelerator,
    )
    runtime_params = _extract_runtime_params(config, encoder_list)
    previous_handler = _setup_sigint(runtime, run_log)
    stats = {
      "generated": 0,
      "cache_hit": 0,
      "failed": 0,
      "skipped_missing_source": 0,
      "skipped_source_changed": 0,
      "skipped_unsupported_ext": 0,
    }
    max_workers = len(resolved_devices) if config.max_futs <= 0 else config.max_futs

    try:
      pbar = tqdm(manifest)
      for item in pbar:
        if runtime.stop_requested:
          _append_run_log(run_log, "stop_requested=1 break_loop=1")
          break
        pbar.set_description(
          f"Inference {Path(item.wsi_file).name} as tag: '{config.tag}' by '{','.join(encoder_list)}'"
        )
        wsi_path = Path(item.wsi_file)
        base_payload: Dict[str, object] = {
          "timestamp": _now_iso(),
          "wsi_file": item.wsi_file,
          "relative_path": item.relative_path,
          "expected_h5_path": item.expected_h5_path,
          "resolved_accelerator": resolved_accelerator,
          "resolved_devices": resolved_devices,
          "params": runtime_params,
        }

        if not wsi_path.exists():
          payload = dict(base_payload)
          payload["status"] = "skipped_missing_source"
          _append_jsonl(success_jsonl, payload)
          stats["skipped_missing_source"] += 1
          continue

        if not _is_supported_wsi(wsi_path):
          payload = dict(base_payload)
          payload["status"] = "skipped_unsupported_ext"
          _append_jsonl(success_jsonl, payload)
          stats["skipped_unsupported_ext"] += 1
          continue

        current_stat = wsi_path.stat()
        if item.size >= 0 and int(current_stat.st_size) != int(item.size):
          payload = dict(base_payload)
          payload["status"] = "skipped_source_changed"
          payload["manifest_size"] = int(item.size)
          payload["actual_size"] = int(current_stat.st_size)
          _append_jsonl(success_jsonl, payload)
          stats["skipped_source_changed"] += 1
          continue
        if item.mtime_ns >= 0 and int(current_stat.st_mtime_ns) != int(item.mtime_ns):
          payload = dict(base_payload)
          payload["status"] = "skipped_source_changed"
          payload["manifest_mtime_ns"] = int(item.mtime_ns)
          payload["actual_mtime_ns"] = int(current_stat.st_mtime_ns)
          _append_jsonl(success_jsonl, payload)
          stats["skipped_source_changed"] += 1
          continue

        try:
          wsi_md5 = _file_md5(item.wsi_file)
          expected_h5 = Path(item.expected_h5_path)
          if config.skip_existing and _validate_h5_cache(
            expected_h5,
            runtime_params=runtime_params,
            wsi_file=item.wsi_file,
            wsi_md5=wsi_md5,
          ):
            payload = dict(base_payload)
            payload["status"] = "cache_hit"
            payload["wsi_md5"] = wsi_md5
            payload["output_size"] = int(expected_h5.stat().st_size)
            _append_jsonl(success_jsonl, payload)
            stats["cache_hit"] += 1
            continue

          tissue_output = get_tissue_region(
            item.wsi_file,
            read_scale=config.read_scale,
            operate_scale=config.operate_scale,
            patch_size_h=config.patch_size,
            patch_size_w=config.patch_size,
            patch_stride_h=config.stride_size,
            patch_stride_w=config.stride_size,
            min_ratio=config.min_tissue_ratio,
            shrink=config.tissue_shrink,
            return_tissue=True,
            tissue_scale=1.25,
          )
          region_np = tissue_output.region_np
          tissue = tissue_output.tissue
          options = InferenceOptions(
            read_scale=config.read_scale,
            patch_size=config.patch_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            to_gray=config.to_gray,
            show_progress=True,
          )
          encoder_feature_dict: Dict[str, np.ndarray] = {}
          for encoder_name in encoder_list:
            tasks = build_chunk_tasks(
              wsi_file=item.wsi_file,
              region_np=region_np,
              devices=resolved_devices,
              model_engines=model_engines[encoder_name],
            )
            encoder_feature_dict[encoder_name] = run_chunk_tasks(
              tasks=tasks,
              options=options,
              max_workers=max_workers,
            )

          _write_h5_atomic(
            expected_h5,
            wsi_file=item.wsi_file,
            wsi_md5=wsi_md5,
            runtime_params=runtime_params,
            tissue=tissue,
            region_np=region_np,
            attention_np=None,
            encoder_feature_dict=encoder_feature_dict,
          )

          patch_num = int(region_np.shape[0])
          if config.output is not None:
            _append_record_to_xlsx(
              config.output,
              wsi_file=item.wsi_file,
              wsi_md5=wsi_md5,
              h5_path=item.expected_h5_path,
              patch_num=patch_num,
            )
          payload = dict(base_payload)
          payload["status"] = "generated"
          payload["wsi_md5"] = wsi_md5
          payload["patch_num"] = patch_num
          payload["output_size"] = int(expected_h5.stat().st_size)
          _append_jsonl(success_jsonl, payload)
          if patch_num == 0:
            _append_run_log(run_log, f"warning patch_num=0 wsi={item.wsi_file}")
          stats["generated"] += 1
        except Exception as exc:
          payload = dict(base_payload)
          payload["status"] = "failed"
          payload["error_type"] = type(exc).__name__
          payload["error"] = str(exc)
          payload["traceback"] = _short_traceback(exc)
          _append_jsonl(failed_jsonl, payload)
          _append_run_log(run_log, f"failed wsi={item.wsi_file} error={type(exc).__name__}:{exc}")
          stats["failed"] += 1
      _append_run_log(
        run_log,
        "summary generated={} cache_hit={} failed={} skipped_missing_source={} skipped_source_changed={} skipped_unsupported_ext={}".format(
          stats["generated"],
          stats["cache_hit"],
          stats["failed"],
          stats["skipped_missing_source"],
          stats["skipped_source_changed"],
          stats["skipped_unsupported_ext"],
        ),
      )
    finally:
      signal.signal(signal.SIGINT, previous_handler)
  finally:
    lock_path.unlink(missing_ok=True)
  return
