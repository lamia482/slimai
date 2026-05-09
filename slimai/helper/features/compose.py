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
import time
import traceback
from typing import Dict, List, Optional, Sequence, Set, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from sdk.reader import get_reader_by_ext
from .extract import build_feature_extractor
from .pipeline import get_tissue_region
from .task import (
  InferenceOptions,
  PersistentWorkerPool,
  build_chunk_tasks,
  create_persistent_worker_pool,
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
  batch_size: int = -1
  num_workers: int = -1
  patch_size: int = 224
  stride_size: int = 192
  read_scale: float = 20
  operate_scale: float = 1.25
  to_gray: bool = False
  skip_existing: bool = True
  incremental_h5: bool = False
  verify_existing_md5: bool = False
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


def _sec(start_ts: float) -> float:
  return max(time.perf_counter() - start_ts, 0.0)


def _fmt_sec(sec: float) -> str:
  return f"{max(sec, 0.0):.3f}s"


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


def _read_h5_attr_str(path: Path, key: str, default: str = "") -> str:
  try:
    with h5py.File(path.as_posix(), "r") as fp:
      return str(fp.attrs.get(key, default))
  except Exception:
    return default


def _validate_h5_cache(
  path: Path,
  *,
  runtime_params: Dict[str, object],
  wsi_file: str,
  source_size: Optional[int] = None,
  source_mtime_ns: Optional[int] = None,
  wsi_md5: Optional[str] = None,
  allow_encoder_superset: bool = False,
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
      if source_size is not None:
        cached_size = attrs.get("source_size_bytes", None)
        if cached_size is not None and int(cached_size) != int(source_size):
          return False
      if source_mtime_ns is not None:
        cached_mtime_ns = attrs.get("source_mtime_ns", None)
        if cached_mtime_ns is not None and int(cached_mtime_ns) != int(source_mtime_ns):
          return False
      if wsi_md5 is not None and str(attrs.get("wsi_md5", "")) != wsi_md5:
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
      if allow_encoder_superset:
        if not set(expected_encoders).issubset(set(cached_encoders)):
          return False
      else:
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


@dataclass(frozen=True)
class ExistingH5State:
  wsi_md5: str
  patch_num: int
  existing_encoders: List[str]
  region_np: np.ndarray


def _read_json_dict(path: Path) -> Optional[Dict[str, object]]:
  if not path.exists():
    return None
  try:
    payload = json.loads(path.read_text(encoding="utf-8"))
  except Exception:
    return None
  if not isinstance(payload, dict):
    return None
  return payload


def _encoder_list_from_config_payload(payload: Dict[str, object]) -> List[str]:
  patch_encoders = str(payload.get("patch_encoders", "") or "").strip()
  if patch_encoders != "":
    return [item.strip() for item in patch_encoders.split(",") if item.strip() != ""]
  patch_encoder = str(payload.get("patch_encoder", "") or "").strip()
  if patch_encoder == "":
    return []
  return [patch_encoder]


def _dedupe_preserve(items: Sequence[str]) -> List[str]:
  deduped: List[str] = []
  seen: Set[str] = set()
  for item in items:
    if item in seen:
      continue
    seen.add(item)
    deduped.append(item)
  return deduped


def _collect_overwrite_risks(
  manifest: Sequence[ManifestItem],
  *,
  runtime_params: Dict[str, object],
  config: CreateFeatureConfig,
) -> None:
  if config.incremental_h5:
    return
  sample: List[str] = []
  for item in manifest:
    expected_h5 = Path(item.expected_h5_path)
    if not expected_h5.exists():
      continue
    if not config.skip_existing:
      sample.append(
        f"{expected_h5.as_posix()} (requested --no-skip-existing will replace this file)"
      )
      if len(sample) >= 3:
        break
      continue
    if not _validate_h5_cache(
      expected_h5,
      runtime_params=runtime_params,
      wsi_file=item.wsi_file,
      source_size=item.size if item.size >= 0 else None,
      source_mtime_ns=item.mtime_ns if item.mtime_ns >= 0 else None,
      wsi_md5=None,
      allow_encoder_superset=False,
    ):
      sample.append(
        (
          f"{expected_h5.as_posix()} (existing h5 is incompatible; rerun would overwrite). "
          "Use --incremental-h5 to append encoder datasets or change --out-dir/--tag."
        )
      )
      if len(sample) >= 3:
        break
  if len(sample) == 0:
    return
  raise ValueError(
    "Refuse to overwrite existing h5 files due to high-risk parameters. "
    "Examples: {}".format(" | ".join(sample))
  )


def _read_existing_h5_state(
  *,
  path: Path,
  runtime_params: Dict[str, object],
  wsi_file: str,
  source_size: int,
  source_mtime_ns: int,
  wsi_md5: Optional[str],
) -> ExistingH5State:
  with h5py.File(path.as_posix(), "r") as fp:
    attrs = fp.attrs
    if str(attrs.get("schema_version", "")) != "slimai_embedding_v1":
      raise ValueError(f"incompatible schema_version: {attrs.get('schema_version', '')}")
    if str(attrs.get("wsi_file", "")) != wsi_file:
      raise ValueError("incompatible wsi_file")
    cached_size = attrs.get("source_size_bytes", None)
    if cached_size is not None and int(cached_size) != int(source_size):
      raise ValueError(f"incompatible source_size_bytes: {cached_size} != {source_size}")
    cached_mtime_ns = attrs.get("source_mtime_ns", None)
    if cached_mtime_ns is not None and int(cached_mtime_ns) != int(source_mtime_ns):
      raise ValueError(f"incompatible source_mtime_ns: {cached_mtime_ns} != {source_mtime_ns}")
    if wsi_md5 is not None and str(attrs.get("wsi_md5", "")) != wsi_md5:
      raise ValueError("incompatible wsi_md5")
    if int(attrs.get("patch_size", -1)) != int(runtime_params["patch_size"]):
      raise ValueError("incompatible patch_size")
    if int(attrs.get("stride_size", -1)) != int(runtime_params["stride_size"]):
      raise ValueError("incompatible stride_size")
    if float(attrs.get("read_scale", -1.0)) != float(runtime_params["read_scale"]):
      raise ValueError("incompatible read_scale")
    if float(attrs.get("operate_scale", -1.0)) != float(runtime_params["operate_scale"]):
      raise ValueError("incompatible operate_scale")
    if bool(attrs.get("to_gray", False)) != bool(runtime_params["to_gray"]):
      raise ValueError("incompatible to_gray")
    if float(attrs.get("min_tissue_ratio", -1.0)) != float(runtime_params["min_tissue_ratio"]):
      raise ValueError("incompatible min_tissue_ratio")
    if str(attrs.get("tissue_shrink", "")) != str(runtime_params["tissue_shrink"]):
      raise ValueError("incompatible tissue_shrink")
    if str(attrs.get("embedding_tag", "")) != str(runtime_params["embedding_tag"]):
      raise ValueError("incompatible embedding_tag")
    if "region_np" not in fp:
      raise ValueError("region_np dataset missing")
    patch_num = int(attrs.get("patch_num", -1))
    if patch_num < 0:
      raise ValueError("invalid patch_num")
    region_np = fp["region_np"][:].astype(np.float32)
    if int(region_np.shape[0]) != patch_num:
      raise ValueError("region_np length mismatch patch_num")
    existing_encoders = _normalize_encoder_list(attrs.get("encoder_list"))
    for encoder_name in existing_encoders:
      feature_name = f"{encoder_name}_feature_np"
      if feature_name not in fp:
        raise ValueError(f"missing dataset: {feature_name}")
      if int(fp[feature_name].shape[0]) != patch_num:
        raise ValueError(f"dataset length mismatch: {feature_name}")
    return ExistingH5State(
      wsi_md5=str(attrs.get("wsi_md5", "")),
      patch_num=patch_num,
      existing_encoders=existing_encoders,
      region_np=region_np,
    )


def _append_h5_features(
  *,
  path: Path,
  runtime_params: Dict[str, object],
  encoder_feature_dict: Dict[str, np.ndarray],
) -> None:
  if len(encoder_feature_dict) == 0:
    return
  with h5py.File(path.as_posix(), "a") as fp:
    attrs = fp.attrs
    patch_num = int(attrs.get("patch_num", -1))
    if patch_num < 0:
      raise ValueError("invalid patch_num in existing h5")
    existing_encoders = _normalize_encoder_list(attrs.get("encoder_list"))
    existing_encoder_set = set(existing_encoders)
    for encoder_name, feature_np in encoder_feature_dict.items():
      feature_name = f"{encoder_name}_feature_np"
      if feature_name in fp:
        continue
      if int(feature_np.shape[0]) != patch_num:
        raise ValueError(
          f"feature row mismatch for {feature_name}: {feature_np.shape[0]} != {patch_num}"
        )
      tmp_name = f"tmp_{feature_name}_{os.getpid()}"
      if tmp_name in fp:
        del fp[tmp_name]
      fp.create_dataset(tmp_name, data=feature_np.astype(np.float32), dtype=np.float32)
      fp.move(tmp_name, feature_name)
      if encoder_name not in existing_encoder_set:
        existing_encoders.append(encoder_name)
        existing_encoder_set.add(encoder_name)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    deduped_encoders = _dedupe_preserve(existing_encoders)
    if "encoder_list" in attrs:
      del attrs["encoder_list"]
    attrs.create("encoder_list", np.asarray(deduped_encoders, dtype=object), dtype=string_dtype)
    if "patch_encoder_list" in attrs:
      del attrs["patch_encoder_list"]
    attrs.create("patch_encoder_list", np.asarray(deduped_encoders, dtype=object), dtype=string_dtype)
  return


def _write_h5_atomic(
  path: Path,
  *,
  wsi_file: str,
  source_size: int,
  source_mtime_ns: int,
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
    fp.attrs["source_size_bytes"] = int(source_size)
    fp.attrs["source_mtime_ns"] = int(source_mtime_ns)
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
  payload_overrides: Optional[Dict[str, object]] = None,
) -> None:
  payload: Dict[str, object] = asdict(config)
  if payload_overrides is not None and len(payload_overrides) > 0:
    payload.update(payload_overrides)
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


def _check_existing_config_or_raise(
  path: Path,
  config: CreateFeatureConfig,
  *,
  requested_encoders: Sequence[str],
) -> Dict[str, object]:
  old_payload = _read_json_dict(path)
  if old_payload is None:
    return {}
  critical_keys = [
    "tag",
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
  old_encoders = _encoder_list_from_config_payload(old_payload)
  new_encoders = _dedupe_preserve([str(item) for item in requested_encoders])
  if old_encoders == new_encoders:
    return {}
  if not config.incremental_h5:
    raise ValueError(
      "Existing config.json has mismatched encoder list old={} new={}. "
      "To prevent accidental overwrite, either use --incremental-h5 to append datasets "
      "or choose a new --out-dir/--tag.".format(old_encoders, new_encoders)
    )
  merged_encoders = _dedupe_preserve(old_encoders + new_encoders)
  overrides: Dict[str, object] = {}
  if len(merged_encoders) == 1:
    overrides["patch_encoder"] = merged_encoders[0]
    overrides["patch_encoders"] = None
  else:
    overrides["patch_encoder"] = None
    overrides["patch_encoders"] = ",".join(merged_encoders)
  return overrides


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


def _safe_model_precheck(
  *,
  encoder_name: str,
  resolved_accelerator: str,
  device_id: int,
  slide_encoder_name: Optional[str],
) -> Dict[str, object]:
  extractor = None
  try:
    extractor = build_feature_extractor(
      encoder_name,
      slide_encoder_name,
      device_id=device_id,
      accelerator=resolved_accelerator,
    )
    result = {
      "ok": True,
      "encoder": encoder_name,
      "device": str(extractor.device),
      "feature_dim": int(extractor.feature_dim),
    }
  except Exception as exc:
    result = {
      "ok": False,
      "encoder": encoder_name,
      "error_type": type(exc).__name__,
      "error": str(exc),
    }
  finally:
    if extractor is not None:
      del extractor
    try:
      if resolved_accelerator == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache() # type: ignore[attr-defined]
      elif resolved_accelerator == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    except Exception:
      pass
  return result


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

    auto_num_workers = config.num_workers
    if auto_num_workers < 0:
      cpu_count = os.cpu_count() or 8
      device_count = max(len(resolved_devices), 1)
      auto_num_workers = max(2, min(8, cpu_count // device_count // 2))

    auto_batch_size = config.batch_size
    if auto_batch_size < 0:
      auto_batch_size = max(min(auto_num_workers * 4, 16), auto_num_workers)

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

    runtime_params = _extract_runtime_params(config, encoder_list)
    config_snapshot_overrides = _check_existing_config_or_raise(
      config_path,
      config,
      requested_encoders=encoder_list,
    )
    _collect_overwrite_risks(
      manifest,
      runtime_params=runtime_params,
      config=config,
    )
    if config.preflight_only or not preflight_path.exists():
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
    else:
      _append_run_log(
        run_log,
        f"preflight_reused=1 model_precheck_skipped=1 path={preflight_path.as_posix()}",
      )

    if config.preflight_only:
      _append_run_log(run_log, "preflight_only=1 exit_without_inference=1")
      return

    _write_config_snapshot(
      config_path,
      config=config,
      runtime=runtime,
      manifest_path=manifest_path,
      payload_overrides=config_snapshot_overrides,
    )

    if config.manifest_only:
      _append_run_log(run_log, "manifest_only=1 exit_without_inference=1")
      return

    options = InferenceOptions(
      read_scale=config.read_scale,
      patch_size=config.patch_size,
      batch_size=auto_batch_size,
      num_workers=auto_num_workers,
      to_gray=config.to_gray,
      show_progress=True,
    )
    previous_handler = _setup_sigint(runtime, run_log)
    persistent_pools: Dict[str, PersistentWorkerPool] = {}
    persistent_disabled = False
    stats = {
      "generated": 0,
      "cache_hit": 0,
      "incremental_generated": 0,
      "incremental_cache_hit": 0,
      "failed": 0,
      "skipped_missing_source": 0,
      "skipped_source_changed": 0,
      "skipped_unsupported_ext": 0,
    }
    perf_totals: Dict[str, float] = {
      "cache_hit_total_sec": 0.0,
      "generated_total_sec": 0.0,
      "cache_validate_sec": 0.0,
      "source_md5_sec": 0.0,
      "tissue_region_sec": 0.0,
      "build_tasks_sec": 0.0,
      "run_chunks_sec": 0.0,
      "write_h5_sec": 0.0,
      "record_success_sec": 0.0,
    }
    chunk_metric_totals: Dict[str, float] = {
      "spawn_to_start_sec": 0.0,
      "build_feature_extractor_sec": 0.0,
      "dataloader_setup_sec": 0.0,
      "first_batch_latency_sec": 0.0,
      "infer_loop_sec": 0.0,
      "save_npy_sec": 0.0,
      "worker_total_sec": 0.0,
      "throughput_patch_per_sec": 0.0,
    }
    chunk_metric_count = 0
    max_workers = len(resolved_devices) if config.max_futs <= 0 else config.max_futs

    try:
      pbar = tqdm(manifest)
      for item in pbar:
        item_start = time.perf_counter()
        stage_times: Dict[str, float] = {
          "cache_validate_sec": 0.0,
          "source_md5_sec": 0.0,
          "tissue_region_sec": 0.0,
          "build_tasks_sec": 0.0,
          "run_chunks_sec": 0.0,
          "write_h5_sec": 0.0,
          "record_success_sec": 0.0,
        }
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
          expected_h5 = Path(item.expected_h5_path)
          if config.verify_existing_md5:
            existing_md5_start = time.perf_counter()
            existing_wsi_md5 = _file_md5(item.wsi_file)
            stage_times["source_md5_sec"] += _sec(existing_md5_start)
          else:
            existing_wsi_md5 = None
          cache_validate_start = time.perf_counter()
          if config.skip_existing and _validate_h5_cache(
            expected_h5,
            runtime_params=runtime_params,
            wsi_file=item.wsi_file,
            source_size=int(current_stat.st_size),
            source_mtime_ns=int(current_stat.st_mtime_ns),
            wsi_md5=existing_wsi_md5,
            allow_encoder_superset=config.incremental_h5,
          ):
            stage_times["cache_validate_sec"] += _sec(cache_validate_start)
            payload = dict(base_payload)
            payload["status"] = "incremental_cache_hit" if config.incremental_h5 else "cache_hit"
            payload["wsi_md5"] = existing_wsi_md5 or _read_h5_attr_str(expected_h5, "wsi_md5")
            payload["output_size"] = int(expected_h5.stat().st_size)
            payload["perf"] = {
              "cache_validate_sec": stage_times["cache_validate_sec"],
              "source_md5_sec": stage_times["source_md5_sec"],
              "wsi_total_sec": _sec(item_start),
            }
            _append_jsonl(success_jsonl, payload)
            if payload["status"] == "incremental_cache_hit":
              stats["incremental_cache_hit"] += 1
            else:
              stats["cache_hit"] += 1
            perf_totals["cache_hit_total_sec"] += payload["perf"]["wsi_total_sec"]  # type: ignore[index]
            perf_totals["cache_validate_sec"] += stage_times["cache_validate_sec"]
            perf_totals["source_md5_sec"] += stage_times["source_md5_sec"]
            _append_run_log(
              run_log,
              "perf_cache_hit wsi={} total={} cache_validate={} source_md5={}".format(
                item.wsi_file,
                _fmt_sec(payload["perf"]["wsi_total_sec"]),  # type: ignore[index]
                _fmt_sec(stage_times["cache_validate_sec"]),
                _fmt_sec(stage_times["source_md5_sec"]),
              ),
            )
            continue
          stage_times["cache_validate_sec"] += _sec(cache_validate_start)

          incremental_mode = False
          encoder_list_to_run = list(encoder_list)
          wsi_md5 = existing_wsi_md5 or ""
          if config.incremental_h5 and expected_h5.exists():
            incremental_state = _read_existing_h5_state(
              path=expected_h5,
              runtime_params=runtime_params,
              wsi_file=item.wsi_file,
              source_size=int(current_stat.st_size),
              source_mtime_ns=int(current_stat.st_mtime_ns),
              wsi_md5=existing_wsi_md5,
            )
            existing_encoder_set = set(incremental_state.existing_encoders)
            encoder_list_to_run = [name for name in encoder_list if name not in existing_encoder_set]
            if len(encoder_list_to_run) == 0:
              payload = dict(base_payload)
              payload["status"] = "incremental_cache_hit"
              payload["wsi_md5"] = wsi_md5 or incremental_state.wsi_md5
              payload["output_size"] = int(expected_h5.stat().st_size)
              payload["perf"] = {
                "cache_validate_sec": stage_times["cache_validate_sec"],
                "source_md5_sec": stage_times["source_md5_sec"],
                "wsi_total_sec": _sec(item_start),
              }
              _append_jsonl(success_jsonl, payload)
              stats["incremental_cache_hit"] += 1
              perf_totals["cache_hit_total_sec"] += payload["perf"]["wsi_total_sec"]  # type: ignore[index]
              continue
            region_np = incremental_state.region_np
            tissue = None
            if wsi_md5 == "":
              wsi_md5 = incremental_state.wsi_md5
            incremental_mode = True
          else:
            tissue_start = time.perf_counter()
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
            stage_times["tissue_region_sec"] += _sec(tissue_start)
            region_np = tissue_output.region_np
            tissue = tissue_output.tissue
          if wsi_md5 == "":
            source_md5_start = time.perf_counter()
            wsi_md5 = _file_md5(item.wsi_file)
            stage_times["source_md5_sec"] += _sec(source_md5_start)
          encoder_feature_dict: Dict[str, np.ndarray] = {}
          for encoder_name in encoder_list_to_run:
            build_tasks_start = time.perf_counter()
            tasks = build_chunk_tasks(
              wsi_file=item.wsi_file,
              encoder_name=encoder_name,
              slide_encoder_name=config.slide_encoder,
              region_np=region_np,
              devices=resolved_devices,
              accelerator=resolved_accelerator,
            )
            stage_times["build_tasks_sec"] += _sec(build_tasks_start)
            run_chunks_start = time.perf_counter()
            pool = None
            created_pool = False
            if not persistent_disabled:
              pool = persistent_pools.get(encoder_name)
              if pool is None:
                pool = create_persistent_worker_pool(
                  encoder_name=encoder_name,
                  slide_encoder_name=config.slide_encoder,
                  accelerator=resolved_accelerator,
                  devices=resolved_devices,
                  options=options,
                )
                persistent_pools[encoder_name] = pool
                created_pool = True
            try:
              feature_np, chunk_metrics = run_chunk_tasks(
                tasks=tasks,
                options=options,
                max_workers=max_workers,
                persistent_pool=pool,
              )
              if created_pool and pool is not None:
                for metric in sorted(pool.startup_metrics.values(), key=lambda x: int(x.get("device_id", -1))):
                  _append_run_log(
                    run_log,
                    (
                      "perf_worker_startup encoder={} device={} pid={} spawn_to_ready={} "
                      "build_feature_extractor={} dataloader_setup={} startup_total={}"
                    ).format(
                      encoder_name,
                      metric.get("device_id"),
                      metric.get("pid"),
                      _fmt_sec(float(metric.get("spawn_to_ready_sec", 0.0))),
                      _fmt_sec(float(metric.get("build_feature_extractor_sec", 0.0))),
                      _fmt_sec(float(metric.get("dataloader_setup_sec", 0.0))),
                      _fmt_sec(float(metric.get("worker_startup_total_sec", 0.0))),
                    ),
                  )
            except Exception as pool_exc:
              if pool is None:
                raise
              _append_run_log(
                run_log,
                (
                  "persistent_pool_failed encoder={} wsi={} error_type={} error={} fallback_ephemeral=1"
                ).format(
                  encoder_name,
                  item.wsi_file,
                  type(pool_exc).__name__,
                  pool_exc,
                ),
              )
              persistent_disabled = True
              for worker_pool in persistent_pools.values():
                worker_pool.close()
              persistent_pools.clear()
              feature_np, chunk_metrics = run_chunk_tasks(
                tasks=tasks,
                options=options,
                max_workers=max_workers,
                persistent_pool=None,
              )
            stage_times["run_chunks_sec"] += _sec(run_chunks_start)
            encoder_feature_dict[encoder_name] = feature_np
            if len(chunk_metrics) > 0:
              per_chunk_tokens = []
              for chunk_metric in sorted(chunk_metrics, key=lambda x: int(x.get("chunk_index", -1))):
                device_id = int(chunk_metric.get("device_id", -1))
                throughput = float(chunk_metric.get("throughput_patch_per_sec", 0.0))
                worker_total_sec = float(chunk_metric.get("worker_total_sec", 0.0))
                per_chunk_tokens.append(f"npu:{device_id}:{_fmt_sec(worker_total_sec)}:{throughput:.1f}patch/s")
                for metric_name in chunk_metric_totals.keys():
                  chunk_metric_totals[metric_name] += float(chunk_metric.get(metric_name, 0.0))
                chunk_metric_count += 1
              _append_run_log(
                run_log,
                "perf_chunks wsi={} encoder={} {}".format(
                  item.wsi_file,
                  encoder_name,
                  " ".join(per_chunk_tokens),
                ),
              )

          write_h5_start = time.perf_counter()
          if incremental_mode and expected_h5.exists():
            _append_h5_features(
              path=expected_h5,
              runtime_params=runtime_params,
              encoder_feature_dict=encoder_feature_dict,
            )
          else:
            _write_h5_atomic(
              expected_h5,
              wsi_file=item.wsi_file,
              source_size=int(current_stat.st_size),
              source_mtime_ns=int(current_stat.st_mtime_ns),
              wsi_md5=wsi_md5,
              runtime_params=runtime_params,
              tissue=tissue,
              region_np=region_np,
              attention_np=None,
              encoder_feature_dict=encoder_feature_dict,
            )
          stage_times["write_h5_sec"] += _sec(write_h5_start)

          patch_num = int(region_np.shape[0])
          record_start = time.perf_counter()
          if config.output is not None:
            _append_record_to_xlsx(
              config.output,
              wsi_file=item.wsi_file,
              wsi_md5=wsi_md5,
              h5_path=item.expected_h5_path,
              patch_num=patch_num,
            )
          payload = dict(base_payload)
          payload["status"] = "incremental_generated" if incremental_mode else "generated"
          payload["wsi_md5"] = wsi_md5
          payload["patch_num"] = patch_num
          if incremental_mode:
            payload["generated_encoders"] = list(encoder_list_to_run)
          payload["output_size"] = int(expected_h5.stat().st_size)
          stage_times["record_success_sec"] += _sec(record_start)
          payload["perf"] = {
            "cache_validate_sec": stage_times["cache_validate_sec"],
            "source_md5_sec": stage_times["source_md5_sec"],
            "tissue_region_sec": stage_times["tissue_region_sec"],
            "build_tasks_sec": stage_times["build_tasks_sec"],
            "run_chunks_sec": stage_times["run_chunks_sec"],
            "write_h5_sec": stage_times["write_h5_sec"],
            "record_success_sec": stage_times["record_success_sec"],
            "wsi_total_sec": _sec(item_start),
          }
          _append_jsonl(success_jsonl, payload)
          if patch_num == 0:
            _append_run_log(run_log, f"warning patch_num=0 wsi={item.wsi_file}")
          if incremental_mode:
            stats["incremental_generated"] += 1
          else:
            stats["generated"] += 1
          perf_totals["generated_total_sec"] += payload["perf"]["wsi_total_sec"]  # type: ignore[index]
          perf_totals["cache_validate_sec"] += stage_times["cache_validate_sec"]
          perf_totals["source_md5_sec"] += stage_times["source_md5_sec"]
          perf_totals["tissue_region_sec"] += stage_times["tissue_region_sec"]
          perf_totals["build_tasks_sec"] += stage_times["build_tasks_sec"]
          perf_totals["run_chunks_sec"] += stage_times["run_chunks_sec"]
          perf_totals["write_h5_sec"] += stage_times["write_h5_sec"]
          perf_totals["record_success_sec"] += stage_times["record_success_sec"]
          _append_run_log(
            run_log,
            (
              "perf_generated wsi={} total={} cache_validate={} source_md5={} tissue={} "
              "build_tasks={} run_chunks={} write_h5={} record={}"
            ).format(
              item.wsi_file,
              _fmt_sec(payload["perf"]["wsi_total_sec"]),  # type: ignore[index]
              _fmt_sec(stage_times["cache_validate_sec"]),
              _fmt_sec(stage_times["source_md5_sec"]),
              _fmt_sec(stage_times["tissue_region_sec"]),
              _fmt_sec(stage_times["build_tasks_sec"]),
              _fmt_sec(stage_times["run_chunks_sec"]),
              _fmt_sec(stage_times["write_h5_sec"]),
              _fmt_sec(stage_times["record_success_sec"]),
            ),
          )
        except Exception as exc:
          payload = dict(base_payload)
          payload["status"] = "failed"
          payload["error_type"] = type(exc).__name__
          payload["error"] = str(exc)
          payload["traceback"] = _short_traceback(exc)
          _append_jsonl(failed_jsonl, payload)
          _append_run_log(run_log, f"failed wsi={item.wsi_file} error={type(exc).__name__}:{exc}")
          stats["failed"] += 1
      total_generated = stats["generated"] + stats["incremental_generated"]
      total_cache_hit = stats["cache_hit"] + stats["incremental_cache_hit"]
      _append_run_log(
        run_log,
        "summary generated={} incremental_generated={} cache_hit={} incremental_cache_hit={} failed={} skipped_missing_source={} skipped_source_changed={} skipped_unsupported_ext={}".format(
          stats["generated"],
          stats["incremental_generated"],
          stats["cache_hit"],
          stats["incremental_cache_hit"],
          stats["failed"],
          stats["skipped_missing_source"],
          stats["skipped_source_changed"],
          stats["skipped_unsupported_ext"],
        ),
      )
      _append_run_log(
        run_log,
        (
          "perf_summary avg_generated={} avg_cache_hit={} avg_cache_validate={} avg_source_md5={} avg_tissue={} "
          "avg_build_tasks={} avg_run_chunks={} avg_write_h5={} avg_record_success={}"
        ).format(
          _fmt_sec(perf_totals["generated_total_sec"] / max(total_generated, 1)),
          _fmt_sec(perf_totals["cache_hit_total_sec"] / max(total_cache_hit, 1)),
          _fmt_sec(perf_totals["cache_validate_sec"] / max(total_generated + total_cache_hit, 1)),
          _fmt_sec(perf_totals["source_md5_sec"] / max(total_generated + total_cache_hit, 1)),
          _fmt_sec(perf_totals["tissue_region_sec"] / max(total_generated, 1)),
          _fmt_sec(perf_totals["build_tasks_sec"] / max(total_generated, 1)),
          _fmt_sec(perf_totals["run_chunks_sec"] / max(total_generated, 1)),
          _fmt_sec(perf_totals["write_h5_sec"] / max(total_generated, 1)),
          _fmt_sec(perf_totals["record_success_sec"] / max(total_generated, 1)),
        ),
      )
      if chunk_metric_count > 0:
        _append_run_log(
          run_log,
          (
            "perf_worker_summary avg_spawn_to_start={} avg_build_feature_extractor={} avg_dataloader_setup={} "
            "avg_first_batch_latency={} avg_infer_loop={} avg_save_npy={} avg_worker_total={} avg_throughput={:.2f}patch/s"
          ).format(
            _fmt_sec(chunk_metric_totals["spawn_to_start_sec"] / chunk_metric_count),
            _fmt_sec(chunk_metric_totals["build_feature_extractor_sec"] / chunk_metric_count),
            _fmt_sec(chunk_metric_totals["dataloader_setup_sec"] / chunk_metric_count),
            _fmt_sec(chunk_metric_totals["first_batch_latency_sec"] / chunk_metric_count),
            _fmt_sec(chunk_metric_totals["infer_loop_sec"] / chunk_metric_count),
            _fmt_sec(chunk_metric_totals["save_npy_sec"] / chunk_metric_count),
            _fmt_sec(chunk_metric_totals["worker_total_sec"] / chunk_metric_count),
            chunk_metric_totals["throughput_patch_per_sec"] / chunk_metric_count,
          ),
        )
    finally:
      for worker_pool in persistent_pools.values():
        worker_pool.close()
      signal.signal(signal.SIGINT, previous_handler)
  finally:
    lock_path.unlink(missing_ok=True)
  return
