from __future__ import annotations

import hashlib
import os.path as osp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from sdk.reader import get_reader_by_ext

from .extract import FeatureExtractor, build_feature_extractor
from .pipeline import get_tissue_region
from .task import InferenceOptions, build_chunk_tasks, parse_devices, run_chunk_tasks


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


def _parse_encoder_list(config: CreateFeatureConfig) -> List[str]:
  if config.patch_encoders is not None and config.patch_encoders.strip() != "":
    encoders = [item.strip() for item in config.patch_encoders.split(",") if item.strip() != ""]
  elif config.patch_encoder is not None and config.patch_encoder.strip() != "":
    encoders = [config.patch_encoder.strip()]
  else:
    raise ValueError("Either --patch-encoder or --patch-encoders is required.")

  deduped = []
  seen = set()
  for encoder in encoders:
    if encoder in seen:
      continue
    seen.add(encoder)
    deduped.append(encoder)
  return deduped


def _read_wsi_files(input_file: str, wsi_col: str) -> List[str]:
  input_path = Path(input_file)
  suffix = input_path.suffix.lower()

  try:
    get_reader_by_ext(suffix, strict=True)
    return [input_path.as_posix()]
  except Exception:
    pass

  *sheet_name, wsi_col_name = wsi_col.rsplit(":", 1)
  sheet_name_value = None if len(sheet_name) == 0 else sheet_name[0]

  if suffix in [".xlsx", ".xls"]:
    df = pd.read_excel(input_path, sheet_name=sheet_name_value)
  else:
    raise ValueError(f"Unsupported --input-file suffix: {suffix}")

  if wsi_col_name not in df.columns:
    raise ValueError(f"Column not found in input file: {wsi_col_name}. Available: {list(df.columns)}")

  base_dir = input_path.parent
  wsi_files = []
  for value in df[wsi_col_name].tolist():
    if value is None:
      continue
    item = str(value).strip()
    if item == "" or item.lower() == "nan":
      continue
    path = Path(item)
    if not path.is_absolute():
      path = (base_dir / path).resolve()
    wsi_files.append(path.as_posix())
  return wsi_files


def _resolve_out_path(out_dir: str, wsi_file: str, embedding_tag: str) -> str:
  out_dir_path = Path(out_dir)
  out_dir_path.mkdir(parents=True, exist_ok=True)
  stem = Path(wsi_file).stem
  return (out_dir_path / f"{stem}.wsi_feat_{embedding_tag}.h5").as_posix()


def _file_md5(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
  digest = hashlib.md5()
  with open(path, "rb") as fp:
    while True:
      chunk = fp.read(chunk_size)
      if not chunk:
        break
      digest.update(chunk)
  return digest.hexdigest()


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


def _build_encoder_engines(
  encoder_list: Sequence[str],
  devices: Sequence[int],
  slide_encoder_name: Optional[str],
) -> Dict[str, Dict[int, FeatureExtractor]]:
  model_engines: Dict[str, Dict[int, FeatureExtractor]] = {}
  for encoder_name in encoder_list:
    model_engines[encoder_name] = {}
    for device_id in devices:
      model_engines[encoder_name][device_id] = build_feature_extractor(
        encoder_name,
        slide_encoder_name,
        device_id=device_id,
      )
  return model_engines


def _write_h5(
  out_path: str,
  *,
  wsi_file: str,
  wsi_md5: str,
  tissue: Optional[np.ndarray],
  region_np: np.ndarray,
  attention_np: Optional[np.ndarray],
  encoder_list: Sequence[str],
  encoder_feature_dict: Dict[str, np.ndarray],
) -> None:
  string_dtype = h5py.string_dtype(encoding="utf-8")
  with h5py.File(out_path, "w") as fp:
    fp.attrs["wsi_file"] = wsi_file
    fp.attrs["wsi_md5"] = wsi_md5
    fp.attrs["patch_num"] = int(region_np.shape[0])
    fp.attrs.create("encoder_list", np.asarray(list(encoder_list), dtype=object), dtype=string_dtype)

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
  return


def main(config: CreateFeatureConfig) -> None:
  embedding_tag = config.tag
  wsi_files = _read_wsi_files(config.input_file, config.wsi_col)
  if len(wsi_files) == 0:
    raise ValueError("No wsi files found in input.")

  devices = parse_devices(config.devices)
  if len(devices) == 0:
    raise ValueError("No devices specified. Use --devices 0 or --devices 0,1,...")

  encoder_list = _parse_encoder_list(config)
  model_engines = _build_encoder_engines(encoder_list, devices, config.slide_encoder)
  max_workers = len(devices) if config.max_futs <= 0 else config.max_futs

  pbar = tqdm(wsi_files)
  for wsi_file in pbar:
    pbar.set_description(
      f"Inference {Path(wsi_file).name} as tag: '{embedding_tag}' by '{','.join(encoder_list)}'"
    )
    out_path = _resolve_out_path(config.out_dir, wsi_file, embedding_tag)
    if config.skip_existing and osp.exists(out_path):
      continue

    logger.info("Get tissue region...")
    tissue_output = get_tissue_region(
      wsi_file,
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

    logger.info("Run feature extraction...")
    encoder_feature_dict: Dict[str, np.ndarray] = {}
    for encoder_name in encoder_list:
      tasks = build_chunk_tasks(
        wsi_file=wsi_file,
        region_np=region_np,
        devices=devices,
        model_engines=model_engines[encoder_name],
      )
      encoder_feature_dict[encoder_name] = run_chunk_tasks(
        tasks=tasks,
        options=options,
        max_workers=max_workers,
      )

    wsi_md5 = _file_md5(wsi_file)
    _write_h5(
      out_path,
      wsi_file=wsi_file,
      wsi_md5=wsi_md5,
      tissue=tissue,
      region_np=region_np,
      attention_np=None,
      encoder_list=encoder_list,
      encoder_feature_dict=encoder_feature_dict,
    )

    if config.output is not None:
      _append_record_to_xlsx(
        config.output,
        wsi_file=wsi_file,
        wsi_md5=wsi_md5,
        h5_path=out_path,
        patch_num=int(region_np.shape[0]),
      )

  return
