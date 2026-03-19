import argparse
import hashlib
import concurrent.futures
import importlib
import os.path as osp
import sys
import cv2
import itertools
import pandas as pd
import h5py
import numpy as np
import torch
from loguru import logger
from typing import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from slimai.helper.shape import find_patch_region_from_mask, segment_foreground_mask
from sdk.reader import get_reader_by_ext, get_reader_by_file


accelerator = torch.device(0).type

class PatchDataset(Dataset):
  def __init__(self, wsi_file, coords, scale, patch_size, transform, to_gray=False):
    super().__init__()
    self.wsi_file = wsi_file
    self.coords = coords
    self.scale = scale
    self.patch_size = patch_size
    self.reader = None
    self.transform = transform
    self.to_gray = to_gray
    return

  def __len__(self):
    return len(self.coords)

  def __getitem__(self, index):
    if self.reader is None:
      self.reader = get_reader_by_file(self.wsi_file, scale=self.scale)
    assert (
      self.reader is not None
    ), "reader is not initialized"
    region = self.coords[index]
    if len(region) == 2:
      x, y, w, h, scale = *region, self.patch_size, self.patch_size, self.reader.getReadScale()
    elif len(region) == 4:
      x, y, w, h, scale = *region, self.reader.getReadScale()
    elif len(region) == 5:
      x, y, w, h, scale = region
    else:
      raise ValueError(f"Invalid region: {region}")

    patch = self.reader.ReadRoi(
      x, y, w, h, scale=scale,
    )
    assert (
      patch is not None
    ), "patch is not read"
    if self.to_gray:
      patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
      patch = cv2.merge([patch, patch, patch])
    patch = Image.fromarray(patch[..., ::-1])
    return self.transform(patch)

def build_inference_model(model_name, model_repo, *, device_id):
  MODEL_IMPORT_DICT = {
    "UNI": dict(
      repo="/hzztai/projects/research/aigc/jupyter/pathology_fms",
      module="uni",
      helper="UNIHelper",
      k_dim=1024,
    )
  }
  model_info: dict = MODEL_IMPORT_DICT[model_name]
  if model_repo is None:
    model_repo = model_info["repo"]
  if model_repo not in sys.path:
    sys.path.append(model_repo)
  ModelHelper = getattr(importlib.import_module(model_info["module"]), model_info["helper"])
  torch.get_device_module(accelerator).set_device(device_id) # bind device to accelerator
  logger.info(f"Build {model_name} model on device: {accelerator}:{device_id}")
  feature_extractor, transform = ModelHelper.build_model() # type: ignore
  return feature_extractor.to(f"{accelerator}:{device_id}").eval(), transform, model_info["k_dim"]

def get_tissue_region(wsi_file, *, read_scale=20, operate_scale=1.25, 
                      patch_size_h=224, patch_size_w=224, 
                      patch_stride_h=192, patch_stride_w=192,
                      min_ratio=0.05, kernel_size=5, iterations=3, 
                      shrink="tissue", return_vis=False, vis_scale=1.25):
  output = dict()
  wsi_reader = get_reader_by_file(wsi_file, scale=read_scale)
  if not wsi_reader.status:
    raise ValueError(f"Failed to read WSI file: {wsi_file}")

  # xy_arr in operate_scale, vis in operate_scale
  if shrink == "tissue":
    operate_wsi_image = wsi_reader.get_wsi(scale=operate_scale)
    # return mask&vis in operate_scale
    mask, vis = segment_foreground_mask(operate_wsi_image, return_vis=return_vis, 
                                        kernel_size=kernel_size, iterations=iterations)
    xy_arr = find_patch_region_from_mask(mask, min_ratio=min_ratio, 
                patch_size=(patch_size_h*operate_scale/read_scale, patch_size_w*operate_scale/read_scale), 
                patch_stride=(patch_stride_h*operate_scale/read_scale, patch_stride_w*operate_scale/read_scale) # type: ignore
              )[:, :2]
  else: # tissue-free, return wsi grid
    xy_arr = np.array(list(itertools.product(
      range(0, wsi_reader.getReadWidth(), patch_stride_w), 
      range(0, wsi_reader.getReadHeight(), patch_stride_h)
    ))).reshape(-1, 2)
    if return_vis:
      vis = wsi_reader.get_wsi(scale=operate_scale)

  output["coords"] = np.array([
    [x, y, patch_size_w, patch_size_h, read_scale]
    for x, y in xy_arr / operate_scale * read_scale]
  ).astype("float32") # commit coords to read_scale

  # resize to vis_scale to render
  if return_vis:
    vis_ratio = vis_scale / operate_scale
    vis = cv2.resize(cv2.merge([vis, vis, vis]), None, fx=vis_ratio, fy=vis_ratio) # type: ignore
    for (x, y) in (xy_arr * vis_ratio).astype("int"):
      cv2.rectangle(vis, (x, y), (
        x+int(patch_size_w*vis_scale/read_scale), y+int(patch_size_h*vis_scale/read_scale)
        ), (0, 255, 0), 2)
    output["vis"] = vis # commit vis to vis_scale
  return output

def run_inference(model, transform, wsi_file, coords, args, *, device):
  dataset = PatchDataset(
    wsi_file=wsi_file,
    coords=coords,
    scale=args.read_scale,
    patch_size=args.patch_size,
    transform=transform,
    to_gray=args.to_gray,
  )
  loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False,
    drop_last=False,
  )
  outputs = []
  for batch in tqdm(loader, desc=f"Infer<{Path(wsi_file).stem}> on {device}", leave=False):
    batch = batch.to(device)
    with torch.inference_mode():
      output = model(batch).cpu()
    outputs.append(output)
  return outputs

def _parse_devices(devices_str):
  if devices_str is None:
    devices_str = ""
  parts = [p.strip() for p in (devices_str or "").split(",") if p.strip() != ""]
  if len(parts) == 0:
    return list(range(torch.get_device_module(accelerator).device_count()))
  devices = []
  for p in parts:
    if not p.isdigit():
      raise ValueError(f"Invalid --devices value: {devices_str}")
    devices.append(int(p))
  return devices

def _read_wsi_files(input_file, wsi_col):
  input_path = Path(input_file)
  suffix = input_path.suffix.lower()

  try:
    get_reader_by_ext(suffix, strict=True)
    return [input_path.as_posix()]
  except:
    pass

  if suffix == ".csv":
    df = pd.read_csv(input_path)
  elif suffix in [".xlsx", ".xls"]:
    df = pd.read_excel(input_path)
  else:
    raise ValueError(f"Unsupported --input-file suffix: {suffix}")

  if wsi_col not in df.columns:
    raise ValueError(f"Column not found in input file: {wsi_col}. Available: {list(df.columns)}")

  base_dir = input_path.parent
  wsi_files = []
  for v in df[wsi_col].tolist():
    if v is None:
      continue
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
      continue
    p = Path(s)
    if not p.is_absolute():
      p = (base_dir / p).resolve()
    wsi_files.append(p.as_posix())
  return wsi_files

def _resolve_out_path(out_dir, wsi_file, embedding_tag):
  out_dir_path = Path(out_dir)
  out_dir_path.mkdir(parents=True, exist_ok=True)
  stem = Path(wsi_file).stem
  return (out_dir_path / f"{stem}.wsi_feat_{embedding_tag}.h5").as_posix()

def _worker_run(
  *,
  task,
  device_id,
  patch_size,
  read_scale,
  batch_size,
  num_workers,
  to_gray,
):
  if torch.get_device_module(accelerator).is_available():
    torch.get_device_module(accelerator).set_device(device_id)
    device = torch.device(f"{accelerator}:{device_id}")
  else:
    device = torch.device("cpu")
  args = SimpleNamespace(
    read_scale=read_scale,
    patch_size=patch_size,
    batch_size=batch_size,
    num_workers=num_workers,
    to_gray=to_gray,
  )

  outputs = run_inference(
    task.model, task.transform, task.wsi_file, task.coords_chunk, args, device=device
  )
  return torch.cat(outputs, dim=0).numpy() if len(outputs) > 0 else np.zeros([0, task.k_dim])

def parse_args():
  parser = argparse.ArgumentParser(description="Create WSI embedding h5 files.")
  parser.add_argument("--input-file", required=True, help="WSI file or Path to input csv/xlsx.")
  parser.add_argument("--wsi-col", dest="wsi_col", default="wsi_file", help="Column name for wsi file path.")
  parser.add_argument(
    "--model-name",
    required=True,
    choices=["UNI", "CytoKD", "CervicalKF"],
    help="Foundation model name.",
  )
  parser.add_argument("--model-repo", default=None, help="External repository path for model helper.")
  parser.add_argument("--tag", required=True, help="User-specified embedding tag.")
  parser.add_argument("--out-dir", required=True, help="Output directory (flat outputs).")
  parser.add_argument("--devices", default=None, help="Comma-separated device ids, e.g. 0,1,2,3")
  parser.add_argument("--max-futs", type=int, default=0, help="Max pending futures. 0 means auto.")
  parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size.")
  parser.add_argument("--num-workers", type=int, default=2, help="Data loader workers.")
  parser.add_argument("--patch-size", type=int, default=224, help="Patch size.")
  parser.add_argument("--stride-size", type=int, default=192, help="Stride when reset-coords.")
  parser.add_argument("--read-scale", type=float, default=20, help="Read scale for WSI.")
  parser.add_argument("--operate-scale", type=float, default=1.25, help="Operate scale for tissue shrink.")
  parser.add_argument("--to-gray", action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--min-tissue-ratio", type=float, default=0.05, help="Minimum tissue ratio for tissue shrink.")
  parser.add_argument("--tissue-shrink", type=str, default="tissue", help="Tissue shrink method.")
  return parser.parse_args()

@dataclass(frozen=True)
class _ChunkTask:
  wsi_file: str
  coords_chunk: np.ndarray
  chunk_index: int
  model: torch.nn.Module
  transform: Callable
  k_dim: int

def main():
  args = parse_args()

  embedding_tag = args.tag
  if args.to_gray:
    embedding_tag = embedding_tag + "_GRAY"

  # extract wsi file list from input file or excel file
  wsi_files = _read_wsi_files(args.input_file, args.wsi_col)
  if len(wsi_files) == 0:
    raise ValueError("No wsi files found in input.")

  # parse devices like "0,1,2,3"
  devices = _parse_devices(args.devices)
  if len(devices) == 0:
    raise ValueError("No devices specified. Use --devices 0 or --devices 0,1,...")
  model_engines = []
  for device_id in devices:
    model, transform, k_dim = build_inference_model(args.model_name, args.model_repo, device_id=device_id)
    model_engines.append(dict(model=model, transform=transform, k_dim=k_dim))

  pbar = tqdm(wsi_files)
  for wsi_file in wsi_files:
    pbar.set_description("Inference {} as tag: '{}' by {}".format(osp.basename(wsi_file), embedding_tag, args.model_name))
    out_path = _resolve_out_path(args.out_dir, wsi_file, embedding_tag)
    if args.skip_existing and osp.exists(out_path):
      pbar.update()
      continue

    # get tissue region
    logger.info(f"Get tissue region...")
    output = get_tissue_region(
      wsi_file,
      read_scale=args.read_scale,
      operate_scale=args.operate_scale,
      patch_size_h=args.patch_size,
      patch_size_w=args.patch_size,
      patch_stride_h=args.stride_size,
      patch_stride_w=args.stride_size,
      min_ratio=args.min_tissue_ratio,
      shrink=args.tissue_shrink,
      return_vis=True,
      vis_scale=1.25,
    )

    # build chunk tasks
    logger.info(f"Build chunk tasks...")
    buckets = dict()
    coords, vis = output["coords"], output["vis"]
    chunks = [c for c in np.array_split(coords, len(devices)) if len(c) > 0]
    for chunk_index, coords_chunk in enumerate(chunks):
      device_id = chunk_index % len(devices)
      buckets[device_id] = _ChunkTask(
        wsi_file=wsi_file, coords_chunk=coords_chunk, chunk_index=chunk_index, 
        **model_engines[device_id])

    # submit tasks
    logger.info(f"Submit tasks...")
    futures = []
    ctx = torch.multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as ex:
      for device_id, device_task in buckets.items():
        fut = ex.submit(_worker_run,
          task=device_task,
          device_id=device_id,
          patch_size=args.patch_size,
          read_scale=args.read_scale,
          batch_size=args.batch_size,
          num_workers=args.num_workers,
          to_gray=args.to_gray,
        )
        futures.append(fut)

    # collect results
    logger.info(f"Collect results...")
    embeddings = []
    for fut in futures:
      embeddings.append(fut.result())
    embeddings = np.concatenate(embeddings, axis=0)

    # write to h5 file
    logger.info(f"Write to h5 file...")
    with h5py.File(out_path, "w") as fp:
      fp.attrs["wsi_file"] = wsi_file
      fp.attrs["wsi_md5"] = hashlib.md5(wsi_file.encode()).hexdigest()
      fp.attrs["embedding_model"] = args.model_name
      fp.attrs["embedding_tag"] = embedding_tag
      fp.create_dataset("tissue", data=vis, dtype=np.uint8)
      fp.create_dataset("embeddings", data=embeddings, dtype=np.float32)
      fp.create_dataset("coords", data=coords, dtype=np.float32)
      fp.create_dataset("attentions", data=np.zeros(0), dtype=np.float32)

    pbar.update()

  return

if __name__ == "__main__":

  if len(sys.argv) == 1:
    sys.argv.extend([
      "--input-file", "/.slimai/cache/wsi-group-in-multiple-formats/goods/B2025049786A.sdpc", 
      "--model-name", "UNI", 
      "--tag", "debug", 
      "--out-dir", "/hzztai/slimai/_debug_/rst", 
      "--devices", "0",
      "--operate-scale", "1.25",
    ])

  main()
