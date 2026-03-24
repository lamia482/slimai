import argparse
import hashlib
import concurrent.futures
import os.path as osp
import sys
import cv2
import itertools
import pandas as pd
import h5py
import numpy as np
import torch
import timm
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
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

def build_encoder(patch_encoder_name, slide_encoder_name, *, device_id):
  PATCH_ENCODER_DICT = {
    "UNI": dict(
      name="hf_hub:MahmoodLab/UNI", 
      kwargs=dict(
        init_values=1e-5,
        dynamic_img_size=True,
      )
    ), 
    "UNI2": dict(
      name="hf_hub:MahmoodLab/UNI2-h", 
      kwargs=dict(
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667*2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked, # type: ignore
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
      )
    )
  }

  def _build_encoder_from_huggingface(name, kwargs, cache_dir="/.slimai/cache/huggingface/hub"):
    model = timm.create_model(name, pretrained=True, **kwargs, cache_dir=cache_dir) # type: ignore
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model.eval()
    num_features = model.num_features
    return model, transform, num_features

  patch_encoder, transform, k_dim = _build_encoder_from_huggingface(PATCH_ENCODER_DICT[patch_encoder_name]["name"], PATCH_ENCODER_DICT[patch_encoder_name]["kwargs"])
  slide_encoder = None

  patch_encoder = patch_encoder.to(f"{accelerator}:{device_id}").eval()
  if slide_encoder is not None:
    slide_encoder = slide_encoder.to(f"{accelerator}:{device_id}").eval()

  return patch_encoder, transform, k_dim, slide_encoder

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

def run_inference(patch_encoder, slide_encoder, transform, wsi_file, coords, args, *, device):
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
      output = patch_encoder(batch).cpu()
    outputs.append(output)
  if slide_encoder is not None:
    with torch.inference_mode():
      output = slide_encoder(torch.cat(outputs, dim=0).to(device)).cpu()
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

  *sheet_name, wsi_col = wsi_col.rsplit(":", 1)
  sheet_name = None if len(sheet_name) == 0 else sheet_name[0]

  if suffix in [".xlsx", ".xls"]:
    df = pd.read_excel(input_path, sheet_name=sheet_name)
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

def _append_record_to_xlsx(output_file, *, wsi_file, wsi_md5, h5_path):
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
    df = pd.DataFrame(columns=columns) # type: ignore

  df.loc[len(df)] = [wsi_file, wsi_md5, h5_path]
  df.to_excel(output_path, index=False)
  return

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
    task.patch_encoder, task.slide_encoder, task.transform, 
    task.wsi_file, task.coords_chunk, args, device=device
  )
  return torch.cat(outputs, dim=0).numpy() if len(outputs) > 0 else np.zeros([0, task.k_dim])

def parse_args():
  parser = argparse.ArgumentParser(description="Create WSI embedding h5 files.")
  parser.add_argument("--input-file", required=True, help="WSI file or Path to input xlsx/xls.")
  parser.add_argument("--wsi-col", dest="wsi_col", default="wsi_file", help="Column name for wsi file path.")
  parser.add_argument(
    "--patch-encoder", required=True, help="Patch encoder name.",
    choices=[
      "UNI", "UNI2", "CONCH", "CONCHV1_5", 
    ]
  )
  parser.add_argument(
    "--slide-encoder", required=False, help="Slide encoder name.",
    choices=[
      "TITAN", 
    ] 
  )
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
  parser.add_argument("--output", default=None, help="Path to xlsx record file, e.g. record.xlsx.")
  return parser.parse_args()

@dataclass(frozen=True)
class _ChunkTask:
  wsi_file: str
  coords_chunk: np.ndarray
  chunk_index: int
  slide_encoder: torch.nn.Module
  patch_encoder: torch.nn.Module
  transform: Callable
  k_dim: int

def main():
  args = parse_args()

  embedding_tag = args.tag

  # extract wsi file list from input file or excel file
  wsi_files = _read_wsi_files(args.input_file, args.wsi_col)
  if len(wsi_files) == 0:
    raise ValueError("No wsi files found in input.")

  # parse devices like "0,1,2,3"
  devices = _parse_devices(args.devices)
  if len(devices) == 0:
    raise ValueError("No devices specified. Use --devices 0 or --devices 0,1,...")
  model_engines = dict()
  for device_id in devices:
    patch_encoder, transform, k_dim, slide_encoder = build_encoder(args.patch_encoder, args.slide_encoder, device_id=device_id)
    model_engines[device_id] = dict(slide_encoder=slide_encoder, patch_encoder=patch_encoder, transform=transform, k_dim=k_dim)

  pbar = tqdm(wsi_files)
  for wsi_file in wsi_files:
    pbar.set_description("Inference {} as tag: '{}' by '{}+{}'".format(
      osp.basename(wsi_file), embedding_tag, args.patch_encoder, args.slide_encoder))
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
      device_id = devices[chunk_index % len(devices)]
      buckets[device_id] = _ChunkTask(
        wsi_file=wsi_file, coords_chunk=coords_chunk, chunk_index=chunk_index, 
        **model_engines[device_id])

    # submit tasks
    logger.info(f"Submit tasks...")
    futures = []
    with concurrent.futures.ThreadPoolExecutor(
      max_workers=len(devices), 
      # mp_context=torch.multiprocessing.get_context("spawn")
      ) as ex:
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
      with open(wsi_file, "rb") as f:
        wsi_md5 = hashlib.md5(f.read()).hexdigest()
      fp.attrs["wsi_md5"] = wsi_md5
      fp.attrs["patch_encoder"] = args.patch_encoder
      fp.attrs["slide_encoder"] = args.slide_encoder
      fp.attrs["patch_embedding_tag"] = embedding_tag
      fp.create_dataset("tissue", data=vis, dtype=np.uint8)
      fp.create_dataset("patch_coords", data=coords, dtype=np.float32)
      fp.create_dataset("patch_embeddings", data=embeddings, dtype=np.float32)
      fp.create_dataset("patch_attentions", data=np.zeros(0), dtype=np.float32)

    if args.output is not None:
      _append_record_to_xlsx(
        args.output,
        wsi_file=wsi_file,
        wsi_md5=wsi_md5,
        h5_path=out_path,
      )

    pbar.update()

  return

if __name__ == "__main__":

  if len(sys.argv) == 1:
    # sys.argv.extend([
    #   "--input-file", "/.slimai/cache/wsi-group-in-multiple-formats/goods/T2020-14513-20X-KFB.kfb", 
    #   "--patch-encoder", "UNI", 
    #   "--slide-encoder", "TITAN", 
    #   "--tag", "debug", 
    #   "--out-dir", "/hzztai/slimai/_debug_/rst", 
    #   "--devices", "1",
    #   "--operate-scale", "1.25",
    # ])
    sys.argv.extend([
      "--input-file", "/hzztai/slimai/_debug_/thca/embeddings/merged_dataset_by_center.xlsx", 
      "--wsi-col", "厦大中山:wsi_file", 
      "--patch-encoder", "UNI", 
      "--slide-encoder", "TITAN", 
      "--tag", "UNI_GRAY",
      "--to-gray", 
      "--out-dir", "/hzztai/slimai/_debug_/thca/embeddings/xiamen", 
      "--output", "/hzztai/slimai/_debug_/thca/embeddings/xiamen/record.xlsx", 
      "--devices", "1",
      "--operate-scale", "1.25",
    ])

  main()
