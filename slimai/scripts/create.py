import argparse
import glob
import importlib
import os
import os.path as osp
import sys
from pathlib import Path

import h5py
import mmengine
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Script lives at slimai/scripts/create.py; repo root is parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
  sys.path.insert(0, REPO_ROOT.as_posix())

from slimai.models.feature import apply_lora_to_vit


K_DIM_DICT = {
  "UNI": 1024,
  "CytoKD": 768,
  "CervicalKF": 1152,
}

MODEL_IMPORT_DICT = {
  "UNI": ("uni", "UNIHelper"),
  "CytoKD": ("cyto_helper", "CytoKDHelper"),
  "CervicalKF": ("kfd_helper", "KFDHelper"),
}


def get_reader_by_file(file_path, scale):
  try:
    from sdk.reader import get_reader_by_file as _get_reader_by_file
  except ImportError as ex:
    raise ImportError("`sdk.reader` is required for scripts/create.py") from ex
  return _get_reader_by_file(file_path, scale=scale)


class PatchDataset(Dataset):
  def __init__(self, kfb_file, coords, scale, patch_size, transform, to_gray=False):
    super().__init__()
    self.kfb_file = kfb_file
    self.coords = coords
    self.scale = scale
    self.reader = None
    self.transform = transform
    self.patch_size = patch_size
    self.to_gray = to_gray
    return

  def __len__(self):
    return len(self.coords)

  def __getitem__(self, index):
    if self.reader is None:
      self.reader = get_reader_by_file(self.kfb_file, scale=self.scale)
    x, y = self.coords[index]
    patch = self.reader.ReadRoi(
      x,
      y,
      self.patch_size,
      self.patch_size,
      scale=self.reader.getReadScale(),
    )
    if self.to_gray:
      import cv2
      patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)  # type: ignore
      patch = cv2.merge([patch, patch, patch])
    patch = Image.fromarray(patch[..., ::-1])
    return self.transform(patch)


def parse_lora_from_weight_name(weight_path):
  file_name = Path(weight_path).name
  for part in file_name.split("-"):
    if not part.startswith("SFT_LORA_"):
      continue
    values = part.split("_")[2:]
    if len(values) < 3:
      return None
    return int(float(values[0])), float(values[1]), float(values[2])
  return None


def build_model(model_name, model_repo, sft_weight=None):
  module_name, helper_name = MODEL_IMPORT_DICT[model_name]
  if model_repo is not None and model_repo not in sys.path:
    sys.path.append(model_repo)

  module = importlib.import_module(module_name)
  helper = getattr(module, helper_name)
  feature_extractor, transform = helper.build_model()
  k_dim = K_DIM_DICT[model_name]

  if sft_weight is not None:
    lora_args = parse_lora_from_weight_name(sft_weight)
    if lora_args is not None:
      feature_extractor = apply_lora_to_vit(feature_extractor, *lora_args)
    weight = torch.load(sft_weight, map_location="cpu")
    num_classes = weight[list(weight.keys())[-1]].shape[-1]
    model = torch.nn.Sequential(feature_extractor, torch.nn.Linear(k_dim, num_classes))
    model.load_state_dict(weight)
    feature_extractor = model[0]

  return feature_extractor, transform


def collect_h5_paths(dataset_file, keys):
  dataset = mmengine.load(dataset_file)
  path_list = []
  for key in keys:
    path_list.extend([v[0] for v in dataset["dataset"][key]])
  return sorted(path_list)


def collect_kfb_dict(kfb_root):
  kfb_files = glob.glob(osp.join(kfb_root, "**", "*.kfb"), recursive=True)
  return {Path(path).stem: path for path in tqdm(kfb_files, desc="Indexing kfb files")}


def resolve_new_h5_path(old_h5_path, embedding_tag):
  return old_h5_path.replace(".kfb_feat.h5", f".kfb_feat_{embedding_tag}.h5")


def get_coords(old_h5_path, kfb_file, *, reset_coords, stride_size, scale):
  if not reset_coords:
    with h5py.File(old_h5_path, "r") as fp:
      return fp["coords"][:]

  reader = get_reader_by_file(kfb_file, scale=scale)
  width, height = reader.getReadWidth(), reader.getReadHeight()
  coords = []
  for x in range(0, width, stride_size):
    for y in range(0, height, stride_size):
      coords.append([x, y])
  return np.array(coords)


def run_inference(model, transform, kfb_file, coords, args):
  dataset = PatchDataset(
    kfb_file=kfb_file,
    coords=coords,
    scale=args.scale,
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
  for batch in tqdm(loader, desc=f"Infer<{Path(kfb_file).stem}>", leave=False):
    batch = batch.to(args.device)
    with torch.inference_mode():
      output = model(batch).cpu()
    outputs.append(output)
  return torch.cat(outputs, dim=0).numpy() if len(outputs) > 0 else np.zeros([0, 0])


def parse_args():
  parser = argparse.ArgumentParser(description="Create THCA embedding h5 files.")
  parser.add_argument("--dataset-file", required=True, help="Path to THCA dataset pkl.")
  parser.add_argument(
    "--keys",
    nargs="+",
    default=["DEV-K1-TRAIN-TBS", "DEV-K1-VAL-TBS", "ET1-TEST-TBS", "ET2-TEST-TBS"],
    help="Dataset keys used to collect h5 paths.",
  )
  parser.add_argument("--kfb-root", required=True, help="Root directory of kfb files.")
  parser.add_argument(
    "--model-name",
    default="CervicalKF",
    choices=["UNI", "CytoKD", "CervicalKF"],
    help="Foundation model name.",
  )
  parser.add_argument("--model-repo", required=True, help="External repository path for model helper.")
  parser.add_argument("--sft-weight", default=None, help="Optional SFT model weight file.")
  parser.add_argument("--embedding-tag", default=None, help="Output embedding suffix tag.")
  parser.add_argument("--device", default="cuda:0", help="Device string.")
  parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size.")
  parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers.")
  parser.add_argument("--patch-size", type=int, default=224, help="Patch size.")
  parser.add_argument("--stride-size", type=int, default=112, help="Stride when reset-coords.")
  parser.add_argument("--scale", type=int, default=20, help="Reader scale.")
  parser.add_argument("--start-id", type=int, default=int(os.environ.get("START_ID", 0)))
  parser.add_argument("--end-id", type=int, default=-1, help="Exclusive end index. -1 means all.")
  parser.add_argument("--reset-coords", action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument("--to-gray", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
  return parser.parse_args()


def main():
  args = parse_args()

  embedding_tag = args.embedding_tag or args.model_name
  if args.sft_weight is not None:
    embedding_tag = embedding_tag + "_SFT"
  if args.reset_coords:
    embedding_tag = "WSI_" + embedding_tag
  if args.to_gray:
    embedding_tag = embedding_tag + "_GRAY"

  model, transform = build_model(args.model_name, args.model_repo, args.sft_weight)
  model = model.to(args.device)
  model.eval()

  h5_paths = collect_h5_paths(args.dataset_file, args.keys)
  kfb_map = collect_kfb_dict(args.kfb_root)

  end_id = len(h5_paths) if args.end_id < 0 else min(args.end_id, len(h5_paths))
  for i, old_h5_path in enumerate(tqdm(h5_paths, desc=f"Generating<{embedding_tag}>")):
    if not (args.start_id <= i < end_id):
      continue
    stem = Path(Path(old_h5_path).stem).stem
    kfb_file = kfb_map.get(stem, None)
    if kfb_file is None:
      continue

    new_h5_path = resolve_new_h5_path(old_h5_path, embedding_tag)
    if args.skip_existing and osp.exists(new_h5_path):
      continue

    coords = get_coords(
      old_h5_path=old_h5_path,
      kfb_file=kfb_file,
      reset_coords=args.reset_coords,
      stride_size=args.stride_size,
      scale=args.scale,
    )
    features = run_inference(model, transform, kfb_file, coords, args)
    with h5py.File(new_h5_path, "w") as fp:
      fp.create_dataset("features", data=features, dtype=np.float32)
      fp.create_dataset("coords", data=coords, dtype=np.float32)


if __name__ == "__main__":
  main()
