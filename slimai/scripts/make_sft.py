import argparse
import glob
import json
import os
import os.path as osp
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def get_reader_by_file(file_path, scale):
  try:
    from sdk.reader import get_reader_by_file as _get_reader_by_file
  except ImportError as ex:
    raise ImportError("`sdk.reader` is required for scripts/make_sft.py") from ex
  return _get_reader_by_file(file_path, scale=scale)


def parse_args():
  parser = argparse.ArgumentParser(description="Create SFT patch dataset from annotation json.")
  parser.add_argument("--ann-root", required=True, help="Root folder of annotation json files.")
  parser.add_argument("--dst-data-dir", required=True, help="Directory to save output patches.")
  parser.add_argument("--vai-root", required=True, help="Root used to resolve json path to kfb path.")
  parser.add_argument("--json-encoding", default="gbk", help="Annotation file encoding.")
  parser.add_argument("--scale", type=int, default=20, help="Reader scale.")
  parser.add_argument("--tile-size", type=int, default=224, help="Tile crop size.")
  parser.add_argument("--pad-size", type=int, default=256, help="Padded output crop size.")
  return parser.parse_args()


def get_info(json_file, vai_root, json_encoding):
  with open(json_file, "r", encoding=json_encoding) as fp:
    json_data = json.load(fp)

  kfb_file = osp.join(vai_root, json_data["path"]).replace("\\", "/")
  anns = json_data["annotation"]
  subclass_list, polygon_list = zip(*[(elem["sub_class"], elem["points"]) for elem in anns])
  return kfb_file, subclass_list, polygon_list


def crop_region(raw_region_xywh, tile_size=224, pad_size=256):
  x, y, w, h = raw_region_xywh
  xywh_list = []

  if w < tile_size or h < tile_size:
    center_x = x + w // 2
    center_y = y + h // 2
    new_x = max(0, center_x - tile_size // 2)
    new_y = max(0, center_y - tile_size // 2)
    xywh_list.append((new_x, new_y, tile_size, tile_size))
  else:
    for yy in range(y, y + h - tile_size + 1, tile_size):
      for xx in range(x, x + w - tile_size + 1, tile_size):
        xywh_list.append((xx, yy, tile_size, tile_size))
    if (h % tile_size) != 0:
      for xx in range(x, x + w - tile_size + 1, tile_size):
        xywh_list.append((xx, y + h - tile_size, tile_size, tile_size))
    if (w % tile_size) != 0:
      for yy in range(y, y + h - tile_size + 1, tile_size):
        xywh_list.append((x + w - tile_size, yy, tile_size, tile_size))
    if (w % tile_size) != 0 and (h % tile_size) != 0:
      xywh_list.append((x + w - tile_size, y + h - tile_size, tile_size, tile_size))

  pad_xywh_list = []
  for xx, yy, tw, th in xywh_list:
    cx = xx + tw // 2
    cy = yy + th // 2
    px = max(0, cx - pad_size // 2)
    py = max(0, cy - pad_size // 2)
    pad_xywh_list.append((px, py, pad_size, pad_size))
  return pad_xywh_list


def prepare_patches(kfb_file, polygon, scale=20, tile_size=224, pad_size=256):
  reader = get_reader_by_file(kfb_file, scale=scale)
  polygon = (
    np.array(
      [
        v / reader.getScanScale() * reader.getReadScale()
        for v in np.array([(v["x"], v["y"]) for v in polygon])
      ]
    )
    + 0.5
  ).astype("int")
  region_xywh = cv2.boundingRect(polygon)
  region_xywh_list = crop_region(region_xywh, tile_size=tile_size, pad_size=pad_size)
  patch_list = [reader.ReadRoi(*xywh, scale=reader.getReadScale()) for xywh in region_xywh_list]
  return region_xywh_list, patch_list


def main():
  args = parse_args()
  json_files = sorted(glob.glob(osp.join(args.ann_root, "**", "*.json"), recursive=True))
  subclass_counter = Counter()

  for json_file in tqdm(json_files, desc="Processing annotation files"):
    kfb_file, subclass_list, polygon_list = get_info(
      json_file=json_file,
      vai_root=args.vai_root,
      json_encoding=args.json_encoding,
    )
    for ann_id, (subclass, polygon) in enumerate(zip(subclass_list, polygon_list)):
      if subclass == "ROI":
        continue
      subclass_counter[subclass] += 1
      xywh_list, patch_list = prepare_patches(
        kfb_file=kfb_file,
        polygon=polygon,
        scale=args.scale,
        tile_size=args.tile_size,
        pad_size=args.pad_size,
      )
      for patch_id, ([x, y, w, h], patch) in enumerate(zip(xywh_list, patch_list)):
        patch_file = osp.join(
          args.dst_data_dir,
          subclass,
          Path(json_file).stem,
          str(ann_id),
          f"{ann_id}-th_{patch_id}-th_{x}_{y}_{w}_{h}.png",
        )
        os.makedirs(osp.dirname(patch_file), exist_ok=True)
        cv2.imwrite(patch_file, patch)  # type: ignore

  print("Subclass distribution:")
  for subclass, count in subclass_counter.most_common():
    print(f"{subclass}: {count}")


if __name__ == "__main__":
  main()
