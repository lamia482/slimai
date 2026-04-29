from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from sdk.reader import get_reader_by_file
from slimai.helper.shape import find_patch_region_from_mask, segment_foreground_mask


@dataclass(frozen=True)
class TissueRegionOutput:
  region_np: np.ndarray
  tissue: Optional[np.ndarray]


class PatchDataset(Dataset):
  def __init__(
    self,
    wsi_file: str,
    coords: np.ndarray,
    scale: float,
    patch_size: int,
    transform: Callable,
    to_gray: bool = False,
  ):
    super().__init__()
    self.wsi_file = wsi_file
    self.coords = coords
    self.scale = scale
    self.patch_size = patch_size
    self.transform = transform
    self.to_gray = to_gray
    self.reader = None
    return

  def __len__(self) -> int:
    return len(self.coords)

  def __getitem__(self, index: int):
    if self.reader is None:
      self.reader = get_reader_by_file(self.wsi_file, scale=self.scale)
    assert self.reader is not None, "reader is not initialized"

    region = self.coords[index]
    if len(region) == 2:
      x, y, w, h, magnification = (
        *region,
        self.patch_size,
        self.patch_size,
        self.reader.getReadScale(),
      )
    elif len(region) == 4:
      x, y, w, h, magnification = (*region, self.reader.getReadScale())
    elif len(region) == 5:
      x, y, w, h, magnification = region
    else:
      raise ValueError(f"Invalid region: {region}")

    patch = self.reader.ReadRoi(x, y, w, h, scale=magnification)
    assert patch is not None, "patch is not read"

    if self.to_gray:
      patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
      patch = cv2.merge([patch, patch, patch])

    patch = Image.fromarray(patch[..., ::-1])
    return self.transform(patch)


def get_tissue_region(
  wsi_file: str,
  *,
  read_scale: float = 20,
  operate_scale: float = 1.25,
  patch_size_h: int = 224,
  patch_size_w: int = 224,
  patch_stride_h: int = 192,
  patch_stride_w: int = 192,
  min_ratio: float = 0.05,
  kernel_size: int = 5,
  iterations: int = 3,
  shrink: str = "tissue",
  return_tissue: bool = False,
  tissue_scale: float = 1.25,
) -> TissueRegionOutput:
  wsi_reader = get_reader_by_file(wsi_file, scale=read_scale)
  if not wsi_reader.status:
    raise ValueError(f"Failed to read WSI file: {wsi_file}")

  tissue = None
  if shrink == "tissue":
    operate_wsi_image = wsi_reader.get_wsi(scale=operate_scale)
    mask, vis = segment_foreground_mask(
      operate_wsi_image,
      return_vis=return_tissue,
      kernel_size=kernel_size,
      iterations=iterations,
    )
    xy_arr = find_patch_region_from_mask(
      mask,
      min_ratio=min_ratio,
      patch_size=(
        patch_size_h * operate_scale / read_scale,
        patch_size_w * operate_scale / read_scale,
      ),
      patch_stride=(
        patch_stride_h * operate_scale / read_scale, # type: ignore
        patch_stride_w * operate_scale / read_scale, # type: ignore
      ),
    )[:, :2]
  else:
    xy_arr = np.array(
      list(
        itertools.product(
          range(0, wsi_reader.getReadWidth(), patch_stride_w),
          range(0, wsi_reader.getReadHeight(), patch_stride_h),
        )
      )
    ).reshape(-1, 2)
    if return_tissue:
      vis = wsi_reader.get_wsi(scale=operate_scale)

  region_np = np.array(
    [
      [x, y, patch_size_w, patch_size_h, read_scale]
      for x, y in (xy_arr / operate_scale * read_scale)
    ]
  ).astype(np.float32)

  if return_tissue:
    vis_ratio = tissue_scale / operate_scale
    tissue = cv2.resize(cv2.merge([vis, vis, vis]), None, fx=vis_ratio, fy=vis_ratio)  # type: ignore
    for (x, y) in (xy_arr * vis_ratio).astype("int"):
      cv2.rectangle(
        tissue,
        (x, y),
        (
          x + int(patch_size_w * tissue_scale / read_scale),
          y + int(patch_size_h * tissue_scale / read_scale),
        ),
        (0, 255, 0),
        2,
      )

  return TissueRegionOutput(region_np=region_np, tissue=tissue)


def get_tissue_region_dict(**kwargs) -> Dict[str, Optional[np.ndarray]]:
  output = get_tissue_region(**kwargs)
  return {
    "region_np": output.region_np,
    "tissue": output.tissue,
  }
