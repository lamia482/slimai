import itertools
import cv2
import numpy as np
import torch
from .utils import scale_image


def segment_foreground_mask(image, 
                            speed_up: float=1.0, 
                            kernel_size: int=5, 
                            iterations: int=3, 
                            return_vis: bool=False):
  """
  Args:
    wsi: numpy like or tensor like image
    speed_up: float, speed up the process by this ratio
    kernel_size: int, the size of the kernel
    iterations: int, the number of iterations
    return_vis: bool, whether to return the visualization image

  Returns:
    mask: numpy like or tensor like image, in dtype=bool
  """
  ratio = 1/speed_up

  wsi = scale_image.to_batch_numpy_image(image)[0] # make it to [H, W, ?C] in dtype=uint8
  gray = cv2.resize(cv2.cvtColor(wsi, cv2.COLOR_RGB2GRAY), None, fx=ratio, fy=ratio)
  _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  
  kernel = np.ones((kernel_size, kernel_size), np.uint8)
  mask = cv2.dilate(binary, kernel, iterations=iterations)
  mask = cv2.erode(mask, kernel, iterations=iterations)

  vis = None
  if return_vis:
    vis = cv2.addWeighted(gray, 0.5, mask, 0.5, 0)
  
  mask = cv2.resize(mask, wsi.shape[:2][::-1])
  mask = mask.astype("bool")

  if isinstance(image, torch.Tensor):
    mask = torch.as_tensor(mask, device=image.device)

  if return_vis:
    return mask, vis
  return mask


def segment_background_mask(wsi, speed_up=1, kernel_size=5, iterations=3):
  return ~segment_foreground_mask(wsi, speed_up, kernel_size, iterations, return_vis=False)

def find_patch_region_from_mask(foregroun_mask, 
                                patch_size, 
                                patch_stride=None, 
                                min_foreground_ratio=0.1):
  """
  Args:
    foregroun_mask: numpy like or tensor like image, in dtype=bool
    patch_size: int or tuple, the size of the patch in (H, W)
    patch_stride: int or tuple, the stride of the patch in (H, W)

  Returns:
    patch_region: list of tuple, each tuple is (x, y, w, h)
  """
  if not isinstance(patch_size, (tuple, list)):
    patch_size = (patch_size, patch_size)
  patch_height, patch_width = patch_size

  if patch_stride is None:
    patch_stride = patch_size

  if not isinstance(patch_stride, (tuple, list)):
    patch_stride = (patch_stride, patch_stride)
  patch_stride_height, patch_stride_width = patch_stride

  foregroun_mask = foregroun_mask * 1.0
  mask_height, mask_width = foregroun_mask.shape[:2]
  patch_region = []
  mask_xy_list = list(itertools.product(
    range(0, mask_width, patch_stride_width), 
    range(0, mask_height, patch_stride_height)
  ))
  for x, y in mask_xy_list:
    m = foregroun_mask[y:y+patch_height, x:x+patch_width]
    if m.mean() >= min_foreground_ratio:
      patch_region.append((x, y, patch_width, patch_height))

  patch_region = np.array(patch_region).reshape(-1, 4)
  
  return patch_region
