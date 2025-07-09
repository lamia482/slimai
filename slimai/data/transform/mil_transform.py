import numpy as np
import torch
import cv2
import itertools
import torch.nn.functional as F
from torchvision import tv_tensors
from slimai.helper.help_build import TRANSFORMS
from slimai.helper.shape import segment_foreground_mask, find_patch_region_from_mask
from .base_transform import BaseTransform


@TRANSFORMS.register_module()
class MILTransform(BaseTransform):
  mil_keys = ["image", "label"]

  def __init__(self, *args, 
               shrink=None, 
               tile_size, 
               tile_stride, 
               random_crop_patch_size=None, 
               random_crop_patch_num=None, 
               transform_schema="individual",
               topk=None, 
               shuffle=False, 
               padding_value=255, 
               **kwargs):
    """
    Brief:
      This transform is used to produce MIL views from a WSI image.
      It can produce tiles by sliding windows from input image(expect to be a WSI image in shape (H, W, C))
      then random crop n patches from each tile if needed, otherwise use tiles as views
      finally, transfroms views and stack them together as multiple instances

    Args:
      shrink: str, "tissue" or None
      tile_size: int
      tile_stride: int
      random_crop_patch_size: int or None
      random_crop_patch_num: int or None
      transform_schema: str, "individual" or "group"
      topk: int or None
      shuffle: bool
      padding_value: int
    """
    super().__init__(*args, **kwargs)
    self.shrink = shrink
    assert (
      self.shrink in [None, "tissue"]
    ), "shrink is expected to be one of [None, 'tissue'], but got: {}".format(self.shrink)

    self.tile_size = tile_size
    self.tile_stride = tile_stride
    
    self.random_crop_patch_size = random_crop_patch_size
    assert (
      self.random_crop_patch_size is None or (
        isinstance(self.random_crop_patch_size, int) and (0 < self.random_crop_patch_size < self.tile_size)
      )
    ), "random_crop_patch_size must be an integer between 0 and tile_size, but got {}".format(self.random_crop_patch_size)
    
    self.random_crop_patch_num = random_crop_patch_num or 0
    assert (
      self.random_crop_patch_num is None or (
        isinstance(self.random_crop_patch_num, int) and (self.random_crop_patch_num >= 0)
      )
    ), "random_crop_patch_num must be an integer greater than 0, but got {}".format(self.random_crop_patch_num)
    
    self.use_patch_as_view = (self.random_crop_patch_num > 0)

    self.transform_schema = transform_schema
    assert (
      self.transform_schema in ["individual", "group"]
    ), "transform_schema must be one of ['individual', 'group'], but got {}".format(self.transform_schema)

    self.topk = topk or 0
    assert (
      isinstance(self.topk, (int, float))
    ), "topk must be an integer/float, but got {}".format(self.topk)

    self.shuffle = shuffle
    self.padding_value = padding_value
    return
  
  def __call__(self, data):
    """ 
    1. produce tiles by sliding windows from input image(expect to be a WSI image in shape (H, W, C))
    2. then random crop n patches from each tile if needed, otherwise use tiles as views
    3. finally, transfroms views and stack them together as result
    """
    inp_data = {
      k: data[k] for k in self.mil_keys
      if k in data
    }

    wsi_image = inp_data["image"] # [C, H, W]
    assert (
      isinstance(wsi_image, tv_tensors.Image)
    ), f"wsi image is expected to be tv_tensors.Image, but got: {type(wsi_image)}"

    xy_list = self.process_shrink(wsi_image)

    tiles = []
    for (x, y) in xy_list:
      tile = wsi_image[..., y:y+self.tile_size, x:x+self.tile_size]
      tile = F.pad(tile, [0, self.tile_size-tile.shape[-1], 0, self.tile_size-tile.shape[-2]], 
                   mode="constant", value=255)
      tiles.append(tile)
        
    views = tiles
    if self.use_patch_as_view:
      views = []
      for tile in tiles:
        # Random crop patches from each tile
        patches = []
        for _ in range(self.random_crop_patch_num):
          # Random starting position for cropping
          x_start = np.random.randint(0, self.tile_size - self.random_crop_patch_size + 1)
          y_start = np.random.randint(0, self.tile_size - self.random_crop_patch_size + 1)
          # Extract the patch
          patch = tile[..., y_start:y_start+self.random_crop_patch_size, x_start:x_start+self.random_crop_patch_size]
          views.append(patch)

    # we want MIL not sensitive to the order of tiles
    if self.shuffle:
      np.random.shuffle(views)

    topk = len(views) if self.topk <= 0 else (
      self.topk if isinstance(self.topk, int) else max(1, int(self.topk * len(views)))
    )
    topk_views = views[:topk]
    topk_views = torch.stack(topk_views, dim=0) # (N, C, H, W)
    patches = tv_tensors.Image(topk_views) # (N, C, H, W)

    # produce output images in (N, C, H, W)
    if self.transform_schema == "individual":
      out_patches = torch.stack([self.transforms(dict(image=img))["image"] for img in patches], dim=0)
    elif self.transform_schema == "group":
      out_patches = self.transforms(dict(image=patches))["image"]
    else:
      raise ValueError(f"transform_schema must be one of ['individual', 'group'], but got {self.transform_schema}")

    data.update(dict(image=out_patches))

    # add extra info log
    data["meta"].update(dict(
      patch_num=len(out_patches), 
    ))
    return data

  def process_shrink(self, image):
    wsi_height, wsi_width = image.shape[1:] # (C, H, W)
    if self.shrink == "tissue":
      mask = segment_foreground_mask(image, speed_up=20/5, kernel_size=5, iterations=3)
      xy_list = find_patch_region_from_mask(mask, self.tile_size, self.tile_stride)
      xy_list = xy_list[:, :2].tolist()
    else:
      xy_list = list(itertools.product(
        range(0, wsi_width, self.tile_stride), 
        range(0, wsi_height, self.tile_stride)
      ))

    return xy_list
  
  def compose(self, transforms):
    return self._compose(transforms=transforms, source=[TRANSFORMS])
