import numpy as np
import torch
from slimai.helper.help_build import TRANSFORMS
from .base_transform import BaseTransform


@TRANSFORMS.register_module()
class MILTransform(BaseTransform):
  mil_keys = ["image", "label"]

  def __init__(self, *args, 
               tile_size, 
               tile_stride, 
               random_crop_patch_size=None, 
               random_crop_patch_num=None, 
               topk=0, 
               shuffle=False, 
               padding_value=255, 
               **kwargs):
    super().__init__(*args, **kwargs)
    self.tile_size = tile_size
    self.tile_stride = tile_stride
    self.random_crop_patch_size = random_crop_patch_size
    assert (
      self.random_crop_patch_size is None or (
        isinstance(self.random_crop_patch_size, int) and (0 < self.random_crop_patch_size < self.tile_size)
      )
    ), "random_crop_patch_size must be an integer between 0 and tile_size, but got {}".format(self.random_crop_patch_size)
    self.random_crop_patch_num = random_crop_patch_num
    assert (
      self.random_crop_patch_num is None or (
        isinstance(self.random_crop_patch_num, int) and (self.random_crop_patch_num > 0)
      )
    ), "random_crop_patch_num must be an integer greater than 0, but got {}".format(self.random_crop_patch_num)
    self.use_patch_as_view = (self.random_crop_patch_size is not None)
    self.topk = topk
    assert (
      isinstance(self.topk, int)
    ), "topk must be an integer, but got {}".format(self.topk)
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

    wsi_image = inp_data["image"]
    wsi_height, wsi_width = wsi_image.shape[:2]
    tiles = []
    for x in range(0, wsi_width, self.tile_stride):
      for y in range(0, wsi_height, self.tile_stride):
        tile = wsi_image[y:y+self.tile_size, x:x+self.tile_size]
        if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
          tile = np.pad(tile, ((0, self.tile_size-tile.shape[0]), (0, self.tile_size-tile.shape[1]), (0, 0)), mode="constant", constant_values=self.padding_value)
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
          patch = tile[y_start:y_start+self.random_crop_patch_size, x_start:x_start+self.random_crop_patch_size]
          views.append(patch)

    # we want MIL not sensitive to the order of tiles
    if self.shuffle:
      np.random.shuffle(views)

    topk = len(views) if self.topk <= 0 else self.topk
    topk_views = views[:topk]

    # produce output images in (N, C, H, W)
    out_images = list(map(lambda x: self.transforms(dict(image=x))["image"], topk_views))
    stack_tensor = torch.stack(out_images, dim=0)
    data.update(dict(image=stack_tensor))
    return data
  
  def compose(self, transforms):
    return self._compose(transforms=transforms, source=[TRANSFORMS])
