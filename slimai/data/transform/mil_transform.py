import numpy as np
import torch
from slimai.helper.help_build import TRANSFORMS
from .base_transform import BaseTransform


@TRANSFORMS.register_module()
class MILTransform(BaseTransform):
  mil_keys = ["image", "label"]

  def __init__(self, *args, topk, shuffle=False, **kwargs):
    super().__init__(*args, **kwargs)
    self.topk = topk
    assert (
      isinstance(self.topk, int)
    ), "topk must be an integer, but got {}".format(self.topk)
    self.shuffle = shuffle
    return
  
  def __call__(self, data):  
    inp_data = {
      k: data[k] for k in self.mil_keys
      if k in data
    }
    inp_images = inp_data["image"]
    # we want MIL not sensitive to the order of tiles
    if self.shuffle:
      np.random.shuffle(inp_images)

    if self.topk <= 0:
      filled_topk_images = inp_images
    else:
      topk_images = inp_images[:self.topk]
      if len(topk_images) < self.topk:
        indices = np.random.choice(len(topk_images), self.topk - len(topk_images), replace=True)
        filled_topk_images = topk_images + [topk_images[i] for i in indices]
      else:
        filled_topk_images = topk_images

    # produce output images in (N, C, H, W)
    out_images = list(map(lambda x: self.transforms(dict(image=x))["image"], filled_topk_images))
    stack_tensor = torch.stack(out_images, dim=0)
    data.update(dict(image=stack_tensor))
    return data
  
  def compose(self, transforms):
    return self._compose(transforms=transforms, source=[TRANSFORMS])
