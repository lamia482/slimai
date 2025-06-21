import numpy as np
from PIL import Image
from torchvision.transforms import v2 as T
from slimai.helper.help_build import TRANSFORMS
from .base_transform import BaseTransform


@TRANSFORMS.register_module()
class TorchTransform(BaseTransform):
  image_key = "image"
  ann_keys = ["label", "mask", "instance", "text"]

  def __call__(self, data):
    img_data = data[self.image_key]
    ann_data = {
      k: data[k] for k in self.ann_keys
      if k in data.keys()
    }

    assert (
      len(ann_data) <= 1
    ), f"Only one annotation key is supported, but got: {list(ann_data.keys())}"

    has_ann = (len(ann_data) > 0)

    if has_ann:
      inp_data = (img_data, *ann_data.values())
      t_img, *t_ann = self.transforms(*inp_data)
      for k, v in zip(ann_data.keys(), t_ann):
        data[k] = v
    else:
      t_img = self.transforms(img_data)

    data[self.image_key] = t_img
    
    return data
  
  def compose(self, transforms):
    return T.Compose(self._compose(
                          transforms=transforms, 
                          source=[T, TRANSFORMS], 
                          recursive_key="transforms"))

