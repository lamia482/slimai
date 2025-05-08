import numpy as np
from PIL import Image
from torchvision import transforms as T
from slimai.helper.help_build import TRANSFORMS
from .base_transform import BaseTransform


@TRANSFORMS.register_module()
class TorchTransform(BaseTransform):
  image_key = "image"

  def __call__(self, data):  
    inp_data = data[self.image_key]
    if isinstance(inp_data, np.ndarray):
      inp_data = Image.fromarray(inp_data)
    out_data = self.transforms(inp_data)
    data[self.image_key] = out_data
    return data
  
  def compose(self, transforms):
    return T.Compose(self._compose(
                          transforms=transforms, 
                          source=[T, TRANSFORMS], 
                          recursive_key="transforms"))

