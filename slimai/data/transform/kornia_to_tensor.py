import kornia
import numpy as np
from slimai.helper.help_build import TRANSFORMS
from slimai.helper.utils import dist_env


@TRANSFORMS.register_module()
class KorniaToTensor(object):
  def __init__(self, on_cpu=False):
    self.on_cpu = on_cpu
    return

  def __call__(self, image):  
    image = np.array(image)
    if image.dtype == "uint8":
      image = image.astype(np.float32) / 255.0
    image = kornia.utils.image_to_tensor(image)
    if not self.on_cpu:
      image = image.to(f"{dist_env.device}:{dist_env.local_rank}")
    return image
