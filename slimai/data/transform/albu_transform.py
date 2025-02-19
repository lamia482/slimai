import albumentations as A
import albumentations.pytorch as AP
from slimai.helper.help_build import TRANSFORMS
from .base_transform import BaseTransform


@TRANSFORMS.register_module()
class AlbuTransform(BaseTransform):
  albu_keys = ["image", "mask", "bboxes", "keypoints"]

  def __call__(self, data):  
    inp_data = {
      k: data[k] for k in self.albu_keys
      if k in data
    }
    out_data = self.transforms(**inp_data)
    data.update(out_data)
    return data
  
  def compose(self, transforms):
    return A.Compose(self._compose(
                          transforms=transforms, 
                          source=[A, AP], 
                          recursive_key="transforms"))
