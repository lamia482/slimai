import albumentations as A
import albumentations.pytorch as AP
from slimai.helper.help_build import TRANSFORMS
from .base_transform import BaseTransform

@TRANSFORMS.register_module()
class AlbuTransform(BaseTransform):
  def __call__(self, data):
    data = self.transforms(**data)
    return data
  
  def compose(self, transforms):
    return A.Compose(self._compose(
                          transforms=transforms, 
                          libs=[A, AP]))
