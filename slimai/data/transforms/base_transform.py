from abc import ABC, abstractmethod
from slimai.helper import help_build


class BaseTransform(ABC, object):
  def __init__(self, transforms, **kwargs):
    self.transforms = self.compose(transforms)
    return

  @abstractmethod
  def __call__(self, data):
    """Apply transforms to data"""
    raise NotImplementedError
  
  @abstractmethod
  def compose(self, transforms):
    """Compose transforms by cls._compose"""
    raise NotImplementedError
  
  @classmethod
  def _compose(cls, *, transforms, source=None):
    return help_build.compose_components(transforms, 
                                         source=source, 
                                         recursive_key="transforms")
  
  def __repr__(self):
    return str(self.transforms)
