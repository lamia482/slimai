from abc import ABC, abstractmethod


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
  def _compose(cls, *, transforms, libs):
    if not isinstance(transforms, (tuple, list)):
      transforms = [transforms]
      
    def _recursive_compose(transform):
      if isinstance(transform, (tuple, list)):
        return list(map(_recursive_compose, transform))
      
      assert (
        isinstance(transform, dict) and "type" in transform
      ), "Transform must be a dict with 'type' key, but got: {}".format(transform)

      transform = transform.copy()

      transform_type = transform.pop("type")
      for lib in libs:
        transform_cls = getattr(lib, transform_type, None)
        if transform_cls is not None:
          break
      assert (
        transform_cls is not None
      ), "Transform: {} not found.".format(transform_type)

      inner_transforms = transform.pop("transforms", None)

      # no transforms means it self is a basic transform
      if inner_transforms is None:
        return transform_cls(**transform)
      
      if not isinstance(inner_transforms, (list, tuple)):
        inner_transforms = [inner_transforms]
      
      return transform_cls(_recursive_compose(inner_transforms))
    
    return list(map(_recursive_compose, transforms))
  
  def __repr__(self):
    return str(self.transforms)
