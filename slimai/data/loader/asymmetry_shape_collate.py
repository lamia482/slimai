import torch
from torch.utils.data import default_collate
from slimai.helper.help_build import LOADERS


@LOADERS.register_module()
class AsymmetryShapeCollate():
  image_key = "image"

  def __init__(self):
    return
  
  def __call__(self, original_batch):
    assert (
      isinstance(original_batch, list) and len(original_batch) > 0 and 
      all(map(lambda x: isinstance(x, dict), original_batch)) and 
      all(map(lambda x: original_batch[0].keys() == x.keys(), original_batch))
    ), "original_batch must be a list of dicts with the same keys"

    keys = original_batch[0].keys()
    assert (
      self.image_key in keys
    ), "image_key must be in the keys of the batch"
    
    images = [v.pop(self.image_key) for v in original_batch]
    data = default_collate(original_batch)
    data[self.image_key] = images
    return data