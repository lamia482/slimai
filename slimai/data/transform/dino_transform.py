from slimai.helper.help_build import TRANSFORMS
from .base_transform import BaseTransform


@TRANSFORMS.register_module()
class DINOTransform(BaseTransform):
  dino_keys = ["image"]

  def __init__(self, *, 
               global_transforms, 
               global_ncrops=2, 
               local_transforms, 
               local_ncrops=8):
    super().__init__(transforms=dict(
      global_transforms=global_transforms, 
      global_ncrops=global_ncrops, 
      local_transforms=local_transforms, 
      local_ncrops=local_ncrops, 
    ))
    return
  
  def __call__(self, data):  
    inp_data = {
      k: data[k] for k in self.dino_keys
      if k in data
    }
    out_data = self.transforms(inp_data)
    data.update(out_data)
    return data
  
  def compose(self, transforms):
    global_transforms = transforms["global_transforms"]
    local_transforms = transforms["local_transforms"]
    global_ncrops = transforms["global_ncrops"]
    local_ncrops = transforms["local_ncrops"]

    global_transforms, local_transforms = self._compose(transforms=[
      global_transforms, 
      local_transforms, 
    ], source=[TRANSFORMS])
    
    def transform_wrap(data):
      image = data["image"]
      data["image"] = dict(
        global_views=[global_transforms(dict(image=image))["image"] for _ in range(global_ncrops)],
        local_views=[local_transforms(dict(image=image))["image"] for _ in range(local_ncrops)],
      )
      return data
    
    return transform_wrap
