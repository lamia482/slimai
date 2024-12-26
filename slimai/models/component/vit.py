import torch
import torchvision.models as models
from slimai.helper.help_build import MODELS


@MODELS.register_module()
class ViT(torch.nn.Module):
  def __init__(self, 
               arch="base", 
               patch_size=16, 
               img_size=224, 
               use_lora=False, 
               pretrained_weight=None, 
               **kwargs):
    super().__init__()
    self.vit = getattr(models, f"vit_{arch[0]}_{patch_size}")(weights=pretrained_weight)
    return
  
  def forward(self, x):
    return self.vit(x)
