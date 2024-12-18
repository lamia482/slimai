import torch
from .component import BaseComponent, MODELS


@MODELS.register_module()
class ViT(BaseComponent):
  def __init__(self, 
               arch="vit_base", 
               patch_size=16, 
               img_size=224, 
               use_lora=False, 
               pretrained_weight=None, 
               **kwargs):
    super().__init__(pretrained_weight)
    return
  
  def forward(self, x):
    pass
