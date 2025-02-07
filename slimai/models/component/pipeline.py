import torch
from typing import Dict, Union
from slimai.helper import help_build


class Pipeline(torch.nn.Module):
  def __init__(self, backbone, neck, head):
    super().__init__()
    self.backbone = help_build.build_model(backbone)
    self.neck = help_build.build_model(neck)
    self.head = help_build.build_model(head)
    return
  
  def forward(self, batch_data: torch.Tensor, 
              return_flow: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    backbone = self.backbone(batch_data)
    neck = self.neck(backbone)
    head = self.head(neck)
    if return_flow:
      output = dict(
        backbone=backbone, 
        neck=neck, 
        head=head, 
      )
    else:
      output = head
    return output
  