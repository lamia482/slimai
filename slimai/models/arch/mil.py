import torch
from typing import Union, Dict
from functools import partial
from slimai.helper import help_utils
from slimai.helper.help_build import MODELS, build_model
from slimai.helper.structure import DataSample
from .base_arch import BaseArch


__all__ = [
  "MIL",
]

@MODELS.register_module()
class MIL(BaseArch):
  def __init__(self, *, 
               encoder=dict(
                 backbone=None, neck=None, 
               ), 
               decoder=dict(
                 head=None, 
               ), 
               loss=None, 
               solver=None, 
               embedding_group_size=1, 
               freeze_backbone=False,
               ):
    super().__init__(encoder=encoder, decoder=decoder, loss=loss, solver=solver)
    if freeze_backbone:
      help_utils.print_log("Freezing backbone.")
      help_utils.PytorchNetworkUtils.freeze(self.model.backbone)
    self.embedding_group_size = embedding_group_size
    return

  def init_layers(self, encoder, decoder) -> torch.nn.Module:
    help_utils.print_log(
      f"Using default `init_layers` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    backbone = build_model(encoder.backbone)
    neck = build_model(encoder.neck)
    head = build_model(decoder.head)
    return torch.nn.ModuleDict(dict(backbone=backbone, neck=neck, head=head))
  
  def _forward_tensor(self, 
                batch_data: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                return_flow: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    def forward_backbone(images, group_size=self.embedding_group_size):
      if group_size <= 0:
        group_size = len(images)
      output = []
      for i in range(0, len(images), group_size):
        embedding = self.model.backbone(images[i:i+group_size])
        output.append(embedding)
      return torch.cat(output, dim=0)

    # batch_data in shape (B, ~N, C, H, W)
    backbone = list(map(forward_backbone, batch_data)) # (B, ~N, D)
    neck = self.model.neck(backbone) # (B, D)
    head = self.model.head(neck) # (B, C)

    if return_flow:
      return dict(
        backbone=backbone,
        neck=neck,
        head=head,
      )
    else:
      return head
  
  def postprocess(self, 
                  batch_data: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                  batch_info: DataSample) -> DataSample:
    # Postprocess the output by assigning it to batch_info
    batch_info.output = batch_data
    return batch_info
