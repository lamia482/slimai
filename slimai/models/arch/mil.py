import torch
from typing import Union, Dict
from slimai.helper import help_utils
from slimai.helper.help_build import MODELS
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
               ):
    super().__init__(encoder=encoder, decoder=decoder, loss=loss, solver=solver)
    help_utils.PytorchNetworkUtils.freeze(self.model.backbone)
    self.embedding_group_size = embedding_group_size
    return
  
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
    help_utils.print_log(f"Start forward backbone with B={len(batch_data)}, GS={self.embedding_group_size}")
    backbone = list(map(forward_backbone, batch_data)) # (B, ~N, D)
    help_utils.print_log("Embedding gathered.")
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
  
  def export_model(self) -> torch.nn.Module:
    # Export model for inference and export to onnx
    teacher_without_ddp = help_utils.PytorchNetworkUtils.get_module(self.model.teacher)
    backbone = teacher_without_ddp.backbone
    return backbone
  
  def postprocess(self, 
                  batch_data: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                  batch_info: DataSample) -> DataSample:
    # Postprocess the output by assigning it to batch_info
    batch_info.output = batch_data
    return batch_info
