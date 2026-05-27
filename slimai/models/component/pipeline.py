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
              return_flow: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
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

  def _export_submodule(self, module: torch.nn.Module, name: str) -> torch.nn.Module:
    if not hasattr(module, "export_model"):
      raise ValueError(f"Pipeline {name} {type(module).__name__} does not implement export_model().")
    return module.export_model()

  def export_model(self) -> torch.nn.Module:
    return _PipelineExport(self).eval()


class _PipelineExport(torch.nn.Module):
  def __init__(self, source: Pipeline):
    super().__init__()
    self.backbone = source._export_submodule(source.backbone, "backbone")
    self.neck = source._export_submodule(source.neck, "neck")
    self.head = source._export_submodule(source.head, "head")
    return

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.head(self.neck(self.backbone(x)))
