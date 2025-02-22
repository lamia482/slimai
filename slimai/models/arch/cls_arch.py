import torch
from slimai.helper.help_build import MODELS
from slimai.helper.structure import DataSample
from .base_arch import BaseArch


@MODELS.register_module()
class ClassificationArch(BaseArch):
  def export_model(self) -> torch.nn.Module:
    # Export model for inference and export to onnx
    return self.model
  
  def postprocess(self, 
                  batch_data: torch.Tensor, 
                  batch_info: DataSample) -> DataSample:
    # Postprocess the output by assigning it to batch_info
    logits = batch_data
    scores = torch.softmax(logits, dim=1)
    batch_info.output = scores
    return batch_info
