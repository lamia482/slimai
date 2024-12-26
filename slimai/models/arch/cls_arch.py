import torch
from slimai.helper.help_build import MODELS
from slimai.helper.structure import DataSample
from .base_arch import BaseArch


@MODELS.register_module()
class ClassificationArch(BaseArch):
  def postprocess(self, 
                  batch_data: torch.Tensor, 
                  batch_info: DataSample):
    return batch_data
