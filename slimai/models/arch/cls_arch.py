from typing import List
import torch
from slimai.helper.help_build import MODELS
from slimai.helper.structure import DataSample
from .base_arch import BaseArch


@MODELS.register_module()
class ClassificationArch(BaseArch):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    return
  
  def _init_layers(self):
    pass
  
  def tensor(self, 
             batch_inputs: torch.Tensor, 
             *,
             batch_datasamples: List[DataSample]=None):
    return

  def loss(self, 
           batch_inputs: torch.Tensor, 
           *,
           batch_datasamples: List[DataSample]=None):
    return

  def predict(self, 
              batch_inputs: torch.Tensor, 
              *,
              batch_datasamples: List[DataSample]=None):
    return
