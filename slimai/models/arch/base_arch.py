from abc import abstractmethod
import torch
from typing import List, Optional, Union, Dict
from slimai.helper import help_build, help_utils
from slimai.helper.structure import DataSample


class BaseArch(torch.nn.Module):
  def __init__(self, *, 
               encoder=dict(
                 backbone=None, neck=None, 
               ), 
               decoder=dict(
                 head=None, 
               ), 
               loss=None):
    super().__init__()
    
    # init model layers
    self.encoder = torch.nn.ModuleDict({
      component: help_build.build_model(cfg)
      for component, cfg in encoder.items()
    })
    self.decoder = torch.nn.ModuleDict({
      component: help_build.build_model(cfg)
      for component, cfg in decoder.items()
    })
    self.loss = help_build.build_loss(loss)

    help_utils.print_log(f"Model({__class__.__name__}) built successfully with {help_utils.PytorchNetworkUtils.get_params_size(self)} parameters")
    return
  
  @property
  def device(self):
    return next(self.parameters()).device
  
  def forward(self, 
              batch_data: torch.Tensor, 
              batch_info: Optional[Union[Dict, DataSample]] = None,
              mode="tensor"):
    expected_modes = ["tensor", "loss", "predict"]
    if mode not in expected_modes:
      raise RuntimeError(f"Invalid mode \"{mode}\". Only supports {expected_modes}")
    func = getattr(self, f"do_{mode}")

    if batch_info is None:
      batch_info = DataSample()
    if not isinstance(batch_info, DataSample):
      batch_info = DataSample(**batch_info)

    return func(batch_data, batch_info)

  def do_tensor(self, 
                batch_data: torch.Tensor, 
                batch_info: DataSample):
    data = batch_data
    data = self.encoder.backbone(data)
    data = self.encoder.neck(data)
    data = self.decoder.head(data)
    return data

  @abstractmethod
  def do_loss(self, 
              batch_data: torch.Tensor, 
              batch_info: DataSample):
    raise NotImplementedError

  @abstractmethod
  def postprocess(self, 
                  batch_data: torch.Tensor, 
                  batch_info: DataSample):
    raise NotImplementedError

  def predict(self, 
              batch_data: torch.Tensor, 
              batch_info: List[DataSample]):
    data = self.do_tensor(batch_data, batch_info)
    return self.postprocess(data, batch_info)