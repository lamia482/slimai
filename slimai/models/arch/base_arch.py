from abc import abstractmethod
import math
import sys
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
              mode="tensor") -> Union[Dict, torch.Tensor, DataSample]:
    expected_modes = ["tensor", "loss", "predict"]
    if mode not in expected_modes:
      raise RuntimeError(f"Invalid mode \"{mode}\". Only supports {expected_modes}")

    if batch_info is None:
      batch_info = DataSample()
    if not isinstance(batch_info, DataSample):
      batch_info = DataSample(**batch_info)

    if mode == "tensor":
      output = self._forward_tensor(batch_data)

    elif mode == "loss":
      embedding_dict = self._forward_tensor(batch_data, return_flow=True)
      loss_dict = self._forward_loss(embedding_dict, batch_info)

      loss = sum(loss_dict.values())
      if not math.isfinite(loss.item()):
          help_utils.print_log("Loss is {}, stopping training".format(loss), level="ERROR")
          sys.exit(1)
      output = loss_dict

    elif mode == "predict":
      output = self.predict(batch_data, batch_info)

    return output

  def _forward_tensor(self, 
                batch_data: torch.Tensor, 
                return_flow: bool = False) -> torch.Tensor:
    backbone = self.encoder.backbone(batch_data)
    neck = self.encoder.neck(backbone)
    head = self.decoder.head(neck)
    if return_flow:
      output = dict(
        backbone=backbone, 
        neck=neck, 
        head=head, 
      )
    else:
      output = head
    return output

  def _forward_loss(self, 
              embedding_dict: Dict[str, torch.Tensor], 
              batch_info: DataSample) -> Dict[str, torch.Tensor]:
    logits = embedding_dict["head"]
    targets = batch_info.label
    loss = self.loss(logits, targets)
    return loss

  @abstractmethod
  def postprocess(self, 
                  batch_data: torch.Tensor, 
                  batch_info: DataSample) -> DataSample:
    raise NotImplementedError

  def predict(self, 
              batch_data: torch.Tensor, 
              batch_info: DataSample) -> DataSample:
    output = self._forward_tensor(batch_data)
    return self.postprocess(output, batch_info)