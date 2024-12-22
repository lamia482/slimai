from abc import abstractmethod
import torch
from typing import List
from slimai.helper import help_build, help_utils
from slimai.helper.structure import DataSample


class BaseArch(torch.nn.Module):
  def __init__(self, *, 
               encoder, 
               decoder, 
               loss, 
               solver=None):
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
    self.solver = help_build.build_solver(solver, self)

    self._init_layers()
    help_utils.print_log(f"Model({__class__.__name__}) built successfully with {help_utils.PytorchNetworkUtils.get_params_size(self)} parameters")
    return

  @abstractmethod
  def _init_layers(self):
    pass
  
  def forward(self, 
              batch_inputs: torch.Tensor, 
              *,
              batch_datasamples: List[DataSample]=None, 
              mode="tensor"):
    expected_modes = ["tensor", "loss", "predict"]
    if mode not in expected_modes:
      raise RuntimeError(f"Invalid mode \"{mode}\". Only supports {expected_modes}")
    func = getattr(self, f"do_{mode}")
    return func(batch_inputs, batch_datasamples)

  @abstractmethod
  def do_tensor(self, 
             batch_inputs: torch.Tensor, 
             *,
             batch_datasamples: List[DataSample]=None):
    raise NotImplementedError

  @abstractmethod
  def do_loss(self, 
              batch_inputs: torch.Tensor, 
              *,
              batch_datasamples: List[DataSample]=None):
    raise NotImplementedError

  @abstractmethod
  def predict(self, 
              batch_inputs: torch.Tensor, 
              *,
              batch_datasamples: List[DataSample]=None):
    raise NotImplementedError
