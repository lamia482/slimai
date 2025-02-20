from abc import abstractmethod
import math
import sys
import torch
import torch.utils.checkpoint as checkpoint
from functools import partial
from contextlib import ExitStack, contextmanager
from typing import Optional, Union, Dict
from slimai.helper import help_build, help_utils
from slimai.helper.utils.dist_env import dist_env
from slimai.helper.structure import DataSample
from ..component.pipeline import Pipeline


class BaseArch(object):
  def __init__(self, *, 
               encoder=dict(
                 backbone=None, neck=None, 
               ), 
               decoder=dict(
                 head=None, 
               ), 
               loss=None, 
               solver=None):
    """initialize model and solver
    1. create model and initialize weight randomly
    2. wrap with ddp
    3. create solver
    4. create loss and move to model device
    """
    super().__init__()
    
    # Initialize model layers
    self.model = self.init_layers(encoder, decoder)
    self.model.apply(help_utils.PytorchNetworkUtils.init_weights)
    self.model = dist_env.init_dist(module=self.model)
    help_utils.print_log(self.model)

    # Initialize solver
    self.solver = self.init_solver(solver, self.model)

    # Initialize loss
    self.loss = self.init_loss(loss)
    self.loss = dist_env.init_dist(module=self.loss)

    # Log model parameter size
    help_utils.print_log(f"Model({self.__class__.__name__}) built successfully "
                         f"with {help_utils.PytorchNetworkUtils.get_params_size(self.export_model)} parameters")
    return

  @abstractmethod
  def init_layers(self, encoder, decoder) -> torch.nn.Module:
    help_utils.print_log(
      f"Using default `init_layers` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    model = Pipeline(encoder.backbone, encoder.neck, decoder.head)
    return model
  
  @property
  @abstractmethod
  def export_model(self):
    help_utils.print_log(
      f"Using default `export_model` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    return self.model

  @abstractmethod
  def init_solver(self, solver, module):
    help_utils.print_log(
      f"Using default `init_solver` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    params = help_utils.PytorchNetworkUtils.get_module_params(module, grad_mode="trainable")
    return help_build.build_solver(solver, params=params)

  @abstractmethod
  def init_loss(self, loss) -> torch.nn.Module:
    help_utils.print_log(
      f"Using default `init_loss` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    return help_build.build_loss(loss)
  
  @property
  def device(self):
    return next(self.model.parameters()).device
  
  @abstractmethod
  def load_state_dict(self, state_dict, strict=True):
    help_utils.print_log(
      f"Using default `load_state_dict` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    self.model.load_state_dict(state_dict, strict=strict)
    return
  
  @contextmanager
  def no_sync(self):
    with ExitStack() as stack:
      if not isinstance(self.model, torch.nn.ModuleDict):
        stack.enter_context(self.model.no_sync())
      else:
        for module in self.model.values():
          stack.enter_context(module.no_sync())
      yield
  
  @abstractmethod
  def epoch_precede_hooks(self, *, runner):
    help_utils.print_log(
      f"Using default `epoch_precede_hooks` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    # set train and clear grad before next epoch in case of former evaluation
    self.model.train()
    return
  
  @abstractmethod
  def epoch_succeed_hooks(self, *, runner):
    help_utils.print_log(
      f"Using default `epoch_succeed_hooks` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    return
  
  @abstractmethod
  def step_precede_hooks(self, *, runner):
    # Default step_precede_hooks
    help_utils.print_log(
      f"Using default `step_precede_hooks` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    self.current_train_epoch = runner.epoch - 1 # epoch in runner start from 1
    self.max_train_epoch = runner.max_epoch
    self.current_train_step = runner.step # step in runner start from 0
    self.max_train_step = len(runner.train_dataloader)
    return
  
  @abstractmethod
  def step_succeed_hooks(self, *, runner):
    # Default step_succeed_hooks
    help_utils.print_log(
      f"Using default `step_succeed_hooks` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    return
  
  # TODO: add torch.compile
  def __call__(self, 
               batch_data: Union[torch.Tensor, Dict[str, torch.Tensor]], 
               batch_info: Optional[Union[Dict, DataSample]] = None,
               mode="tensor", gradient_checkpointing=False
               ) -> Union[Dict, torch.Tensor, DataSample]:
    # Forward pass with different modes: tensor, loss, predict
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
      if gradient_checkpointing:
        # use checkpoint to save memory during training
        forward_func = partial(self._forward_tensor, return_flow=True)
        embedding_dict = checkpoint.checkpoint(forward_func, batch_data, use_reentrant=False)
      else:
        embedding_dict = self._forward_tensor(batch_data, return_flow=True)
        
      loss_dict = self._forward_loss(embedding_dict, batch_info)
      assert (
        isinstance(loss_dict, dict) and len(loss_dict) > 0
      ), "`loss_dict` after `_forward_loss` must be a non-empty dictionary, but got {}".format(loss_dict)

      loss = sum(loss_dict.values())
      if not math.isfinite(loss.item()):
        help_utils.print_log("Loss is {}, stopping training".format(loss), level="ERROR")
        sys.exit(1)
      output = loss_dict

    elif mode == "predict":
      output = self.predict(batch_data, batch_info)

    return output

  @abstractmethod
  def _forward_tensor(self, 
                batch_data: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                return_flow: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    # Default forward pass through encoder and decoder
    help_utils.print_log(
      f"Using default `_forward_tensor` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    return self.model(batch_data, return_flow=return_flow)

  @abstractmethod
  def _forward_loss(self, 
              embedding_dict: Dict[str, torch.Tensor], 
              batch_info: DataSample) -> Dict[str, torch.Tensor]:
    # Default loss computation
    help_utils.print_log(
      f"Using default `_forward_loss` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    
    logits = embedding_dict["head"]
    targets = batch_info.label
    loss = self.loss(logits, targets)
    return loss

  @abstractmethod
  def postprocess(self, 
                  batch_data: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                  batch_info: DataSample) -> DataSample:
    # Postprocess method to be implemented in subclasses
    raise NotImplementedError("`postprocess` is not implemented and is necessary for `predict`")

  @torch.no_grad()
  def predict(self, 
              batch_data: Union[torch.Tensor, Dict[str, torch.Tensor]], 
              batch_info: DataSample) -> DataSample:
    # Predict method using postprocess
    output = self._forward_tensor(batch_data)
    return self.postprocess(output, batch_info)
  