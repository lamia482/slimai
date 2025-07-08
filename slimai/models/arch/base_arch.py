from abc import abstractmethod
import math
import sys
import torch
from functools import partial
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from typing import Optional, Union, Dict, Tuple
from slimai.helper import help_build, DataSample, Distributed
from slimai.helper.help_utils import print_log
from slimai.helper.utils import PytorchNetworkUtils
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

    self.dist = Distributed.create()

    # Initialize model layers
    model = self.init_layers(encoder, decoder)
    self.model = model.apply(PytorchNetworkUtils.init_weights)
    print_log(model)

    # Initialize solver and scheduler
    self.solver, self.scheduler = self.init_solver(solver, self.model)

    # Initialize loss
    self.loss = self.init_loss(loss)
    return

  def compile(self, compile: bool = False):
    if compile:
      if self.dist.env.accelerator not in ["cuda"]:
        print_log(f"`compile` is not stable on non-CUDA accelerator, but currently is compiling on Accelerator:{self.dist.env.accelerator}", level="WARNING", warn_once=True)
      self.model.compile()
    return

  def checkpointing(self, checkpointing: bool = False, use_reentrant: bool = False):
    if checkpointing:
      self.gradient_checkpoint = partial(gradient_checkpoint, use_reentrant=use_reentrant)
    else:
      self.gradient_checkpoint = lambda func, *args, **kwargs: func(*args, **kwargs)
    return

  def extract(self):
    return self.model, self.solver, self.scheduler, self.loss

  @abstractmethod
  def init_layers(self, encoder, decoder) -> torch.nn.Module:
    print_log(
      f"Using default `init_layers` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    model = Pipeline(encoder.backbone, encoder.neck, decoder.head)
    return model

  @abstractmethod
  def init_solver(self, solver, module):
    print_log(
      f"Using default `init_solver` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    params = PytorchNetworkUtils.get_module_params(module, grad_mode="all")
    return help_build.build_solver(solver, params=params)

  @abstractmethod
  def init_loss(self, loss) -> torch.nn.Module:
    print_log(
      f"Using default `init_loss` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    return help_build.build_loss(loss)
  
  @abstractmethod
  def load_state_dict(self, state_dict, strict=True):
    print_log(
      f"Using default `load_state_dict` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    self.model.load_state_dict(state_dict, strict=strict)
    return
  
  @abstractmethod
  def epoch_precede_hooks(self, *, runner):
    print_log(
      f"Using default `epoch_precede_hooks` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    # set train and clear grad before next epoch in case of former evaluation
    self.model.train()
    return
  
  @abstractmethod
  def epoch_succeed_hooks(self, *, runner):
    print_log(
      f"Using default `epoch_succeed_hooks` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    return
  
  @abstractmethod
  def step_precede_hooks(self, *, runner):
    # Default step_precede_hooks
    print_log(
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
    print_log(
      f"Using default `step_succeed_hooks` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    return
  
  def __call__(self, 
               batch_data: Union[torch.Tensor, Dict[str, torch.Tensor]], 
               batch_info: Optional[Union[Dict, DataSample]] = None,
               mode="tensor"
               ) -> Union[Tuple, Dict, torch.Tensor, DataSample]:
    # Forward pass with different modes: tensor, loss, predict
    expected_modes = ["tensor", "loss", "predict"]
    if mode not in expected_modes:
      raise RuntimeError(f"Invalid mode \"{mode}\". Only supports {expected_modes}")

    if batch_info is None:
      batch_info = DataSample()
    if not isinstance(batch_info, DataSample):
      batch_info = DataSample(**batch_info)

    if mode == "tensor":
      output = self._forward_tensor(batch_data, return_flow=False)

    elif mode == "loss":
      embedding_dict = self.gradient_checkpoint(
        self._forward_tensor, batch_data, return_flow=True
      )
      loss_dict = self._forward_loss(embedding_dict, batch_info) # type: ignore
      assert (
        isinstance(loss_dict, dict) and len(loss_dict) > 0
      ), "`loss_dict` after `_forward_loss` must be a non-empty dictionary, but got {}".format(loss_dict)

      loss = sum(loss_dict.values())
      if not math.isfinite(loss.item()): # type: ignore
        print_log("Loss is {}, stopping training".format(loss), level="ERROR")
        sys.exit(1)
      output = (embedding_dict, loss_dict)

    elif mode == "predict":
      output = self.predict(batch_data, batch_info)

    return output

  @abstractmethod
  def _forward_tensor(self, 
                batch_data: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                return_flow: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    # Default forward pass through encoder and decoder
    print_log(
      f"Using default `_forward_tensor` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    return self.model(batch_data, return_flow=return_flow)

  @abstractmethod
  def _forward_loss(self, 
              embedding_dict: Dict[str, torch.Tensor], 
              batch_info: DataSample) -> Dict[str, torch.Tensor]:
    # Default loss computation
    print_log(
      f"Using default `_forward_loss` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    
    logits = embedding_dict["head"]
    targets = batch_info.label # type: ignore
    loss = self.loss(logits, targets)
    return loss
  
  @abstractmethod
  def export_model(self) -> torch.nn.Module:
    # Export model for inference and export to onnx
    raise NotImplementedError("`export_model` is not implemented and is necessary for `predict`")

  @torch.inference_mode()
  def postprocess(self, *args, **kwargs):
    return self._postprocess(*args, **kwargs)

  @abstractmethod
  def _postprocess(self, 
                   batch_data: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                   batch_info: DataSample) -> DataSample:
    # Postprocess method to be implemented in subclasses
    raise NotImplementedError("`_postprocess` is not implemented and is necessary for `predict`")

  @torch.inference_mode()
  def predict(self, 
              batch_data: Union[torch.Tensor, Dict[str, torch.Tensor]], 
              batch_info: DataSample) -> DataSample:
    # Predict method using postprocess
    infer_model = None
    try:
      print_log("Try exporting model for inference", level="INFO", warn_once=True)
      infer_model = self.export_model()
    except Exception as e:
      print_log("Failed to export model for inference, using default forward pass", level="WARNING", warn_once=True)
      infer_model = partial(self._forward_tensor, return_flow=False)
    output = infer_model(batch_data)
    return self.postprocess(output, batch_info)
  