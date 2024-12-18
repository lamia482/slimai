from abc import abstractmethod
import torch
from mmengine.model import BaseModel as MMengineBaseModel
from mmengine.registry import MODELS
from slimai.helper import help_utils


class BaseComponent(torch.nn.Module):
  def __init__(self, pretrained_weight=None):
    super().__init__()
    self._init_weights(pretrained_weight)
    return

  @abstractmethod
  def _init_weights(self, pretrained_weight):
    if pretrained_weight is not None:
      self.load_state_dict(torch.load(pretrained_weight, map_location="cpu"), strict=False)
    return

  @abstractmethod
  def forward(self, *args, **kwargs):
    raise NotImplementedError


class BaseArch(MMengineBaseModel):
  def __init__(self,  
               backbone, 
               neck=None, 
               encoder=None, 
               decoder=None, 
               head=None, 
               data_preprocessor=None):
    super().__init__(data_preprocessor=data_preprocessor, init_cfg=None)
    
    # init model layers
    self.backbone = MODELS.build(backbone)
    if neck is not None:
      self.neck = MODELS.build(neck)
    if encoder is not None:
      self.encoder = MODELS.build(encoder)
    if decoder is not None:
      self.decoder = MODELS.build(decoder)
    if head is not None:
      self.head = MODELS.build(head)

    self._init_layers()
    help_utils.print_log(f"Model({__class__.__name__}) built successfully with {self.num_parameters()} parameters")
    return

  @abstractmethod
  def _init_layers(self):
    pass
  
  def forward(self, batch_inputs, batch_data_samples, mode="tensor"):
    expected_modes = ["tensor", "loss", "predict"]
    if mode not in expected_modes:
      raise RuntimeError(f"Invalid mode \"{mode}\". Only supports {expected_modes}")
    func = getattr(self, mode)
    return func(batch_inputs, batch_data_samples)

  @abstractmethod
  def tensor(self, batch_inputs, batch_data_samples):
    raise NotImplementedError

  @abstractmethod
  def loss(self, batch_inputs, batch_data_samples):
    raise NotImplementedError

  @abstractmethod
  def predict(self, batch_inputs, batch_data_samples):
    raise NotImplementedError
