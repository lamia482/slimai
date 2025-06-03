import torch
from typing import Union, Dict, List
from slimai.helper.help_utils import print_log
from slimai.helper.help_build import MODELS, build_model
from slimai.helper.utils import PytorchNetworkUtils
from .cls_arch import ClassificationArch


__all__ = [
  "MIL",
]

@MODELS.register_module()
class MIL(ClassificationArch):
  """Multiple Instance Learning (MIL) architecture.
  
  This class implements a MIL architecture for processing bags of instances.
  It consists of a backbone for feature extraction, a neck for aggregating
  instance features into bag-level representations, and a head for classification.
  
  Args:
    encoder (dict): Configuration for the encoder, including backbone and neck components.
    decoder (dict): Configuration for the decoder, including the head component.
    loss (dict, optional): Configuration for the loss function.
    solver (dict, optional): Configuration for the optimizer.
    embedding_group_size (int): Number of instances to process together in the backbone.
      This helps manage memory usage for large bags.
    freeze_backbone (bool): Whether to freeze the backbone parameters during training.
  """
  def __init__(self, *, 
               encoder=dict(
                 backbone=None, neck=None, 
               ), 
               decoder=dict(
                 head=None, 
               ), 
               loss=None, 
               solver=None, 
               embedding_group_size=1, 
               freeze_backbone=False,
               ):
    super().__init__(encoder=encoder, decoder=decoder, loss=loss, solver=solver)
    if freeze_backbone:
      print_log("Freezing backbone.")
      PytorchNetworkUtils.freeze(self.model.backbone)
    self.embedding_group_size = embedding_group_size
    return

  def init_layers(self, encoder, decoder) -> torch.nn.Module:
    """Initialize the model layers.
    
    Args:
      encoder (dict): Configuration for the encoder components.
      decoder (dict): Configuration for the decoder components.
      
    Returns:
      torch.nn.ModuleDict: Dictionary containing the backbone, neck, and head modules.
    """
    print_log(
      f"Using default `init_layers` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    backbone = build_model(encoder.backbone)
    neck = build_model(encoder.neck) 
    head = build_model(decoder.head)
    
    return torch.nn.ModuleDict(dict(backbone=backbone, neck=neck, head=head))
  
  def _forward_tensor(self, 
                batch_data: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                return_flow: bool = False) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:
    """Forward pass for tensor input.
    
    Processes a batch of bags through the model. Each bag contains multiple instances.
    
    Args:
      batch_data: Input data containing bags of instances, shape (B, ~N, C, H, W)
                 where B is batch size, ~N is variable number of instances per bag.
      return_flow: If True, returns intermediate outputs from each component.
      
    Returns:
      If return_flow is True, returns a dictionary with outputs from backbone, neck, and head.
      Otherwise, returns only the head output (classification logits).
    """
    def forward_backbone(images, group_size=self.embedding_group_size):
      """Process instances through backbone in groups to manage memory.
      
      Args:
        images: Tensor of instances from a single bag.
        group_size: Number of instances to process together.
        
      Returns:
        Tensor of instance embeddings.
      """
      if group_size <= 0:
        group_size = len(images)
      output = []
      for i in range(0, len(images), group_size):
        embedding = self.model.backbone(images[i:i+group_size]) # type: ignore
        output.append(embedding)
      return torch.cat(output, dim=0)

    # batch_data in shape (B, ~N, C, H, W)
    backbone = list(map(forward_backbone, batch_data)) # (B, ~N, D)
    neck = self.model.neck(backbone) # type: ignore # (B, D)
    head = self.model.head(neck) # type: ignore # (B, C)

    if return_flow:
      return dict(
        backbone=backbone,
        neck=neck,
        head=head,
      )
    else:
      return head
  