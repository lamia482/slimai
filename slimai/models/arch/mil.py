import torch
from typing import Union, Dict, List, Tuple
from slimai.helper.help_utils import print_log
from slimai.helper.help_build import MODELS, build_model
from slimai.helper.utils import PytorchNetworkUtils
from slimai.helper import DataSample
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
                return_flow: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:
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
    backbone = None
    if meta := getattr(self, "meta", None):
      backbone = meta.get("embeddings", None)
    if backbone is None:
      backbone = list(map(forward_backbone, batch_data)) # (B, ~N, D)

    neck = self.model.neck(backbone) # type: ignore # (B, D)
    if isinstance(neck, tuple):
      neck, atten_logits = neck
    else:
      atten_logits = None
    head = self.model.head(neck) # type: ignore # (B, C)

    if return_flow:
      return dict(
        backbone=backbone,
        atten_logits=atten_logits,
        neck=neck,
        head=head,
      )
    else:
      return atten_logits, head # type: ignore

  def _forward_loss(self, 
              embedding_dict: Dict[str, torch.Tensor], 
              batch_info: DataSample) -> Dict[str, torch.Tensor]:
    backbone = embedding_dict["backbone"]
    atten_logits = embedding_dict["atten_logits"]
    cls_logits = embedding_dict["head"]
    cls_targets = batch_info.label # type: ignore
    loss = self.loss(backbone, atten_logits, cls_logits, cls_targets)
    return loss

  def _postprocess(self, 
                  batch_data: Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]], 
                  batch_info: DataSample) -> DataSample: 
    if isinstance(batch_data, dict):
      atten_logits = batch_data["atten_logits"]
      cls_logits = batch_data["head"]
    else:
      atten_logits, cls_logits = batch_data
    
    super()._postprocess(cls_logits, batch_info)
    
    topk = 8*8 # topk indices to visualize
    atten_softmax = [
      al.squeeze().softmax(-1) # [B, ~N]
      for al in atten_logits
    ]
    atten_topk = [
      al.topk(k=topk, dim=-1, largest=True) # select topk(maximum) indices
      for al in atten_softmax
    ]
    atten_tailk = [
      al.topk(k=topk, dim=-1, largest=False) # select tailk(minimum) indices
      for al in atten_softmax
    ]

    topk_scores, topk_indices = list(map(list, zip(*atten_topk)))
    tailk_scores, tailk_indices = list(map(list, zip(*atten_tailk)))

    batch_info.output.update(dict(
      topk_atten_logits=[a[i] for a, i in zip(atten_logits, topk_indices)], # [B, ~N]
      topk_atten_indices=topk_indices, # [B, ~N]
      topk_atten_scores=topk_scores, # [B, ~N]
      tailk_atten_logits=[a[i] for a, i in zip(atten_logits, tailk_indices)], # [B, ~N]
      tailk_atten_indices=tailk_indices, # [B, ~N]
      tailk_atten_scores=tailk_scores, # [B, ~N]
    ))

    return batch_info
  