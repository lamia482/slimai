import torch
import mmengine
from typing import Union, Dict, List, Tuple
from slimai.helper.help_utils import print_log
from slimai.helper.help_build import MODELS, build_model
from slimai.helper.utils import PytorchNetworkUtils, get_cacher
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
               backbone=None, 
               neck=None, 
               head=None, 
               loss=None, 
               solver=None, 
               embedding_group_size=1, 
               freeze_backbone=False,
               ):
    super().__init__(backbone=backbone, neck=neck, head=head, loss=loss, solver=solver)
    self.freeze_backbone = freeze_backbone
    if freeze_backbone:
      print_log("Freezing backbone.")
      PytorchNetworkUtils.freeze(self.model.backbone)
    self.embedding_group_size = embedding_group_size
    self.cacher = get_cacher()
    return

  def init_layers(self, backbone, neck, head) -> torch.nn.Module:
    """Initialize the model layers.
    
    Args:
      backbone (dict): Configuration for the backbone components.
      neck (dict): Configuration for the neck components.
      head (dict): Configuration for the head components.
      
    Returns:
      torch.nn.ModuleDict: Dictionary containing the backbone, neck, and head modules.
    """
    print_log(
      f"Using default `init_layers` in {self.__class__.__name__}",
      level="WARNING", warn_once=True
    )
    backbone = build_model(backbone)
    neck = build_model(neck) 
    head = build_model(head)
    
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
      if self.freeze_backbone:
        for i in range(0, len(images), group_size):
            with torch.inference_mode():
              embedding = self.model.backbone(images[i:i+group_size]) # type: ignore
            output.append(embedding)
      else:
        for i in range(0, len(images), group_size):
          embedding = self.model.backbone(images[i:i+group_size]) # type: ignore
          output.append(embedding)
      return torch.cat(output, dim=0)

    meta = getattr(self, "meta")

    embedding_list = []
    for vis_image, use_cache, embedding, embedding_key, visual_key in zip(
      batch_data, meta["use_cache"], 
      meta.pop("embedding"), # pop embedding from meta to avoid swanlab error
      meta["embedding_key"], meta["visual_key"]
    ):
      if (not use_cache) or (embedding is None):
        embedding = forward_backbone(vis_image)
      embedding_list.append(embedding)
      if not use_cache:
        continue
      if not self.cacher.has(embedding_key):
        self.cacher.put(embedding_key, embedding)
      if not self.cacher.has(visual_key):
        self.cacher.put(visual_key, vis_image)

    backbone = embedding_list
    neck, atten_weights = self.model.neck(backbone) # type: ignore # (B, D)
    head = self.model.head(neck) # type: ignore # (B, C)

    if return_flow:
      return dict(
        backbone=backbone,
        atten_weights=atten_weights,
        neck=neck,
        head=head,
      )
    else:
      return atten_weights, head # type: ignore

  def _forward_loss(self, 
              embedding_dict: Dict[str, torch.Tensor], 
              batch_info: DataSample) -> Dict[str, torch.Tensor]:
    backbone = embedding_dict["backbone"]
    atten_weights = embedding_dict["atten_weights"]
    cls_logits = embedding_dict["head"]
    cls_targets = batch_info.label # type: ignore
    loss = self.loss(backbone, atten_weights, cls_logits, cls_targets)
    return loss

  def _postprocess(self, 
                  batch_data: Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]], 
                  batch_info: DataSample) -> DataSample: 
    if isinstance(batch_data, dict):
      atten_weights = batch_data["atten_weights"]
      cls_logits = batch_data["head"]
    else:
      atten_weights, cls_logits = batch_data
    
    super()._postprocess(cls_logits, batch_info)

    # Only compute topk/tailk when attention weights are tensors (e.g. ABMIL); WMIL returns list of lists
    if atten_weights and all(torch.is_tensor(al) for al in atten_weights):
      topk = 8*8
      atten_topk = [al.topk(k=min(topk, al.shape[-1]), dim=-1, largest=True) for al in atten_weights]
      atten_tailk = [al.topk(k=min(topk, al.shape[-1]), dim=-1, largest=False) for al in atten_weights]
      topk_scores, topk_indices = list(map(list, zip(*atten_topk)))
      tailk_scores, tailk_indices = list(map(list, zip(*atten_tailk)))
      batch_info.output.update(dict(
        topk_atten_indices=topk_indices,
        topk_atten_scores=topk_scores,
        tailk_atten_indices=tailk_indices,
        tailk_atten_scores=tailk_scores,
      ))

    return batch_info
  