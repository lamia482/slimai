import torch
from typing import Dict
from slimai.helper.help_build import MODELS, build_loss
from slimai.helper.structure import DataSample


@MODELS.register_module()
class BasicClassificationLoss(torch.nn.Module):
  def __init__(self, 
               cls_loss=dict(
                  type="torch.nn.CrossEntropyLoss",
                  label_smoothing=0.1,
               )):
    super().__init__()
    self.cls_loss = build_loss(cls_loss)
    return
  
  def forward(self, 
              embedding_dict: Dict[str, torch.Tensor], 
              batch_info: DataSample) -> Dict[str, torch.Tensor]:
    logits = embedding_dict["head"]
    labels = batch_info.label
    cls_loss = self.cls_loss(logits, labels)
    return dict(cls_loss=cls_loss)
