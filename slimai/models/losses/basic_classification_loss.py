import torch
from slimai.helper.help_build import MODELS, build_loss


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
  
  def forward(self, embedding_dict, batch_info):
    cls_loss = self.cls_loss(embedding_dict["head"], batch_info.label)
    return dict(cls_loss=cls_loss)
