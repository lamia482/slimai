import torch
from slimai.helper.help_build import MODELS
from slimai.helper.structure import DataSample
from .base_arch import BaseArch


@MODELS.register_module()
class ClassificationArch(BaseArch):
  def postprocess(self, 
                  batch_data: torch.Tensor, 
                  batch_info: DataSample) -> DataSample:
    cls_logits = batch_data
    softmax = torch.softmax(cls_logits, dim=1) # [B, C]
    pred_scores = softmax.max(dim=1).values # [B]
    pred_labels = softmax.argmax(dim=1) # [B]

    batch_info.output = dict(
      logits=cls_logits, # [B, C]
      softmax=softmax, # [B, C]
      scores=pred_scores, # [B]
      labels=pred_labels, # [B]
    )
    return batch_info
