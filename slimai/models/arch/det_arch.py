import torch
from typing import Dict
from slimai.helper.help_build import MODELS
from slimai.helper.structure import DataSample
from slimai.helper.utils import box_ops
from .base_arch import BaseArch


@MODELS.register_module()
class DetectionArch(BaseArch):
  def _forward_loss(self, 
              embedding_dict: Dict[str, torch.Tensor], 
              batch_info: DataSample) -> Dict[str, torch.Tensor]:
    cls_logits, bbox_logits = embedding_dict["head"]
    targets = batch_info.instance # type: ignore
    cls_targets, bbox_targets = targets["labels"], targets["bboxes"]

    # concat Nx2 to Nx4, replicate (width, height) to (width, height, width, height)
    whwh = torch.cat([torch.stack([batch_info.width, batch_info.height], dim=1)] * 2, dim=-1) # type: ignore
    bbox_targets = [
      bboxes / whwh[i]
      for i, bboxes in enumerate(bbox_targets)
    ]
    bbox_targets = list(map(box_ops.box_xyxy_to_cxcywh, bbox_targets))

    loss = self.loss(cls_logits, bbox_logits, cls_targets, bbox_targets)
    return loss

  def postprocess(self, 
                  batch_data: torch.Tensor, 
                  batch_info: DataSample) -> DataSample:
    cls_logits, bbox_logits = batch_data

    cls_dist, pred_scores, pred_labels, fg_mask \
      = self.loss.parse_cls_logits(cls_logits) # type: ignore

    fg_mask = torch.as_tensor(fg_mask, device=cls_logits.device) # [B, Q]
    pred_logits = [l[m] for l, m in zip(cls_logits, fg_mask)]
    pred_scores = [s[m] for s, m in zip(pred_scores, fg_mask)]
    pred_labels = [l[m] for l, m in zip(pred_labels, fg_mask)]
    pred_bboxes = [ 
      box_ops.box_cxcywh_to_xyxy(b[m].sigmoid()).clamp(min=0, max=1) * torch.stack([w, h, w, h])
      for (b, m, w, h) in zip(bbox_logits, fg_mask, batch_info.width, batch_info.height) # type: ignore
    ]
    
    batch_info.output = dict(
      cls_logits=pred_logits, # [B, ~Q, C] or [B, ~Q, C+1]
      scores=pred_scores, # [B, ~Q]
      labels=pred_labels, # [B, ~Q]
      bboxes=pred_bboxes, # [B, ~Q, 4], where 4 indicates [xmin, ymin, xmax, ymax]
    )
    return batch_info
