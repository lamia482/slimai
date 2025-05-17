from typing import Dict
import torch
from slimai.helper.help_build import MODELS, build_metric


@MODELS.register_module()
class BasicDetectionMetric(torch.nn.Module):
  def __init__(self, 
               ap=dict(
                type="torchmetrics.detection.MeanAveragePrecision", 
                box_format="xyxy",
                iou_type="bbox",
                iou_thresholds=None,
                class_metrics=True,
               )
              ):
    super().__init__()
    self.ap = build_metric(ap)
    return

  @torch.no_grad()
  def forward(self, 
              output: Dict[str, torch.Tensor], 
              targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    pred_scores, pred_labels, pred_bboxes = output["scores"], output["labels"], output["bboxes"]
    instances = targets["instance"]
    target_labels, target_bboxes = instances["labels"], instances["bboxes"]

    preds=[
      dict(boxes=boxes, scores=scores, labels=labels) 
      for boxes, scores, labels in zip(pred_bboxes, pred_scores, pred_labels)
    ]
    target=[
      dict(boxes=boxes, labels=labels) 
      for boxes, labels in zip(target_bboxes, target_labels)
    ]
    
    self.ap.update(preds, target)
    ap = self.ap.compute()
    return ap
