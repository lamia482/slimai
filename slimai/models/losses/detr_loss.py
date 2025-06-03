import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment
from slimai.helper.help_build import MODELS
from slimai.helper.utils import box_ops
from slimai.helper import Distributed


__all__ = [
  "DETRLoss", 
  "HungarianMatcher", 
]

@MODELS.register_module()
class DETRLoss(torch.nn.Module):
  def __init__(self, 
               *, 
               matcher=dict(
                 cost_class=1,
                 cost_bbox=1,
                 cost_giou=1,
               ), 
               num_classes=1, 
               eos_coef=0.1, 
               cls_weight=1, 
               box_weight=5, 
               giou_weight=2, 
               ):
    super().__init__()
    self.matcher = HungarianMatcher(**matcher)
    self.num_classes = num_classes
    self.eos_coef = eos_coef

    self.cls_weight = cls_weight
    self.box_weight = box_weight
    self.giou_weight = giou_weight

    empty_weight = torch.ones(self.num_classes + 1)
    empty_weight[-1] = self.eos_coef
    self.register_buffer("empty_weight", empty_weight)

    self.dist = Distributed.create()
    return
  
  def forward(self, 
              cls_logits: torch.Tensor, 
              bbox_logits: torch.Tensor, 
              cls_targets: List[torch.Tensor], 
              bbox_targets: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    bbox_preds = bbox_logits.sigmoid()
    indices = self.matcher(cls_logits, bbox_preds, cls_targets, bbox_targets)
    num_bboxes = self.dist.prepare_for_distributed(torch.as_tensor(sum(map(len, bbox_targets)), dtype=torch.float))
    num_bboxes: int = self.dist.env.sync(num_bboxes).clamp_min(min=1).item() # type: ignore

    loss = dict(
      **self.compute_cls_loss(cls_logits, cls_targets, indices, num_bboxes),
      **self.compute_bbox_loss(bbox_preds, bbox_targets, indices, num_bboxes),
      **self.compute_cardinality_loss(cls_logits, cls_targets, indices, num_bboxes),
    )
    
    return loss
  
  def compute_cls_loss(self, cls_logits: torch.Tensor, cls_targets: List[torch.Tensor], 
                       indices: List[Tuple[torch.Tensor, torch.Tensor]], num_bboxes: int):
    idx = self._get_pred_permutation_idx(indices)
    target_classes = torch.full(cls_logits.shape[:2], self.num_classes, dtype=torch.int64, device=cls_logits.device)
    target_classes[idx] = torch.cat([
      t[j] for t, (_, j) in zip(cls_targets, indices)
    ], dim=0)

    cls_loss = F.cross_entropy(cls_logits.flatten(0, 1), 
                               target_classes.flatten(0, 1), self.empty_weight) # type: ignore
    return dict(cls_loss=cls_loss)

  def compute_bbox_loss(self, bbox_preds: torch.Tensor, bbox_targets: List[torch.Tensor], 
                       indices: List[Tuple[torch.Tensor, torch.Tensor]], num_bboxes: int):
    idx = self._get_pred_permutation_idx(indices)
    pred_bboxes = bbox_preds[idx]
    target_bboxes = torch.cat([
      t[j] for t, (_, j) in zip(bbox_targets, indices)
    ], dim=0)

    bbox_loss = F.l1_loss(pred_bboxes, target_bboxes, reduction="none")
    bbox_loss = bbox_loss.sum() / num_bboxes

    giou_loss = 1 - torch.diag(box_ops.generalized_box_iou(
      box_ops.box_cxcywh_to_xyxy(pred_bboxes),
      box_ops.box_cxcywh_to_xyxy(target_bboxes)
    ))
    giou_loss = giou_loss.sum() / num_bboxes

    return dict(bbox_loss=bbox_loss, giou_loss=giou_loss)

  @torch.inference_mode()
  def compute_cardinality_loss(self, cls_logits: torch.Tensor, cls_targets: List[torch.Tensor], 
                               indices: List[Tuple[torch.Tensor, torch.Tensor]], num_bboxes: int):
    tgt_length = torch.as_tensor([len(v) for v in cls_targets], device=cls_logits.device)
    card_pred = torch.sum((
      cls_logits.argmax(-1) != cls_logits.shape[-1] - 1
    ), dim=-1)
    card_err = F.l1_loss(card_pred.float(), tgt_length.float())
    return dict(cardinality_error=card_err)

  def _get_pred_permutation_idx(self, indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

  def _get_target_permutation_idx(self, indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx


class HungarianMatcher(torch.nn.Module):
  def __init__(self, 
               *, 
               cost_class=1, 
               cost_bbox=1, 
               cost_giou=1):
    super().__init__()

    assert (
      (cost_class != 0) or (cost_bbox != 0) or (cost_giou != 0)
    ), "At least one cost term must be non-zero"

    self.cost_class = cost_class
    self.cost_bbox = cost_bbox
    self.cost_giou = cost_giou
    return
  
  @torch.inference_mode()
  def forward(self, 
              cls_logits: torch.Tensor, 
              bbox_preds: torch.Tensor, 
              cls_targets: List[torch.Tensor], 
              bbox_targets: List[torch.Tensor]):
    batch_size, num_queries, num_classes = cls_logits.shape # [B, Q, C]
    
    output_prob = cls_logits.flatten(0, 1).softmax(dim=-1) # [B * Q, C]
    output_bbox = bbox_preds.flatten(0, 1) # [B * Q, 4]

    target_ids = torch.cat(cls_targets)
    target_bbox = torch.cat(bbox_targets)

    cost_class = -output_prob[:, target_ids] # [B * Q, T]
    cost_bbox = torch.cdist(output_bbox, target_bbox, p=1) # [B * Q, T]
    cost_giou = -box_ops.generalized_box_iou(
      box_ops.box_cxcywh_to_xyxy(output_bbox),
      box_ops.box_cxcywh_to_xyxy(target_bbox)
    ) # [B * Q, T]

    C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
    C = C.view(batch_size, num_queries, -1).cpu()
    
    sizes = [len(v) for v in bbox_targets]
    indices = [
      linear_sum_assignment(c[i]) # 对每个batch元素进行匈牙利算法匹配
    for i, c in enumerate(C.split(sizes, -1))]
    return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]