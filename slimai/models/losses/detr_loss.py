import torch
import torchvision
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
               class_weight=None, 
               eos_coef=0.1, 
               cls_weight=1, 
               box_weight=5, 
               giou_weight=2, 
               use_focal_loss=True, 
               score_thresh=0.01, 
               ):
    super().__init__()
    matcher.pop("use_focal_loss", None)
    self.matcher = HungarianMatcher(**matcher, use_focal_loss=use_focal_loss)
    self.num_classes = num_classes

    self.cls_weight = cls_weight
    self.box_weight = box_weight
    self.giou_weight = giou_weight

    if class_weight is None:
      class_weight = [1.] * num_classes

    self.use_focal_loss = use_focal_loss
    self.score_thresh = score_thresh
    
    if use_focal_loss:
      class_weight = torch.as_tensor(class_weight)
    else:
      class_weight = torch.ones(self.num_classes+1) # +1 for background
      class_weight[-1] = eos_coef

    self.register_buffer("class_weight", class_weight)

    self.dist = Distributed.create()
    return
  
  def forward(self, 
              cls_logits: torch.Tensor, 
              bbox_logits: torch.Tensor, 
              cls_targets: List[torch.Tensor], 
              bbox_targets: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    # actual_num_classes = cls_logits.shape[-1]
    # assert (
    #   (self.use_focal_loss and (self.num_classes == actual_num_classes)) or
    #   ((not self.use_focal_loss) and (self.num_classes == actual_num_classes - 1))
    # ), "Class error, in focal loss mode, num_classes should not include extra background class"

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
    target_classes = torch.full(cls_logits.shape[:2], self.num_classes, dtype=torch.int64, device=cls_logits.device)
    src_idx, tgt_idx = self._get_pred_permutation_idx(indices), self._get_target_permutation_idx(indices)
    
    matched_targets = [
      cls_targets[bi][ci] for bi, ci in zip(*tgt_idx)
    ]
    if len(matched_targets) > 0:
      target_classes[src_idx] = torch.stack(matched_targets, dim=0)

    flatten_cls_logits = cls_logits.flatten(0, 1)
    flatten_target_classes = target_classes.flatten(0, 1)

    loss = dict()

    if self.use_focal_loss:
      target_classes_one_hot = F.one_hot(flatten_target_classes, 
                                         num_classes=self.num_classes+1
                                         ).type_as(flatten_cls_logits)[..., :-1] # [B * Q, C]
      focal_loss = torchvision.ops.sigmoid_focal_loss(flatten_cls_logits[..., :self.num_classes], 
                                                      target_classes_one_hot, 
                                                      reduction="none") # [B * Q, C]
      focal_loss = (focal_loss * self.class_weight).sum() / num_bboxes # type: ignore
      loss["cls_focal_loss"] = focal_loss
    else:
      cls_loss = F.cross_entropy(flatten_cls_logits, 
                                 flatten_target_classes, 
                                 self.class_weight) # type: ignore
      loss["cls_ce_loss"] = cls_loss

    loss = {k: self.cls_weight * v for (k, v) in loss.items()}
    return loss

  def compute_bbox_loss(self, bbox_preds: torch.Tensor, bbox_targets: List[torch.Tensor], 
                       indices: List[Tuple[torch.Tensor, torch.Tensor]], num_bboxes: int):
    src_idx, tgt_idx = self._get_pred_permutation_idx(indices), self._get_target_permutation_idx(indices)
    pred_bboxes = bbox_preds[src_idx]
    matched_targets = [
      bbox_targets[bi][ci] for bi, ci in zip(*tgt_idx)
    ]
    if len(matched_targets) > 0:
      target_bboxes = torch.stack(matched_targets, dim=0)
    else:
      target_bboxes = torch.empty(0, 4, device=bbox_preds.device)

    bbox_loss = F.l1_loss(pred_bboxes, target_bboxes, reduction="none")
    bbox_loss = bbox_loss.sum() / num_bboxes

    giou_loss = 1 - torch.diag(box_ops.generalized_box_iou(
      box_ops.box_cxcywh_to_xyxy(pred_bboxes),
      box_ops.box_cxcywh_to_xyxy(target_bboxes)
    ))
    giou_loss = giou_loss.sum() / num_bboxes

    loss = dict(
      bbox_loss=self.box_weight * bbox_loss, 
      giou_loss=self.giou_weight * giou_loss
    )
    return loss

  @torch.inference_mode()
  def compute_cardinality_loss(self, cls_logits: torch.Tensor, cls_targets: List[torch.Tensor], 
                               indices: List[Tuple[torch.Tensor, torch.Tensor]], num_bboxes: int):
    tgt_length = torch.as_tensor([len(v) for v in cls_targets], device=cls_logits.device)
    _, _, _, bg_mask = self.parse_cls_logits(cls_logits)
    card_pred = bg_mask.sum(dim=-1) # [B, Q] -> [B]
    card_err = F.l1_loss(card_pred.float(), tgt_length.float())
    return dict(cardinality_error=card_err)

  def parse_cls_logits(self, cls_logits: torch.Tensor):
    if self.use_focal_loss:
      cls_dist = cls_logits.sigmoid()[:, :, :self.num_classes] # [B, Q, C]
      pred_scores = cls_dist.max(-1).values # [B, Q]
      pred_labels = cls_dist.argmax(-1) # [B, Q]
      bg_mask = (pred_scores < self.score_thresh) # consider low-confidence predictions as background in focal loss # type: ignore
    else:
      cls_dist = cls_logits.softmax(-1) # [B, Q, C+1]
      pred_scores = cls_dist.max(-1).values # [B, Q]
      pred_labels = cls_dist.argmax(-1) # [B, Q]
      bg_mask = (pred_labels == self.num_classes) | (pred_scores < self.score_thresh) # consider the last class as background in non-focal loss # type: ignore
    return cls_dist, pred_scores, pred_labels, bg_mask

  def _unpack_indices(self, indices, dim):
    batch_idx, data_idx = [], []
    for i, data in enumerate(indices):
      if len(data) == 0:
        continue
      data = data[dim]
      batch_idx.append(torch.full_like(data, i))
      data_idx.append(data)
    if len(batch_idx) == 0:
      batch_idx = torch.empty(0, dtype=torch.int64)
      data_idx = torch.empty(0, dtype=torch.int64)
    else:
      batch_idx = torch.cat(batch_idx)
      data_idx = torch.cat(data_idx)
    return batch_idx, data_idx

  def _get_pred_permutation_idx(self, indices):
    # permute predictions following indices
    return self._unpack_indices(indices, 0)

  def _get_target_permutation_idx(self, indices):
    # permute targets following indices
    return self._unpack_indices(indices, 1)


class HungarianMatcher(torch.nn.Module):
  def __init__(self, 
               *, 
               cost_class=1, 
               cost_bbox=1, 
               cost_giou=1,
               use_focal_loss=True,
               focal_alpha=0.25,
               focal_gamma=2.0):
    super().__init__()

    assert (
      (cost_class != 0) or (cost_bbox != 0) or (cost_giou != 0)
    ), "At least one cost term must be non-zero"

    self.cost_class = cost_class
    self.cost_bbox = cost_bbox
    self.cost_giou = cost_giou
    self.use_focal_loss = use_focal_loss
    self.focal_alpha = focal_alpha
    self.focal_gamma = focal_gamma
    return
  
  @torch.inference_mode()
  def forward(self, 
              cls_logits: torch.Tensor, 
              bbox_preds: torch.Tensor, 
              cls_targets: List[torch.Tensor], 
              bbox_targets: List[torch.Tensor]):
    batch_size, num_queries, num_classes = cls_logits.shape # [B, Q, C]
    
    output_prob = cls_logits.flatten(0, 1) # [B * Q, C]
    if self.use_focal_loss:
      output_prob = output_prob.sigmoid()
    else:
      output_prob = output_prob.softmax(dim=-1)

    output_bbox = bbox_preds.flatten(0, 1) # [B * Q, 4]

    """
    先cat到一起统一计算cost, 此时会造成样本间匹配混乱, 即0-th的pred和1-th的target匹配在一起, 
    因此后面需要分batch求解匈牙利算法
    """
    target_ids = torch.cat(cls_targets)
    target_bbox = torch.cat(bbox_targets)

    if len(target_ids) == 0:
      return [
        torch.empty([0, 2], dtype=torch.int64)
        for _ in range(batch_size)
      ]

    if self.use_focal_loss:
      output_prob = output_prob[:, target_ids] # [B * Q, T]
      neg_cost_class = (1 - self.focal_alpha) * (output_prob ** self.focal_gamma) * (-(1 - output_prob + 1e-8).log())
      pos_cost_class = self.focal_alpha * ((1 - output_prob) ** self.focal_gamma) * (-(output_prob + 1e-8).log())
      cost_class = pos_cost_class - neg_cost_class
    else:
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
    
    return [
      (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
      for i, j in indices]