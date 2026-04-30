from typing import Dict, Optional
import torch
from slimai.helper.help_build import MODELS, build_metric


@MODELS.register_module()
class BasicClassificationMetric(torch.nn.Module):
  def __init__(self, 
               acc=dict(
                  type="torchmetrics.Accuracy",
                  task="multiclass", 
                  num_classes=None,
                  sync_on_compute=False,
                ), 
                auc=dict(
                  type="torchmetrics.AUROC",
                  task="multiclass",
                  num_classes=None,
                  sync_on_compute=False,
                ), 
               kappa=dict(
                  type="torchmetrics.CohenKappa",
                  task="multiclass",
                  num_classes=None,
                  sync_on_compute=False,
                ),
               f1=None,
              ):
    super().__init__()
    self.acc = build_metric(acc)
    self.auc = build_metric(auc)
    self.kappa = build_metric(kappa)
    if f1 is None:
      f1 = dict(
        type="torchmetrics.F1Score",
        task="multiclass",
        average="macro",
        num_classes=acc.get("num_classes", None),
        sync_on_compute=False,
      )
    self.f1 = build_metric(f1)
    return

  @torch.inference_mode()
  def forward(self, 
              output: Dict[str, torch.Tensor], 
              targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    softmax = output["softmax"]
    labels = targets["label"]
    valid_mask = labels >= 0
    if valid_mask.any():
      softmax = softmax[valid_mask]
      labels = labels[valid_mask]
    else:
      nan = torch.tensor(float("nan"), device=softmax.device)
      return dict(
        acc=nan,
        auc=nan,
        kappa=nan,
        f1=nan,
        precision_per_class=torch.zeros((0,), dtype=torch.float32, device=softmax.device),
        recall_per_class=torch.zeros((0,), dtype=torch.float32, device=softmax.device),
        f1_per_class=torch.zeros((0,), dtype=torch.float32, device=softmax.device),
        support_per_class=torch.zeros((0,), dtype=torch.float32, device=softmax.device),
        precision_micro=nan,
        recall_micro=nan,
        f1_micro=nan,
        f1_macro_valid=nan,
      )
    metrics = dict(
      acc=self.acc(softmax, labels),
      auc=self.auc(softmax, labels),
      kappa=self.kappa(softmax, labels),
      f1=self.f1(softmax, labels),
    )
    preds = softmax.argmax(dim=1)
    num_classes = int(softmax.shape[1])
    precision, recall, f1_per_class, support, precision_micro, recall_micro, f1_micro, f1_macro_valid = \
      self._compute_classwise_metrics(preds, labels, num_classes=num_classes)
    metrics.update(
      precision_per_class=precision,
      recall_per_class=recall,
      f1_per_class=f1_per_class,
      support_per_class=support,
      precision_micro=precision_micro,
      recall_micro=recall_micro,
      f1_micro=f1_micro,
      f1_macro_valid=f1_macro_valid,
    )
    return metrics

  def _compute_classwise_metrics(self, preds: torch.Tensor, labels: torch.Tensor, *, num_classes: int):
    device = preds.device
    precision = torch.full((num_classes,), float("nan"), dtype=torch.float32, device=device)
    recall = torch.full((num_classes,), float("nan"), dtype=torch.float32, device=device)
    f1_per_class = torch.full((num_classes,), float("nan"), dtype=torch.float32, device=device)
    support = torch.zeros((num_classes,), dtype=torch.float32, device=device)

    tp_total = torch.zeros((), dtype=torch.float32, device=device)
    fp_total = torch.zeros((), dtype=torch.float32, device=device)
    fn_total = torch.zeros((), dtype=torch.float32, device=device)
    for class_idx in range(num_classes):
      pred_pos = (preds == class_idx)
      true_pos = (labels == class_idx)
      tp = (pred_pos & true_pos).sum().to(torch.float32)
      fp = (pred_pos & (~true_pos)).sum().to(torch.float32)
      fn = ((~pred_pos) & true_pos).sum().to(torch.float32)
      sup = true_pos.sum().to(torch.float32)
      support[class_idx] = sup
      tp_total += tp
      fp_total += fp
      fn_total += fn
      if (tp + fp) > 0:
        precision[class_idx] = tp / (tp + fp)
      if (tp + fn) > 0:
        recall[class_idx] = tp / (tp + fn)
      if torch.isfinite(precision[class_idx]) and torch.isfinite(recall[class_idx]) and (precision[class_idx] + recall[class_idx]) > 0:
        f1_per_class[class_idx] = 2 * precision[class_idx] * recall[class_idx] / (precision[class_idx] + recall[class_idx])

    precision_micro = tp_total / (tp_total + fp_total + 1e-12)
    recall_micro = tp_total / (tp_total + fn_total + 1e-12)
    f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro + 1e-12)
    valid_mask = support > 0
    if valid_mask.any():
      f1_macro_valid = torch.nanmean(f1_per_class[valid_mask])
    else:
      f1_macro_valid = torch.tensor(float("nan"), dtype=torch.float32, device=device)
    return precision, recall, f1_per_class, support, precision_micro, recall_micro, f1_micro, f1_macro_valid


@MODELS.register_module()
class HierarchicalClassificationMetric(torch.nn.Module):
  def __init__(
    self,
    *,
    primary: Optional[dict] = None,
    secondary: Optional[dict] = None,
    include_secondary_conditional: bool = True,
    secondary_to_primary_idx: Optional[list] = None,
  ):
    super().__init__()
    if primary is None:
      primary = dict(type="BasicClassificationMetric")
    if secondary is None:
      secondary = dict(type="BasicClassificationMetric")
    self.primary_metric = build_metric(primary)
    self.secondary_metric = build_metric(secondary)
    self.include_secondary_conditional = include_secondary_conditional
    self.secondary_to_primary_idx = list(secondary_to_primary_idx or [])
    return

  def _get_task_output(self, output: Dict[str, torch.Tensor], task_key: str):
    if task_key in output and isinstance(output[task_key], dict):
      return output[task_key]
    if task_key == "label":
      return dict(softmax=output["softmax"])
    raise KeyError(f"Task output key not found: {task_key}")

  @torch.inference_mode()
  def forward(
    self,
    output: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
  ) -> Dict[str, torch.Tensor]:
    results = {}
    primary_output = self._get_task_output(output, "label")
    primary_targets = dict(label=targets["label"])
    primary_metrics = self.primary_metric(primary_output, primary_targets)
    for key, value in primary_metrics.items():
      results[f"label_{key}"] = value

    if "label_secondary" in targets and "label_secondary" in output:
      secondary_output = self._get_task_output(output, "label_secondary")
      secondary_targets = dict(label=targets["label_secondary"])
      secondary_metrics = self.secondary_metric(secondary_output, secondary_targets)
      for key, value in secondary_metrics.items():
        results[f"label_secondary_{key}"] = value

      if len(self.secondary_to_primary_idx) > 0 and "labels" in secondary_output and "labels" in primary_output:
        secondary_pred = secondary_output["labels"].to(torch.int64)
        primary_pred = primary_output["labels"].to(torch.int64)
        mapping = torch.as_tensor(
          self.secondary_to_primary_idx,
          device=secondary_pred.device,
          dtype=torch.int64,
        )
        valid_mask = (secondary_pred >= 0) & (secondary_pred < len(self.secondary_to_primary_idx))
        if valid_mask.any():
          secondary_parent_pred = mapping[secondary_pred[valid_mask]]
          consistency = (secondary_parent_pred == primary_pred[valid_mask]).to(torch.float32).mean()
        else:
          consistency = torch.tensor(float("nan"), device=secondary_pred.device)
        results["label_secondary_parent_consistency"] = consistency

    if (
      self.include_secondary_conditional
      and "label_secondary" in targets
      and "label_secondary_conditional" in output
    ):
      conditional_output = self._get_task_output(output, "label_secondary_conditional")
      conditional_targets = dict(label=targets["label_secondary"])
      conditional_metrics = self.secondary_metric(conditional_output, conditional_targets)
      for key, value in conditional_metrics.items():
        results[f"label_secondary_conditional_{key}"] = value

    return results
