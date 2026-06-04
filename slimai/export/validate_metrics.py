from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, TYPE_CHECKING

import numpy as np
import torch

from slimai.helper import help_build
from slimai.export.validate_ort import OrtRunner, pt_outputs_to_dict
from slimai.export.validate_progress import ValidationPhaseTimer, validation_progress

if TYPE_CHECKING:
  from slimai.runner.report import ExperimentReporter


def build_test_loaders(cfg) -> Dict[str, Any]:
  import slimai.data  # noqa: F401 — register datasets/sources for help_build

  loaders: Dict[str, Any] = {}
  test_cfg = cfg.get("TEST_LOADER", dict())
  if isinstance(test_cfg, dict) and len(test_cfg) > 0:
    test_loader = help_build.build_dataloader(test_cfg)
    if test_loader is not None:
      loaders["inner_test"] = test_loader
  external_cfgs = cfg.get("EXTERNAL_TEST_LOADERS", {}) or {}
  if isinstance(external_cfgs, dict):
    for name, loader_cfg in external_cfgs.items():
      if loader_cfg is None:
        continue
      loader = help_build.build_dataloader(loader_cfg)
      if loader is not None:
        loaders[f"external_{name}"] = loader
  return loaders


def collect_samples_from_dataset(dataset) -> List[Dict[str, Any]]:
  samples: List[Dict[str, Any]] = []
  for idx in dataset.indices:
    embed_path = dataset.files[idx]
    embedding, _coords = dataset.get_embedding(embed_path)
    if hasattr(embedding, "detach"):
      embedding_np = embedding.detach().cpu().numpy()
    else:
      embedding_np = np.asarray(embedding)
    sample = dict(
      h5_path=str(embed_path),
      embedding=np.asarray(embedding_np, dtype=np.float32),
      label=int(dataset.annotations["label"][idx]),
    )
    if "label_secondary" in dataset.annotations:
      sample["label_secondary"] = int(dataset.annotations["label_secondary"][idx])
    samples.append(sample)
  return samples


def merge_sample_lists(lists: Sequence[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
  merged: List[Dict[str, Any]] = []
  for items in lists:
    merged.extend(list(items))
  return merged


def parse_primary_secondary_preds(outputs: Mapping[str, np.ndarray], *, is_hierarchical: bool) -> Dict[str, Any]:
  if not is_hierarchical:
    softmax = outputs.get("softmax")
    logits = outputs.get("logits")
    if softmax is not None:
      prob = np.asarray(softmax, dtype=np.float64).reshape(-1)
    else:
      logits_arr = np.asarray(logits, dtype=np.float64).reshape(-1)
      exp = np.exp(logits_arr - np.max(logits_arr))
      prob = exp / np.sum(exp)
    pred = int(np.argmax(prob))
    return dict(primary_pred=pred, primary_prob=prob)
  primary_prob = np.asarray(outputs["primary_softmax"], dtype=np.float64).reshape(-1)
  primary_pred = int(np.argmax(primary_prob))
  secondary_marginal = int(np.asarray(outputs["secondary_marginal_label"]).reshape(-1)[0])
  secondary_conditional = int(np.asarray(outputs["secondary_conditional_label"]).reshape(-1)[0])
  secondary_prob = np.asarray(outputs["secondary_marginal_softmax"], dtype=np.float64).reshape(-1)
  return dict(
    primary_pred=primary_pred,
    primary_prob=primary_prob,
    secondary_marginal_pred=secondary_marginal,
    secondary_marginal_prob=secondary_prob,
    secondary_conditional_pred=secondary_conditional,
  )


def run_slide_encoder_pt(slide_encoder: torch.nn.Module, embedding: np.ndarray, output_names: Sequence[str]) -> Dict[str, np.ndarray]:
  tensor = torch.from_numpy(np.asarray(embedding, dtype=np.float32))
  with torch.inference_mode():
    outputs = slide_encoder(tensor)
  return pt_outputs_to_dict(outputs, output_names)


def run_slide_encoder_ort(runner: OrtRunner, embedding: np.ndarray) -> Dict[str, np.ndarray]:
  feed = {"embedding_arr": np.asarray(embedding, dtype=np.float32)}
  return runner.run(feed)


def evaluate_slide_sample(
  sample: Dict[str, Any],
  *,
  slide_encoder: torch.nn.Module,
  slide_runner: OrtRunner,
  slide_output_names: Sequence[str],
  is_hierarchical: bool,
) -> Dict[str, Any]:
  embedding = sample["embedding"]
  pt_out = run_slide_encoder_pt(slide_encoder, embedding, slide_output_names)
  ort_out = run_slide_encoder_ort(slide_runner, embedding)
  pt_preds = parse_primary_secondary_preds(pt_out, is_hierarchical=is_hierarchical)
  ort_preds = parse_primary_secondary_preds(ort_out, is_hierarchical=is_hierarchical)
  label_agreement = dict(primary=pt_preds["primary_pred"] == ort_preds["primary_pred"])
  if is_hierarchical:
    label_agreement["secondary_marginal"] = pt_preds["secondary_marginal_pred"] == ort_preds["secondary_marginal_pred"]
    label_agreement["secondary_conditional"] = pt_preds["secondary_conditional_pred"] == ort_preds["secondary_conditional_pred"]
  eval_sample = dict(
    h5_path=sample.get("h5_path", ""),
    label=int(sample["label"]),
    pred=int(ort_preds["primary_pred"]),
    prob=np.asarray(ort_preds["primary_prob"], dtype=np.float64),
    pt_pred=int(pt_preds["primary_pred"]),
    label_agreement=label_agreement,
  )
  if is_hierarchical and "label_secondary" in sample:
    eval_sample["label_secondary"] = int(sample["label_secondary"])
    eval_sample["pred_secondary"] = int(ort_preds["secondary_marginal_pred"])
    eval_sample["prob_secondary"] = np.asarray(ort_preds["secondary_marginal_prob"], dtype=np.float64)
    eval_sample["conditional_pred"] = int(ort_preds["secondary_conditional_pred"])
    eval_sample["conditional_prob"] = np.asarray(ort_preds["secondary_marginal_prob"], dtype=np.float64)
    eval_sample["pt_pred_secondary"] = int(pt_preds["secondary_marginal_pred"])
    eval_sample["pt_conditional_pred"] = int(pt_preds["secondary_conditional_pred"])
  return eval_sample


def _samples_for_level1_metrics(eval_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  return [dict(label=s["label"], pred=s["pred"], prob=s["prob"], h5_path=s.get("h5_path", "")) for s in eval_samples]


def _samples_for_level2_metrics(eval_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  rows = []
  for s in eval_samples:
    if "label_secondary" not in s:
      continue
    rows.append(
      dict(
        label=s["label_secondary"],
        pred=s["pred_secondary"],
        prob=s["prob_secondary"],
        conditional_pred=s.get("conditional_pred"),
        conditional_prob=s.get("conditional_prob"),
        h5_path=s.get("h5_path", ""),
      )
    )
  return rows


def compute_label_agreement_rate(eval_samples: List[Dict[str, Any]], *, is_hierarchical: bool) -> Dict[str, Any]:
  if len(eval_samples) == 0:
    return dict(passed=False, rates={})
  keys = ["primary"]
  if is_hierarchical:
    keys.extend(["secondary_marginal", "secondary_conditional"])
  rates = {}
  passed = True
  for key in keys:
    matches = [bool(s.get("label_agreement", {}).get(key, False)) for s in eval_samples]
    rate = float(np.mean(matches)) if len(matches) > 0 else 0.0
    rates[key] = rate
    if rate < 1.0:
      passed = False
  return dict(passed=passed, rates=rates)




def _resolve_center_path_map(reporter: ExperimentReporter, eval_samples: List[Dict[str, Any]]) -> Dict[str, str]:
  split_file = (reporter.dataset_info.get("test") or {}).get("split_file", None)
  center_path_map = reporter._load_center_path_map(split_file=split_file, split_name="test")
  if len(center_path_map) == 0 and reporter._get_report_cfg().get("inner_test_center_path_heuristic", True):
    center_path_map = reporter._build_heuristic_center_path_map([s.get("h5_path", "") for s in eval_samples])
  return center_path_map


def _group_eval_samples_by_center(
  eval_samples: List[Dict[str, Any]],
  center_path_map: Dict[str, str],
  reporter: ExperimentReporter,
) -> Dict[str, List[Dict[str, Any]]]:
  grouped: Dict[str, List[Dict[str, Any]]] = {}
  for sample in eval_samples:
    center_name = reporter._resolve_center_for_h5_path(sample.get("h5_path", ""), center_path_map)
    if center_name == "":
      continue
    grouped.setdefault(center_name, []).append(sample)
  return grouped


def _build_inner_by_center_block(
  eval_samples: List[Dict[str, Any]],
  *,
  reporter: ExperimentReporter,
  is_hierarchical: bool,
) -> Dict[str, Any]:
  center_path_map = _resolve_center_path_map(reporter, eval_samples)
  if not reporter._should_group_inner_test_by_center(center_path_map):
    return {}
  grouped_eval = _group_eval_samples_by_center(eval_samples, center_path_map, reporter)
  by_center = {}
  for center_name, center_eval in grouped_eval.items():
    center_samples = [
      dict(h5_path=s.get("h5_path", ""), label=s["label"], pred=s["pred"], prob=s["prob"])
      for s in center_eval
    ]
    labels, preds, probs = reporter._samples_to_arrays(center_samples)
    metrics = reporter._compute_classification_metrics_from_arrays(labels, preds, probs) if labels is not None else {}
    figures = reporter.build_eval_figures(
      result_file=Path("."),
      file_prefix=f"v4_inner_{center_name}",
      samples=center_samples,
      class_names=reporter.display_class_names,
    )
    agreement = compute_label_agreement_rate(center_eval, is_hierarchical=is_hierarchical)
    by_center[center_name] = dict(metrics=metrics, figures=figures, label_agreement=agreement)
  return by_center


def run_subset_evaluation(
  subset_name: str,
  raw_samples: List[Dict[str, Any]],
  *,
  slide_encoder: torch.nn.Module,
  slide_runner: OrtRunner,
  slide_output_names: Sequence[str],
  is_hierarchical: bool,
  reporter: ExperimentReporter,
  show_progress: bool,
) -> Dict[str, Any]:
  eval_samples: List[Dict[str, Any]] = []
  for sample in validation_progress(raw_samples, total=len(raw_samples), desc=f"V4 {subset_name}", unit="slide", enabled=show_progress, leave=False):
    eval_samples.append(
      evaluate_slide_sample(
        sample,
        slide_encoder=slide_encoder,
        slide_runner=slide_runner,
        slide_output_names=slide_output_names,
        is_hierarchical=is_hierarchical,
      )
    )
  level1_samples = _samples_for_level1_metrics(eval_samples)
  labels, preds, probs = reporter._samples_to_arrays(level1_samples)
  level1_metrics = reporter._compute_classification_metrics_from_arrays(labels, preds, probs) if labels is not None else {}
  level2_metrics = {}
  level2_figures = {}
  if is_hierarchical:
    level2_samples = _samples_for_level2_metrics(eval_samples)
    l2_labels, l2_preds, l2_probs = reporter._samples_to_arrays(level2_samples)
    if l2_labels is not None:
      level2_metrics = reporter._compute_classification_metrics_from_arrays(l2_labels, l2_preds, l2_probs)
      cond_samples = [dict(label=s["label"], pred=s["conditional_pred"], prob=s["conditional_prob"]) for s in level2_samples if s.get("conditional_pred") is not None]
      cl, cp, cpr = reporter._samples_to_arrays(cond_samples)
      if cl is not None:
        cond_metrics = reporter._compute_classification_metrics_from_arrays(cl, cp, cpr)
        level2_metrics.update({f"conditional_{k}": v for k, v in cond_metrics.items() if k != "n_samples"})
    level2_figures = reporter.build_eval_figures(result_file=Path("."), file_prefix=f"v4_{subset_name}", samples=level2_samples, class_names=reporter.display_secondary_class_names)
  level1_figures = reporter.build_eval_figures(result_file=Path("."), file_prefix=f"v4_{subset_name}", samples=level1_samples, class_names=reporter.display_class_names)
  agreement = compute_label_agreement_rate(eval_samples, is_hierarchical=is_hierarchical)
  by_center = {}
  if subset_name == "inner_test":
    by_center = _build_inner_by_center_block(eval_samples, reporter=reporter, is_hierarchical=is_hierarchical)
  return dict(name=subset_name, n_samples=len(eval_samples), level1=dict(metrics=level1_metrics, figures=level1_figures), level2=dict(metrics=level2_metrics, figures=level2_figures), label_agreement=agreement, by_center=by_center, passed=agreement.get("passed", False))


def _baseline_metrics_for_subset(baseline: Dict[str, Any], subset_name: str, level: str) -> Dict[str, float]:
  if subset_name == "full_test":
    return {}
  level_block = baseline.get(level, {}) or {}
  if subset_name == "inner_test":
    raw = level_block.get("inner_test", {})
  elif subset_name.startswith("external_"):
    raw = (level_block.get("external", {}) or {}).get(subset_name, {})
  else:
    raw = {}
  return {k: float(v) for k, v in (raw or {}).items() if v is not None}


def _metrics_to_float_dict(metrics: Dict[str, Any]) -> Dict[str, float]:
  result: Dict[str, float] = {}
  for key, value in metrics.items():
    if key == "n_samples":
      continue
    if value is None or value == "N/A":
      continue
    try:
      result[key] = float(value)
    except Exception:
      continue
  return result


def _attach_reference_compare(subsets, baseline, metrics_tol: float):
  if baseline is None:
    return
  from slimai.export.validate_reference import compare_metric_tables

  for subset_name, block in subsets.items():
    if subset_name == "full_test":
      block.setdefault("level1", {})["reference_compare_note"] = "no PyTorch baseline in original report"
      block.setdefault("level2", {})["reference_compare_note"] = "no PyTorch baseline in original report"
      continue
    onnx_l1 = block.get("level1", {}).get("metrics", {})
    block.setdefault("level1", {})["reference_compare"] = compare_metric_tables(
      _metrics_to_float_dict(onnx_l1),
      _baseline_metrics_for_subset(baseline, subset_name, "level1"),
      metrics_tol=metrics_tol,
    )
    onnx_l2 = block.get("level2", {}).get("metrics", {})
    if onnx_l2:
      block.setdefault("level2", {})["reference_compare"] = compare_metric_tables(
        _metrics_to_float_dict(onnx_l2),
        _baseline_metrics_for_subset(baseline, subset_name, "level2"),
        metrics_tol=metrics_tol,
      )


def run_v4_test_evaluation(
  *,
  cfg,
  slide_encoder: torch.nn.Module,
  slide_onnx_path: Path,
  slide_output_names: Sequence[str],
  is_hierarchical: bool,
  reference_work_dir: Optional[Path],
  skip_reference_compare: bool,
  metrics_tol: float,
  ort_provider: str,
  show_progress: bool,
  timing: Dict[str, Any],
) -> Dict[str, Any]:
  from slimai.runner.report import ExperimentReporter

  loaders = build_test_loaders(cfg)
  if len(loaders) == 0:
    return dict(skipped=True, reason="no test loaders", passed=True, subsets={})
  slide_runner = OrtRunner(slide_onnx_path, provider=ort_provider)
  work_dir = Path(reference_work_dir) if reference_work_dir is not None else None
  reporter = ExperimentReporter(
    work_dir=work_dir or Path("."),
    cfg=cfg,
    test_dataloader=loaders.get("inner_test"),
    external_test_dataloaders={key.replace("external_", ""): loaders[key] for key in loaders if key.startswith("external_")},
  )
  baseline = None
  if not skip_reference_compare and work_dir is not None:
    from slimai.export.validate_reference import load_experiment_report_baseline

    with ValidationPhaseTimer(timing, "v4_baseline_parse"):
      baseline = load_experiment_report_baseline(work_dir)
  raw_by_subset: Dict[str, List[Dict[str, Any]]] = {}
  for subset_name, loader in loaders.items():
    with ValidationPhaseTimer(timing, f"v4_collect_{subset_name}"):
      raw_by_subset[subset_name] = collect_samples_from_dataset(loader.dataset)
  if "inner_test" in raw_by_subset:
    merged = [raw_by_subset["inner_test"]]
    merged.extend([raw_by_subset[k] for k in sorted(raw_by_subset.keys()) if k.startswith("external_")])
    raw_by_subset["full_test"] = merge_sample_lists(merged)
  subsets: Dict[str, Any] = {}
  for subset_name, raw_samples in raw_by_subset.items():
    with ValidationPhaseTimer(timing, f"v4_eval_{subset_name}"):
      subsets[subset_name] = run_subset_evaluation(
        subset_name,
        raw_samples,
        slide_encoder=slide_encoder,
        slide_runner=slide_runner,
        slide_output_names=slide_output_names,
        is_hierarchical=is_hierarchical,
        reporter=reporter,
        show_progress=show_progress,
      )
  _attach_reference_compare(subsets, baseline, metrics_tol)
  external_count = len([k for k in subsets.keys() if k.startswith("external_")])
  if "full_test" in subsets and external_count == 0:
    subsets["full_test"]["note"] = "external_count=0, same slides as inner_test"
  passed = all(block.get("passed", False) for block in subsets.values())
  return dict(subsets=subsets, external_count=external_count, baseline=baseline, passed=passed)
