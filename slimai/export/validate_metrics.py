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
    if "label_secondary_name" in dataset.annotations:
      raw_name = dataset.annotations["label_secondary_name"][idx]
      if raw_name is not None and str(raw_name).strip() != "":
        sample["label_secondary_name"] = str(raw_name)
    samples.append(sample)
  return samples


def merge_sample_lists(lists: Sequence[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
  merged: List[Dict[str, Any]] = []
  for items in lists:
    merged.extend(list(items))
  return merged


def parse_hierarchical_preds(outputs: Mapping[str, np.ndarray]) -> Dict[str, Any]:
  primary_prob = np.asarray(outputs["primary_softmax"], dtype=np.float64).reshape(-1)
  primary_pred = int(np.argmax(primary_prob))
  marginal_prob = np.asarray(outputs["marginal_softmax"], dtype=np.float64).reshape(-1)
  marginal_pred = int(np.asarray(outputs["marginal_label"]).reshape(-1)[0])
  conditional_prob = np.asarray(outputs["conditional_softmax"], dtype=np.float64).reshape(-1)
  conditional_pred = int(np.asarray(outputs["conditional_label"]).reshape(-1)[0])
  return dict(
    primary_pred=primary_pred,
    primary_prob=primary_prob,
    marginal_pred=marginal_pred,
    marginal_prob=marginal_prob,
    conditional_pred=conditional_pred,
    conditional_prob=conditional_prob,
  )


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
  parsed = parse_hierarchical_preds(outputs)
  return dict(
    primary_pred=parsed["primary_pred"],
    primary_prob=parsed["primary_prob"],
    secondary_marginal_pred=parsed["marginal_pred"],
    secondary_marginal_prob=parsed["marginal_prob"],
    secondary_conditional_pred=parsed["conditional_pred"],
    secondary_conditional_prob=parsed["conditional_prob"],
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
    pt_prob=np.asarray(pt_preds["primary_prob"], dtype=np.float64),
    label_agreement=label_agreement,
  )
  if is_hierarchical and "label_secondary" in sample:
    eval_sample["label_secondary"] = int(sample["label_secondary"])
    if sample.get("label_secondary_name"):
      eval_sample["label_secondary_name"] = str(sample["label_secondary_name"])
    eval_sample["pred_secondary"] = int(ort_preds["secondary_marginal_pred"])
    eval_sample["prob_secondary"] = np.asarray(ort_preds["secondary_marginal_prob"], dtype=np.float64)
    eval_sample["pt_pred_secondary"] = int(pt_preds["secondary_marginal_pred"])
    eval_sample["pt_prob_secondary"] = np.asarray(pt_preds["secondary_marginal_prob"], dtype=np.float64)
    eval_sample["conditional_pred"] = int(ort_preds["secondary_conditional_pred"])
    eval_sample["conditional_prob"] = np.asarray(ort_preds["secondary_conditional_prob"], dtype=np.float64)
    eval_sample["pt_conditional_pred"] = int(pt_preds["secondary_conditional_pred"])
    eval_sample["pt_conditional_prob"] = np.asarray(pt_preds["secondary_conditional_prob"], dtype=np.float64)
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
        label_secondary_name=s.get("label_secondary_name"),
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


def _compare_metric_dicts(metrics_pt: Dict[str, Any], metrics_ort: Dict[str, Any], *, metrics_tol: float) -> Dict[str, Any]:
  parity: Dict[str, Any] = {}
  all_match = True
  for key in sorted(set(metrics_pt.keys()) | set(metrics_ort.keys())):
    if key == "n_samples":
      continue
    pt_val = metrics_pt.get(key)
    ort_val = metrics_ort.get(key)
    if pt_val is None or ort_val is None:
      continue
    try:
      pt_f = float(pt_val)
      ort_f = float(ort_val)
    except Exception:
      continue
    delta = abs(pt_f - ort_f)
    match = delta <= metrics_tol
    parity[key] = dict(pt=pt_f, ort=ort_f, delta=delta, tol=metrics_tol, match=match)
    if not match:
      all_match = False
  return dict(metrics=parity, passed=all_match)


def _task_samples_from_eval(
  eval_samples: List[Dict[str, Any]],
  *,
  task: str,
  backend: str,
) -> List[Dict[str, Any]]:
  rows = []
  for sample in eval_samples:
    if task == "primary":
      label = sample.get("label")
      pred_key = "pt_pred" if backend == "pt" else "pred"
      prob_key = "pt_prob" if backend == "pt" else "prob"
    elif task == "marginal":
      if "label_secondary" not in sample:
        continue
      label = sample["label_secondary"]
      pred_key = "pt_pred_secondary" if backend == "pt" else "pred_secondary"
      prob_key = "pt_prob_secondary" if backend == "pt" else "prob_secondary"
    elif task == "conditional":
      if "label_secondary" not in sample:
        continue
      label = sample["label_secondary"]
      pred_key = "pt_conditional_pred" if backend == "pt" else "conditional_pred"
      prob_key = "pt_conditional_prob" if backend == "pt" else "conditional_prob"
    else:
      raise ValueError(f"Unknown task: {task}")
    if pred_key not in sample or prob_key not in sample:
      continue
    rows.append(dict(label=label, pred=sample[pred_key], prob=sample[prob_key], h5_path=sample.get("h5_path", "")))
    if sample.get("label_secondary_name"):
      rows[-1]["label_secondary_name"] = str(sample["label_secondary_name"])
  return rows


def _task_to_report_task_key(task: str) -> Optional[str]:
  if task == "primary":
    return "label"
  if task == "marginal":
    return "label_secondary"
  if task == "conditional":
    return "label_secondary_conditional"
  return None


def _evaluate_task_block(
  eval_samples: List[Dict[str, Any]],
  *,
  task: str,
  reporter: ExperimentReporter,
  subset_name: str,
  class_names: Sequence[str],
  metrics_tol: float,
) -> Dict[str, Any]:
  pt_samples = _task_samples_from_eval(eval_samples, task=task, backend="pt")
  ort_samples = _task_samples_from_eval(eval_samples, task=task, backend="ort")
  pt_labels, pt_preds, pt_probs = reporter._samples_to_arrays(pt_samples)
  ort_labels, ort_preds, ort_probs = reporter._samples_to_arrays(ort_samples)
  metrics_pt = reporter._compute_classification_metrics_from_arrays(pt_labels, pt_preds, pt_probs) if pt_labels is not None else {}
  metrics_ort = reporter._compute_classification_metrics_from_arrays(ort_labels, ort_preds, ort_probs) if ort_labels is not None else {}
  metric_parity = _compare_metric_dicts(metrics_pt, metrics_ort, metrics_tol=metrics_tol)
  figures_ort = reporter.build_eval_figures(
    result_file=Path("."),
    file_prefix=f"v4_{subset_name}_{task}",
    samples=ort_samples,
    class_names=class_names,
    task_key=_task_to_report_task_key(task),
  )
  return dict(
    metrics_pt=metrics_pt,
    metrics_ort=metrics_ort,
    metric_parity=metric_parity,
    figures_ort=figures_ort,
    passed=metric_parity.get("passed", True),
  )


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
  metrics_tol: float = 1e-4,
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
  agreement = compute_label_agreement_rate(eval_samples, is_hierarchical=is_hierarchical)
  if not is_hierarchical:
    primary_block = _evaluate_task_block(
      eval_samples,
      task="primary",
      reporter=reporter,
      subset_name=subset_name,
      class_names=reporter.display_class_names,
      metrics_tol=metrics_tol,
    )
    passed = agreement.get("passed", False) and primary_block.get("passed", False)
    return dict(
      name=subset_name,
      n_samples=len(eval_samples),
      primary=primary_block,
      label_agreement=agreement,
      passed=passed,
    )

  primary_block = _evaluate_task_block(
    eval_samples, task="primary", reporter=reporter, subset_name=subset_name,
    class_names=reporter.display_class_names, metrics_tol=metrics_tol,
  )
  marginal_block = _evaluate_task_block(
    eval_samples, task="marginal", reporter=reporter, subset_name=subset_name,
    class_names=reporter.display_secondary_class_names, metrics_tol=metrics_tol,
  )
  conditional_block = _evaluate_task_block(
    eval_samples, task="conditional", reporter=reporter, subset_name=subset_name,
    class_names=reporter.display_secondary_class_names, metrics_tol=metrics_tol,
  )
  passed = (
    agreement.get("passed", False)
    and primary_block.get("passed", False)
    and marginal_block.get("passed", False)
    and conditional_block.get("passed", False)
  )
  by_center = {}
  if subset_name == "inner_test":
    by_center = _build_inner_by_center_block(eval_samples, reporter=reporter, is_hierarchical=is_hierarchical)
  return dict(
    name=subset_name,
    n_samples=len(eval_samples),
    primary=primary_block,
    marginal=marginal_block,
    conditional=conditional_block,
    label_agreement=agreement,
    by_center=by_center,
    passed=passed,
  )


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
      continue
    for task_name in ("primary", "marginal", "conditional"):
      task_block = block.get(task_name, {})
      if not task_block:
        continue
      level_key = "level1" if task_name == "primary" else "level2"
      baseline_metrics = {}
      if task_name == "primary":
        baseline_metrics = (baseline.get(level_key, {}) or {}).get(
          "inner_test" if subset_name == "inner_test" else "external", {}
        )
        if subset_name.startswith("external_"):
          baseline_metrics = (baseline.get(level_key, {}).get("external", {}) or {}).get(subset_name, {})
      elif task_name in ("marginal", "conditional"):
        baseline_metrics = (baseline.get("level2", {}) or {}).get(
          "inner_test" if subset_name == "inner_test" else "external", {}
        )
        if subset_name.startswith("external_"):
          baseline_metrics = (baseline.get("level2", {}).get("external", {}) or {}).get(subset_name, {})
      if not baseline_metrics:
        task_block["reference_compare_note"] = "no PyTorch baseline in original report"
        continue
      task_block["reference_compare"] = compare_metric_tables(
        _metrics_to_float_dict(task_block.get("metrics_ort", {})),
        baseline_metrics,
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
        metrics_tol=metrics_tol,
      )
  _attach_reference_compare(subsets, baseline, metrics_tol)
  external_count = len([k for k in subsets.keys() if k.startswith("external_")])
  if "full_test" in subsets and external_count == 0:
    subsets["full_test"]["note"] = "external_count=0, same slides as inner_test"
  passed = all(block.get("passed", False) for block in subsets.values())
  return dict(subsets=subsets, external_count=external_count, baseline=baseline, passed=passed)
