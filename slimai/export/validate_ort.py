from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

PATCH_OUTPUT = "embedding_arr"


def error_stats(diff: np.ndarray) -> Dict[str, float]:
  flat = np.asarray(diff, dtype=np.float64).reshape(-1)
  if flat.size == 0:
    return dict(min=0.0, max=0.0, mean=0.0, variance=0.0)
  return dict(
    min=float(np.min(flat)),
    max=float(np.max(flat)),
    mean=float(np.mean(flat)),
    variance=float(np.var(flat, ddof=0)),
  )


def aggregate_trial_maxes(per_trial_max: Sequence[float]) -> Dict[str, Any]:
  values = [float(v) for v in per_trial_max]
  if len(values) == 0:
    return dict(min=0.0, max=0.0, mean=0.0, variance=0.0, per_trial_max=[])
  arr = np.asarray(values, dtype=np.float64)
  return dict(
    min=float(np.min(arr)),
    max=float(np.max(arr)),
    mean=float(np.mean(arr)),
    variance=float(np.var(arr, ddof=0)) if arr.size > 1 else 0.0,
    per_trial_max=values,
  )


class OrtRunner:
  def __init__(self, onnx_path: Union[str, Path], *, provider: str = "CPUExecutionProvider"):
    import onnxruntime as ort

    self.onnx_path = Path(onnx_path)
    self.provider = provider
    providers = [provider]
    if provider != "CPUExecutionProvider":
      providers.append("CPUExecutionProvider")
    self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
    self.input_names = [item.name for item in self.session.get_inputs()]
    self.output_names = [item.name for item in self.session.get_outputs()]
    return

  def run(self, feed_dict: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    outputs = self.session.run(self.output_names, dict(feed_dict))
    return {name: np.asarray(value) for name, value in zip(self.output_names, outputs)}


def run_ort(
  onnx_path: Union[str, Path],
  feed_dict: Mapping[str, np.ndarray],
  *,
  provider: str = "CPUExecutionProvider",
  runner: Optional[OrtRunner] = None,
) -> Dict[str, np.ndarray]:
  if runner is None:
    runner = OrtRunner(onnx_path, provider=provider)
  return runner.run(feed_dict)


def _is_int_output(name: str, arr: np.ndarray) -> bool:
  if "label" in name.lower():
    return True
  return np.issubdtype(arr.dtype, np.integer)


def float_error_passed(error: Mapping[str, float], *, max_tol: float, mean_tol: float) -> bool:
  return float(error.get("max", 0.0)) < max_tol and float(error.get("mean", 0.0)) < mean_tol


def compare_output_pair(
  name: str,
  pt_arr: np.ndarray,
  ort_arr: np.ndarray,
  *,
  parity_max_tol: float,
  parity_mean_tol: float,
) -> Dict[str, Any]:
  pt_arr = np.asarray(pt_arr)
  ort_arr = np.asarray(ort_arr)
  if _is_int_output(name, pt_arr) or _is_int_output(name, ort_arr):
    exact = bool(np.array_equal(pt_arr, ort_arr))
    scalar_pt = int(np.asarray(pt_arr).reshape(-1)[0]) if pt_arr.size > 0 else None
    scalar_ort = int(np.asarray(ort_arr).reshape(-1)[0]) if ort_arr.size > 0 else None
    return dict(
      kind="int",
      shape_pt=list(pt_arr.shape),
      shape_ort=list(ort_arr.shape),
      value_pt=scalar_pt,
      value_ort=scalar_ort,
      exact_match=exact,
      passed=exact,
    )
  diff = np.abs(pt_arr.astype(np.float64) - ort_arr.astype(np.float64))
  err = error_stats(diff)
  return dict(
    kind="float",
    shape_pt=list(pt_arr.shape),
    shape_ort=list(ort_arr.shape),
    error=err,
    passed=float_error_passed(err, max_tol=parity_max_tol, mean_tol=parity_mean_tol),
  )


def compare_outputs(
  pt_dict: Mapping[str, np.ndarray],
  ort_dict: Mapping[str, np.ndarray],
  output_names: Sequence[str],
  *,
  parity_max_tol: float,
  parity_mean_tol: float,
) -> Tuple[Dict[str, Dict[str, Any]], bool]:
  per_output: Dict[str, Dict[str, Any]] = {}
  all_passed = True
  for name in output_names:
    if name not in pt_dict or name not in ort_dict:
      per_output[name] = dict(kind="missing", passed=False)
      all_passed = False
      continue
    item = compare_output_pair(
      name,
      pt_dict[name],
      ort_dict[name],
      parity_max_tol=parity_max_tol,
      parity_mean_tol=parity_mean_tol,
    )
    per_output[name] = item
    if not item.get("passed", False):
      all_passed = False
  return per_output, all_passed


def pt_outputs_to_dict(outputs: Any, output_names: Sequence[str]) -> Dict[str, np.ndarray]:
  if isinstance(outputs, torch.Tensor):
    outputs = (outputs,)
  elif not isinstance(outputs, (tuple, list)):
    outputs = (outputs,)
  result: Dict[str, np.ndarray] = {}
  for name, tensor in zip(output_names, outputs):
    if hasattr(tensor, "detach"):
      result[name] = tensor.detach().cpu().numpy()
    else:
      result[name] = np.asarray(tensor)
  return result


def run_l1_deterministic(
  *,
  patch_runner: OrtRunner,
  slide_runner: OrtRunner,
  patch_feed: Dict[str, np.ndarray],
  slide_feed: Dict[str, np.ndarray],
  slide_output_names: Sequence[str],
  repeats: int,
  tol: float,
) -> Dict[str, Any]:
  patch_runs = [patch_runner.run(patch_feed)[PATCH_OUTPUT] for _ in range(repeats)]
  patch_max_diff = 0.0
  for idx in range(1, len(patch_runs)):
    patch_max_diff = max(patch_max_diff, float(np.max(np.abs(patch_runs[idx] - patch_runs[0]))))

  slide_runs = [slide_runner.run(slide_feed) for _ in range(repeats)]
  slide_output_diffs: Dict[str, float] = {}
  slide_int_exact: Dict[str, bool] = {}
  slide_passed = True
  for name in slide_output_names:
    arrays = [run[name] for run in slide_runs]
    if _is_int_output(name, arrays[0]):
      exact = all(np.array_equal(arrays[0], arr) for arr in arrays[1:])
      slide_int_exact[name] = exact
      slide_output_diffs[name] = 0.0 if exact else 1.0
      slide_passed = slide_passed and exact
    else:
      max_diff = 0.0
      for idx in range(1, len(arrays)):
        max_diff = max(max_diff, float(np.max(np.abs(arrays[idx] - arrays[0]))))
      slide_output_diffs[name] = max_diff
      if max_diff >= tol:
        slide_passed = False

  patch_passed = patch_max_diff < tol
  return dict(
    patch_encoder_onnx_deterministic=dict(
      repeats=repeats,
      backend="onnxruntime",
      max_diff=patch_max_diff,
      passed=patch_passed,
    ),
    slide_encoder_onnx_deterministic=dict(
      repeats=repeats,
      backend="onnxruntime",
      max_diff_by_output=slide_output_diffs,
      int_exact_by_output=slide_int_exact,
      passed=slide_passed,
    ),
    passed=patch_passed and slide_passed,
  )


def summarize_parity_outputs(
  *,
  output_names: Sequence[str],
  per_trial_metrics: List[Dict[str, Any]],
  parity_max_tol: float,
  parity_mean_tol: float,
) -> Tuple[Dict[str, Any], bool]:
  outputs_block: Dict[str, Any] = {}
  all_passed = True
  for name in output_names:
    trials = []
    per_trial_max: List[float] = []
    for trial in per_trial_metrics:
      item = trial["per_output"][name]
      if item.get("kind") == "int":
        trials.append(dict(batch_size=trial["batch_size"], exact_match=item.get("exact_match", False)))
        per_trial_max.append(0.0 if item.get("exact_match", False) else 1.0)
        if not item.get("exact_match", False):
          all_passed = False
      else:
        err = item.get("error", {})
        trials.append(dict(batch_size=trial["batch_size"], error=err))
        per_trial_max.append(float(err.get("max", 0.0)))
        if not float_error_passed(err, max_tol=parity_max_tol, mean_tol=parity_mean_tol):
          all_passed = False
    block = dict(trials=trials, aggregate=aggregate_trial_maxes(per_trial_max), per_trial_max=per_trial_max)
    if trials and trials[0].get("exact_match") is not None:
      block["kind"] = "int"
    outputs_block[name] = block
  return outputs_block, all_passed
