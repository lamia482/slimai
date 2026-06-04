from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import mmengine
import numpy as np
import torch

from slimai.export.onnx_core import run_l0_onnx_checks
from slimai.export.validate_calibration_doc import write_calibration_v3_markdown
from slimai.export.validate_metrics import run_v4_test_evaluation
from slimai.export.validate_ort import (
  OrtRunner,
  compare_outputs,
  error_stats,
  pt_outputs_to_dict,
  run_l1_deterministic,
  summarize_parity_outputs,
)
from slimai.export.validate_progress import (
  TopLevelValidationProgress,
  ValidationPhaseTimer,
  iter_with_postfix,
)
from slimai.export.validate_report import write_validation_report_html

PATCH_INPUT = "patch_tensor"
PATCH_OUTPUT = "embedding_arr"
SLIDE_INPUT = "embedding_arr"


def _sample_batch_size(rng: np.random.Generator, batch_min: int, batch_max: int) -> int:
  return int(rng.integers(batch_min, batch_max + 1))


def _run_parity_trials(
  *,
  num_trials: int,
  seed: int,
  batch_min: int,
  batch_max: int,
  desc: str,
  show_progress: bool,
  trial_fn,
) -> Dict[str, Any]:
  per_trial: List[Dict[str, Any]] = []
  batch_sizes: List[int] = []
  trial_iter = iter_with_postfix(
    range(num_trials),
    enabled=show_progress,
    desc=desc,
    unit="trial",
    total=num_trials,
  )
  for trial_idx in trial_iter:
    rng = np.random.default_rng(seed + trial_idx)
    batch_size = _sample_batch_size(rng, batch_min, batch_max)
    batch_sizes.append(batch_size)
    if hasattr(trial_iter, "set_postfix"):
      trial_iter.set_postfix(N=batch_size)
    per_trial.append(trial_fn(trial_idx=trial_idx, batch_size=batch_size, rng=rng))
  return dict(per_trial=per_trial, batch_sizes=batch_sizes)


def build_calibration_v3_payload(
  *,
  meta: Dict[str, Any],
  preprocess: Dict[str, Any],
  patch_input: np.ndarray,
  pt_embedding: np.ndarray,
  ort_embedding: np.ndarray,
  slide_output_names: Sequence[str],
  pt_outputs: Dict[str, np.ndarray],
  ort_outputs: Dict[str, np.ndarray],
  per_output: Dict[str, Any],
  parity_max_tol: float,
  parity_mean_tol: float,
) -> Dict[str, Any]:
  diff = np.abs(pt_embedding.astype(np.float64) - ort_embedding.astype(np.float64))
  embedding_error = error_stats(diff)
  failed = [name for name, item in per_output.items() if not item.get("passed", False)]
  embedding_passed = (
    embedding_error["max"] < parity_max_tol
    and embedding_error["mean"] < parity_mean_tol
  )
  return dict(
    schema_version="1.0",
    meta=meta,
    preprocess=dict(preprocess),
    inputs=dict(
      patch_tensor=dict(
        name=PATCH_INPUT,
        array=patch_input,
        shape=list(patch_input.shape),
        dtype=str(patch_input.dtype),
      ),
    ),
    intermediate=dict(
      embedding_arr=dict(
        pt=dict(array=pt_embedding, shape=list(pt_embedding.shape), dtype=str(pt_embedding.dtype)),
        ort=dict(array=ort_embedding, shape=list(ort_embedding.shape), dtype=str(ort_embedding.dtype)),
        error=embedding_error,
        passed=embedding_passed,
      ),
    ),
    outputs=dict(
      slide_output_names=list(slide_output_names),
      pt=pt_outputs,
      ort=ort_outputs,
      per_output=per_output,
    ),
    summary=dict(passed=len(failed) == 0, failed_outputs=failed),
  )


def run_export_validation(
  *,
  patch_encoder: torch.nn.Module,
  slide_encoder: torch.nn.Module,
  patch_onnx_path: Path,
  slide_onnx_path: Path,
  slide_output_names: Sequence[str],
  is_hierarchical: bool,
  model_type: str = "MIL",
  output_dir: Path,
  cfg,
  preprocess: Optional[Dict[str, Any]] = None,
  embedding_dim: int = 1024,
  input_size: int = 224,
  config_path: Optional[str] = None,
  ckpt_path: Optional[str] = None,
  num_trials: int = 3,
  batch_min: int = 8,
  batch_max: int = 32,
  seed: int = 10482,
  deterministic_repeats: int = 3,
  deterministic_tol: float = 1e-8,
  parity_max_tol: float = 5e-5,
  parity_mean_tol: float = 5e-6,
  metrics_tol: float = 5e-4,
  skip_test_eval: bool = False,
  reference_work_dir: Optional[Path] = None,
  skip_reference_compare: bool = False,
  ort_provider: str = "CPUExecutionProvider",
  show_progress: bool = True,
  disable_log: bool = False,
  onnx_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
  del disable_log
  output_dir = Path(output_dir)
  patch_onnx_path = Path(patch_onnx_path)
  slide_onnx_path = Path(slide_onnx_path)
  show_progress = bool(show_progress)
  timing: Dict[str, Any] = {}
  report_data: Dict[str, Any] = dict(
    summary=dict(
      policy="report_only",
      num_trials=num_trials,
      parity_max_tol=parity_max_tol,
      parity_mean_tol=parity_mean_tol,
      deterministic_tol=deterministic_tol,
      metrics_tol=metrics_tol,
      ort_provider=ort_provider,
    ),
    timing=timing,
  )

  with TopLevelValidationProgress(enabled=show_progress, total=8) as top_progress:
    with ValidationPhaseTimer(timing, "l0"):
      l0_patch = run_l0_onnx_checks(patch_onnx_path, simplify_in_place=False)
      l0_slide = run_l0_onnx_checks(slide_onnx_path, simplify_in_place=False)
    report_data["l0"] = dict(patch_encoder=l0_patch, slide_encoder=l0_slide, passed=l0_patch["passed"] and l0_slide["passed"])
    top_progress.update(1)

    patch_runner = OrtRunner(patch_onnx_path, provider=ort_provider)
    slide_runner = OrtRunner(slide_onnx_path, provider=ort_provider)
    l1_patch = torch.randn(16, 3, input_size, input_size)
    l1_embedding = torch.randn(16, embedding_dim)
    l1_patch_feed = {PATCH_INPUT: l1_patch.numpy().astype(np.float32)}
    l1_slide_feed = {SLIDE_INPUT: l1_embedding.numpy().astype(np.float32)}
    with ValidationPhaseTimer(timing, "l1"):
      l1 = run_l1_deterministic(
        patch_runner=patch_runner,
        slide_runner=slide_runner,
        patch_feed=l1_patch_feed,
        slide_feed=l1_slide_feed,
        slide_output_names=slide_output_names,
        repeats=deterministic_repeats,
        tol=deterministic_tol,
      )
    report_data["l1"] = l1
    top_progress.update(1)

    device = "cpu"

    def run_v1_trial(*, trial_idx: int, batch_size: int, rng: np.random.Generator):
      patch = torch.randn(batch_size, 3, input_size, input_size, device=device)
      with torch.inference_mode():
        pt_emb = patch_encoder(patch).detach().cpu().numpy()
      ort_emb = patch_runner.run({PATCH_INPUT: patch.numpy().astype(np.float32)})[PATCH_OUTPUT]
      per_output, passed = compare_outputs(
        {PATCH_OUTPUT: pt_emb},
        {PATCH_OUTPUT: ort_emb},
        [PATCH_OUTPUT],
        parity_max_tol=parity_max_tol,
        parity_mean_tol=parity_mean_tol,
      )
      return dict(batch_size=batch_size, per_output=per_output, passed=passed)

    with ValidationPhaseTimer(timing, "v1"):
      v1_trials = _run_parity_trials(
        num_trials=num_trials,
        seed=seed,
        batch_min=batch_min,
        batch_max=batch_max,
        desc="V1 patch parity",
        show_progress=show_progress,
        trial_fn=run_v1_trial,
      )
    v1_outputs, v1_passed = summarize_parity_outputs(
      output_names=[PATCH_OUTPUT],
      per_trial_metrics=v1_trials["per_trial"],
      parity_max_tol=parity_max_tol,
      parity_mean_tol=parity_mean_tol,
    )
    report_data["v1"] = dict(
      name="patch_encoder_ort_parity",
      num_trials=num_trials,
      batch_size_range=[batch_min, batch_max],
      per_trial_batch_size=v1_trials["batch_sizes"],
      outputs=v1_outputs,
      passed=v1_passed,
    )
    top_progress.update(1)

    def run_v2_trial(*, trial_idx: int, batch_size: int, rng: np.random.Generator):
      embedding = torch.randn(batch_size, embedding_dim, device=device)
      with torch.inference_mode():
        pt_out = pt_outputs_to_dict(slide_encoder(embedding), slide_output_names)
      ort_out = slide_runner.run({SLIDE_INPUT: embedding.numpy().astype(np.float32)})
      per_output, passed = compare_outputs(
        pt_out,
        ort_out,
        slide_output_names,
        parity_max_tol=parity_max_tol,
        parity_mean_tol=parity_mean_tol,
      )
      return dict(batch_size=batch_size, per_output=per_output, passed=passed)

    with ValidationPhaseTimer(timing, "v2"):
      v2_trials = _run_parity_trials(
        num_trials=num_trials,
        seed=seed,
        batch_min=batch_min,
        batch_max=batch_max,
        desc="V2 slide parity",
        show_progress=show_progress,
        trial_fn=run_v2_trial,
      )
    v2_outputs, v2_passed = summarize_parity_outputs(
      output_names=slide_output_names,
      per_trial_metrics=v2_trials["per_trial"],
      parity_max_tol=parity_max_tol,
      parity_mean_tol=parity_mean_tol,
    )
    report_data["v2"] = dict(
      name="slide_encoder_ort_parity",
      num_trials=num_trials,
      batch_size_range=[batch_min, batch_max],
      per_trial_batch_size=v2_trials["batch_sizes"],
      outputs=v2_outputs,
      passed=v2_passed,
    )
    top_progress.update(1)

    calibration_payload = None

    def run_v3_trial(*, trial_idx: int, batch_size: int, rng: np.random.Generator):
      nonlocal calibration_payload
      patch = torch.randn(batch_size, 3, input_size, input_size, device=device)
      with torch.inference_mode():
        pt_emb = patch_encoder(patch).detach().cpu().numpy()
        pt_slide = pt_outputs_to_dict(slide_encoder(torch.from_numpy(pt_emb)), slide_output_names)
      ort_emb = patch_runner.run({PATCH_INPUT: patch.numpy().astype(np.float32)})[PATCH_OUTPUT]
      ort_slide = slide_runner.run({SLIDE_INPUT: ort_emb.astype(np.float32)})
      per_output, passed = compare_outputs(
        pt_slide,
        ort_slide,
        slide_output_names,
        parity_max_tol=parity_max_tol,
        parity_mean_tol=parity_mean_tol,
      )
      if trial_idx == 0:
        meta = dict(
          check="v3_e2e_parity",
          trial_idx=0,
          num_trials=num_trials,
          seed=seed + trial_idx,
          batch_size=batch_size,
          batch_size_range=[batch_min, batch_max],
          embedding_dim=embedding_dim,
          input_size=input_size,
          is_hierarchical=is_hierarchical,
          model_type=model_type,
          ort_provider=ort_provider,
          parity_max_tol=parity_max_tol,
          parity_mean_tol=parity_mean_tol,
          created_at=datetime.now(timezone.utc).isoformat(),
          export_dir=str(output_dir.resolve()),
          patch_onnx="patch_encoder.onnx",
          slide_onnx="slide_encoder.onnx",
          config_path=config_path or "",
          ckpt_path=ckpt_path or "",
        )
        calibration_payload = build_calibration_v3_payload(
          meta=meta,
          preprocess=preprocess or {},
          patch_input=patch.numpy().astype(np.float32),
          pt_embedding=pt_emb,
          ort_embedding=ort_emb,
          slide_output_names=slide_output_names,
          pt_outputs=pt_slide,
          ort_outputs=ort_slide,
          per_output=per_output,
          parity_max_tol=parity_max_tol,
          parity_mean_tol=parity_mean_tol,
        )
      return dict(batch_size=batch_size, per_output=per_output, passed=passed)

    with ValidationPhaseTimer(timing, "v3"):
      v3_trials = _run_parity_trials(
        num_trials=num_trials,
        seed=seed,
        batch_min=batch_min,
        batch_max=batch_max,
        desc="V3 e2e parity",
        show_progress=show_progress,
        trial_fn=run_v3_trial,
      )
      if calibration_payload is not None:
        mmengine.dump(calibration_payload, output_dir / "calibration_v3_trial0.pkl")
        write_calibration_v3_markdown(output_dir, calibration_payload)
    v3_outputs, v3_passed = summarize_parity_outputs(
      output_names=slide_output_names,
      per_trial_metrics=v3_trials["per_trial"],
      parity_max_tol=parity_max_tol,
      parity_mean_tol=parity_mean_tol,
    )
    report_data["v3"] = dict(
      name="e2e_ort_parity",
      num_trials=num_trials,
      batch_size_range=[batch_min, batch_max],
      per_trial_batch_size=v3_trials["batch_sizes"],
      outputs=v3_outputs,
      calibration_files=["calibration_v3_trial0.pkl", "calibration_v3_trial0.md"],
      passed=v3_passed,
    )
    top_progress.update(1)

    if skip_test_eval:
      report_data["v4"] = dict(skipped=True, passed=True)
    else:
      if reference_work_dir is None and ckpt_path is not None:
        reference_work_dir = Path(ckpt_path).resolve().parent.parent
      with ValidationPhaseTimer(timing, "v4"):
        report_data["v4"] = run_v4_test_evaluation(
          cfg=cfg,
          slide_encoder=slide_encoder,
          slide_onnx_path=slide_onnx_path,
          slide_output_names=slide_output_names,
          is_hierarchical=is_hierarchical,
          reference_work_dir=reference_work_dir,
          skip_reference_compare=skip_reference_compare,
          metrics_tol=metrics_tol,
          ort_provider=ort_provider,
          show_progress=show_progress,
          timing=timing,
        )
    top_progress.update(1)

  checks = [report_data.get(k, {}).get("passed", True) for k in ["l0", "l1", "v1", "v2", "v3", "v4"]]
  report_data["summary"]["passed"] = all(bool(x) for x in checks if x is not None)
  if onnx_meta:
    report_data["summary"]["export"] = onnx_meta
  with ValidationPhaseTimer(timing, "report_html"):
    write_validation_report_html(output_dir, report_data)
  return report_data
