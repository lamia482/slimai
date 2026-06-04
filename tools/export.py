import argparse
from pathlib import Path

import slimai.data  # noqa: F401 — register datasets/sources for V4 loaders

from slimai.runner.exporter import Exporter


def parse_args():
  parser = argparse.ArgumentParser(description="Export MIL / HierarchicalMIL to dual ONNX artifacts.")
  parser.add_argument("--config", type=str, required=True, help="Training config.py path.")
  parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (expects ckpt['weight']).")
  parser.add_argument("--output-dir", type=str, default=None, help="Export output directory.")
  parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
  parser.add_argument("--device", type=str, default="cpu", help="Device used for export trace.")
  parser.add_argument("--cache-dir", type=str, default=None, help="HF cache dir for PatchEncoderBackbone build.")
  parser.add_argument("--skip-validation", action="store_true", help="Skip export validation.")
  parser.add_argument("--max-patches", type=int, default=32, help="Max patches upper bound for validation sampling.")
  parser.add_argument("--validation-num-trials", type=int, default=3, help="Shared V1/V2/V3 trial count.")
  parser.add_argument("--validation-seed", type=int, default=10482, help="Validation random seed.")
  parser.add_argument("--validation-batch-min", type=int, default=8, help="Min dynamic batch N for V1-V3.")
  parser.add_argument("--validation-batch-max", type=int, default=32, help="Max dynamic batch N for V1-V3.")
  parser.add_argument("--validation-parity-max-tol", type=float, default=5e-5, help="V1-V3 float max |PT-ORT| tolerance.")
  parser.add_argument("--validation-parity-mean-tol", type=float, default=5e-6, help="V1-V3 float mean |PT-ORT| tolerance.")
  parser.add_argument("--validation-metrics-tol", type=float, default=5e-4, help="V4 metrics delta tolerance.")
  parser.add_argument("--validation-deterministic-repeats", type=int, default=3, help="L1 ORT repeat count.")
  parser.add_argument("--validation-deterministic-tol", type=float, default=1e-8, help="L1 deterministic tolerance.")
  parser.add_argument("--validation-ort-provider", type=str, default="CPUExecutionProvider", help="ORT provider.")
  parser.add_argument("--skip-test-eval", action="store_true", help="Skip V4 test-set evaluation.")
  parser.add_argument("--reference-work-dir", type=str, default=None, help="Override experiment work_dir for V4 baseline.")
  parser.add_argument("--skip-reference-compare", action="store_true", help="Skip V4 HTML baseline compare.")
  parser.add_argument("--no-validation-show-progress", action="store_true", help="Disable validation tqdm progress bars.")
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  output_dir = args.output_dir
  if output_dir is None:
    output_dir = str(Path(args.ckpt).resolve().parent.parent / "export")

  exporter = Exporter(
    args.config,
    args.ckpt,
    cache_dir=args.cache_dir,
  )
  exporter.export(
    output_dir,
    opset_version=args.opset,
    device=args.device,
    skip_validation=args.skip_validation,
    max_patches=args.max_patches,
    num_trials=args.validation_num_trials,
    batch_min=args.validation_batch_min,
    batch_max=args.validation_batch_max,
    seed=args.validation_seed,
    deterministic_repeats=args.validation_deterministic_repeats,
    deterministic_tol=args.validation_deterministic_tol,
    parity_max_tol=args.validation_parity_max_tol,
    parity_mean_tol=args.validation_parity_mean_tol,
    metrics_tol=args.validation_metrics_tol,
    skip_test_eval=args.skip_test_eval,
    reference_work_dir=args.reference_work_dir,
    skip_reference_compare=args.skip_reference_compare,
    ort_provider=args.validation_ort_provider,
    show_progress=not args.no_validation_show_progress,
  )
