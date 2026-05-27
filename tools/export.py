import argparse
from pathlib import Path

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
  parser.add_argument(
    "--validate-embedding-path",
    type=str,
    default=None,
    help="Optional .pkl or .h5 for slide_encoder validation with offline embeddings.",
  )
  parser.add_argument("--max-patches", type=int, default=32, help="Max patches for synthetic validation.")
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
    validate_embedding_path=args.validate_embedding_path,
    max_patches=args.max_patches,
  )
