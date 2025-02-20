import argparse
from slimai.runner import Exporter


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--ckpt_path", type=str, required=True)
  parser.add_argument("--output_dir", type=str, required=True)
  parser.add_argument("--format", type=str, default="onnx")
  args = parser.parse_args()
  exporter = Exporter(args.ckpt_path)
  exporter.export(args.output_dir, format=args.format)
