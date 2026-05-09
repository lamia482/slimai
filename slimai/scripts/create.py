import argparse
import os
from pathlib import Path
import sys

_READER_ENV_DEFAULTS = {
  "ENABLE_SLIDE_MPP_UNIFY": "0",
  "ENABLE_PYRAMID_RESAMPLE": "1",
  "ENABLE_SHRINK_ON_BOUNDARY": "1",
  "ENV_SLIDE_UNIFY_MPP_MAGNIFICATION": "20",
  "ENV_SLIDE_UNIFY_MPP_RESOLUTION": "0.5,0.5",
}


def _apply_reader_env_defaults():
  for key, value in _READER_ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)
  package_dir = Path(__file__).resolve().parents[1]
  project_dir = package_dir.parent
  project_path = project_dir.as_posix()
  if project_path not in sys.path:
    sys.path.insert(0, project_path)
  return


def parse_args():
  parser = argparse.ArgumentParser(description="Create WSI embedding h5 files.")
  parser.add_argument("--input-file", required=True, help="WSI file or Path to input xlsx/xls.")
  parser.add_argument("--wsi-col", dest="wsi_col", default="wsi_file", help="Column name for wsi file path.")
  parser.add_argument("--patch-encoder", required=False, default=None, help="Single patch encoder name.")
  parser.add_argument(
    "--patch-encoders",
    required=False,
    default=None,
    help="Comma-separated patch encoders, e.g. UNI,UNI2,CONCH",
  )
  parser.add_argument(
    "--slide-encoder", required=False, help="Slide encoder name.",
    choices=[
      "TITAN", 
    ] 
  )
  parser.add_argument("--tag", required=True, help="User-specified embedding tag.")
  parser.add_argument("--out-dir", required=True, help="Output directory.")
  parser.add_argument("--devices", default=None, help="Comma-separated device ids, e.g. 0,1,2,3")
  parser.add_argument(
    "--accelerator",
    default="auto",
    choices=["auto", "npu", "cuda", "cpu"],
    help="Accelerator backend. auto checks npu/cuda/cpu in order.",
  )
  parser.add_argument("--max-futs", type=int, default=0, help="Max pending futures. 0 means auto.")
  parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size. -1 means auto.")
  parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers per device. -1 means auto.")
  parser.add_argument("--patch-size", type=int, default=224, help="Patch size.")
  parser.add_argument("--stride-size", type=int, default=192, help="Stride when reset-coords.")
  parser.add_argument("--read-scale", type=float, default=20, help="Read scale for WSI.")
  parser.add_argument("--operate-scale", type=float, default=1.25, help="Operate scale for tissue shrink.")
  parser.add_argument("--to-gray", action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument(
    "--incremental-h5",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Append missing encoder datasets into existing h5 files instead of replacing whole files.",
  )
  parser.add_argument(
    "--verify-existing-md5",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Verify existing h5 cache by hashing source WSI. Disabled by default for faster resume.",
  )
  parser.add_argument("--min-tissue-ratio", type=float, default=0.05, help="Minimum tissue ratio for tissue shrink.")
  parser.add_argument("--tissue-shrink", type=str, default="tissue", help="Tissue shrink method.")
  parser.add_argument("--output", default=None, help="Path to xlsx record file, e.g. record.xlsx.")
  parser.add_argument("--preflight-only", action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument("--manifest-only", action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument("--limit", type=int, default=0, help="Only process first N manifest items. 0 means no limit.")
  return parser.parse_args()


def _import_compose_main():
  from slimai.helper.features.compose import CreateFeatureConfig, main as compose_main
  return CreateFeatureConfig, compose_main


def main():
  args = parse_args()
  _apply_reader_env_defaults()
  CreateFeatureConfig, compose_main = _import_compose_main()
  config = CreateFeatureConfig(
    input_file=args.input_file,
    wsi_col=args.wsi_col,
    patch_encoder=args.patch_encoder,
    patch_encoders=args.patch_encoders,
    slide_encoder=args.slide_encoder,
    tag=args.tag,
    out_dir=args.out_dir,
    devices=args.devices,
    accelerator=args.accelerator,
    max_futs=args.max_futs,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    patch_size=args.patch_size,
    stride_size=args.stride_size,
    read_scale=args.read_scale,
    operate_scale=args.operate_scale,
    to_gray=args.to_gray,
    skip_existing=args.skip_existing,
    incremental_h5=args.incremental_h5,
    verify_existing_md5=args.verify_existing_md5,
    min_tissue_ratio=args.min_tissue_ratio,
    tissue_shrink=args.tissue_shrink,
    output=args.output,
    preflight_only=args.preflight_only,
    manifest_only=args.manifest_only,
    limit=args.limit,
  )
  compose_main(config)
  return

if __name__ == "__main__":
  if len(sys.argv) == 1:
    # sys.argv.extend([
    #   "--input-file", "/.slimai/cache/wsi-group-in-multiple-formats/goods/T2020-14513-20X-KFB.kfb", 
    #   "--patch-encoder", "UNI", 
    #   "--slide-encoder", "TITAN", 
    #   "--tag", "debug", 
    #   "--out-dir", "/hzztai/slimai/_debug_/rst", 
    #   "--devices", "1",
    #   "--operate-scale", "1.25",
    # ])
    sys.argv.extend([
      "--input-file", "/hzztai/slimai/_debug_/thca/embeddings/merged_dataset_by_center.xlsx", 
      "--wsi-col", "厦大中山:wsi_file", 
      "--patch-encoder", "UNI", 
      "--tag", "UNI_GRAY",
      "--to-gray", 
      "--out-dir", "/hzztai/slimai/_debug_/thca/embeddings-K/xiamen", 
      "--output", "/hzztai/slimai/_debug_/thca/embeddings-K/xiamen/record.xlsx", 
      "--devices", "0,1", 
      "--operate-scale", "1.25",
    ])

  main()
