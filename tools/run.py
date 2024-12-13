import os, os.path as osp
import argparse
from mmengine.config import Config
from slimai.runner import Runner
from slimai.helper import help_utils


def parse_args():
  parser = argparse.ArgumentParser(description="Use MMEngine to run the pipeline")
  parser.add_argument("--config", type=str, required=True, 
                      help="the yaml config file of the pipeline")
  parser.add_argument("--action", choices=["train", "infer", "evaluate"], type=str, required=True, 
                      help="the action to run, train, infer, evaluate")
  parser.add_argument("--work_dir", "--work-dir", type=str, default=None, 
                      help="the dir to save the logs and checkpoints")
  parser.add_argument("--amp", action="store_true", help="whether to use amp")
  parser.add_argument("--auto_scale_lr", "--auto-scale-lr", action="store_true", 
                      help="enable automatically scaling LR.")
  parser.add_argument("--resume", nargs="?", type=str, const="auto",
                      help="If specify checkpoint path, resume from it, while if not "
                      "specify, try to auto resume from the latest checkpoint "
                      "in the work directory.")
  parser.add_argument("--launcher", choices=["none", "pytorch", "slurm"], 
                      type=str, default="none", 
                      help="the launcher to use, none, pytorch, slurm")
  # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
  # will pass the `--local-rank` parameter instead of `--local_rank`.
  parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
  args = parser.parse_args()
  if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = str(args.local_rank)
  return args

def parse_config(args):
  cfg = Config.fromfile(args.config)
  cfg.launcher = args.launcher
  
  # work_dir is determined in this priority: CLI > segment in file > filename
  if args.work_dir is not None:
    # update configs according to CLI args if args.work_dir is not None
    cfg.work_dir = osp.join(args.work_dir, cfg.signature)
  elif cfg.get("work_dir", None) is None:
    raise ValueError("work_dir is not specified by CLI or config file")
  
  if args.amp is True:
    optim_wrapper = cfg.optim_wrapper.type
    if optim_wrapper == "AmpOptimWrapper":
      help_utils.print_log("AMP training is already enabled in your config.",
                logger="current", level="WARNING")
    else:
      assert (
        optim_wrapper == "OptimWrapper"
      ), f"`--amp` is only supported when the optimizer wrapper type is `OptimWrapper` but got {optim_wrapper}."
    cfg.optim_wrapper.type = "AmpOptimWrapper"
    cfg.optim_wrapper.loss_scale = "dynamic"

  # enable automatically scaling LR
  if args.auto_scale_lr:
    if "auto_scale_lr" in cfg and \
            "enable" in cfg.auto_scale_lr and \
            "base_batch_size" in cfg.auto_scale_lr:
      cfg.auto_scale_lr.enable = True
    else:
      raise RuntimeError("Can not find \"auto_scale_lr\" or "
                          "\"auto_scale_lr.enable\" or "
                          "\"auto_scale_lr.base_batch_size\" in your"
                          " configuration file.")

  # resume is determined in this priority: resume from > auto_resume
  if args.resume is not None:
    cfg.resume = True
    cfg.load_from = None if args.resume == "auto" else args.resume

  return cfg

def main():

  args = parse_args()
  cfg = parse_config(args)
  runner = Runner(cfg)

  runner.run(action=args.action)

  return

if __name__ == "__main__":
  main()
