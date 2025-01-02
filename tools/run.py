import os
import argparse
from mmengine.config import Config
from slimai.runner import Runner
from slimai.helper import help_utils


def parse_args():
  parser = argparse.ArgumentParser(description="Use MMEngine to run the pipeline")
  parser.add_argument("--config", type=str, required=True, 
                      help="the yaml config file of the pipeline")
  parser.add_argument("--work_dir", "--work-dir", type=str, default=None, 
                      help="the dir to save the logs and checkpoints")
  parser.add_argument("--action", choices=["train", "infer", "evaluate"], type=str, required=True, 
                      help="the action to run, train, infer, evaluate")
  parser.add_argument("--amp", action="store_true", help="whether to use amp")
  parser.add_argument("--resume", nargs="?", type=str, const="auto",
                      help="If specify checkpoint path, resume from it, while if not "
                      "specify, try to auto resume from the latest checkpoint "
                      "in the work directory.")
  # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
  # will pass the `--local-rank` parameter instead of `--local_rank`.
  parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
  args = parser.parse_args()
  if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = str(args.local_rank)
  return args

def parse_config(args):
  cfg = Config.fromfile(args.config)
  
  # work_dir is determined in this priority: CLI > segment in file
  if args.work_dir is not None:
    # update configs according to CLI args if args.work_dir is not None
    cfg.work_dir = args.work_dir
  elif cfg.get("work_dir", None) is None:
    raise ValueError("work_dir is not specified by CLI or config file")
  
  if args.amp is True:
    cfg.RUNNER.gradient.amp = True

  # resume is determined in this priority: resume from > auto_resume
  if args.resume is not None:
    cfg.RUNNER.resume.enable = True
    cfg.RUNNER.resume.resume_from = "latest" if args.resume == "auto" else args.resume

  return cfg

def main():
  # Parse command line arguments and configuration
  args = parse_args()
  cfg = parse_config(args)

  # Initialize distributed environment
  help_utils.dist_env.init_dist()

  # Create and run the runner
  runner = Runner(cfg)
  runner.run(action=args.action)

  # Close distributed environment
  help_utils.dist_env.close_dist()
  return

if __name__ == "__main__":
  main()
