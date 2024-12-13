import os
import logging
from pathlib import Path
from mmengine.logging import print_log as _mm_print_log
from mmengine.dist import is_main_process

def print_log(msg, logger="current", level="INFO", force=False):
  if not is_main_process() and not force:
    return
  assert (
    level in ["INFO", "WARN", "WARNING", "ERROR", "CRITICAL", "FATAL"]
  ), "Invalid log level: {}".format(level)
  level = getattr(logging, level)
  _mm_print_log(msg, logger=logger, level=level)
  return

def collect_env():
  return {k: os.environ[k] for k in [
    "LOCAL_RANK",
    "RANK", 
    "GROUP_RANK",
    "ROLE_RANK",
    "LOCAL_WORLD_SIZE",
    "WORLD_SIZE",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID",
  ]}

def get_folder(file_path):
  return Path(file_path).parent
