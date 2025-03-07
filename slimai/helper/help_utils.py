import sys
import numpy as np
from pathlib import Path
import mmengine
from loguru import logger
logger.remove()
logger_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>"
logger.add(sys.stderr, format=logger_format)

from .utils.dist_env import dist_env
from .utils.network import PytorchNetworkUtils
from .utils.vis import put_gt_on_image, put_pred_on_image, hstack_imgs, vstack_imgs
from .utils.split_dataset import split_dataset


_CACHE_ROOT_DIR_ = "/.slimai/cache"

def update_logger(log_file: Path, log_level: str = "INFO"):
  if not dist_env.is_main_process():
    return
  logger.add(log_file, level=log_level, format=logger_format)
  return

_warned_messages = set()

def print_log(msg, level="INFO", main_process_only=True, warn_once=False, disable_log=False):
  if (not dist_env.is_main_process() and main_process_only) or disable_log:
    return
  assert (
    level in ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
  ), "Invalid log level: {}".format(level)
  
  if warn_once:
    if msg in _warned_messages:
      return
    _warned_messages.add(msg)
    
  logger.log(level, msg)
  return

def get_folder(file_path):
  return Path(file_path).parent


class ProgressBar(mmengine.ProgressBar):
  def __init__(self, *args, desc=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.desc = f"{desc}: " if desc else ""
    return

  def start(self):
    self.timer = mmengine.Timer()
    return

  def update(self, msg: str = "", num_tasks: int = 1, sep="\t"):
    if not dist_env.is_main_process():
      return
    assert num_tasks > 0
    self.completed += num_tasks
    elapsed = self.timer.since_start()
    if elapsed > 0:
      fps = self.completed / elapsed
    else:
      fps = float('inf')

    no_msg = (msg == "")

    if no_msg:
      msg = self.desc
    else:
      msg = f"{self.desc}{msg}"
      print_log(msg, level="INFO")
      

    if self.task_num > 0:
      percentage = self.completed / float(self.task_num)
      eta = int(elapsed * (1 - percentage) / percentage + 0.5)
      msg += f'{sep}[{{}}] {self.completed}/{self.task_num}, ' \
            f'{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, ' \
            f'ETA: {eta:5}s' + ("" if no_msg else "\n")

      bar_width = min(self.bar_width,
                      int(self.terminal_width - len(msg)) + 2,
                      int(self.terminal_width * 0.6))
      bar_width = max(2, bar_width)
      mark_width = int(bar_width * percentage)
      bar_chars = '>' * mark_width + ' ' * (bar_width - mark_width)
      self.file.write(msg.format(bar_chars))
    else:
      self.file.write(
        f'{msg}{sep}'
        f'completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,'
        f' {fps:.1f} tasks/s\n')
    self.file.flush()

  def close(self):
    self.file.write("\n")
    self.file.flush()
    return

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, 
                     warmup_epochs=0, start_warmup_value=0):
  warmup_schedule = np.array([])
  warmup_iters = warmup_epochs * niter_per_ep
  if warmup_epochs > 0:
      warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

  iters = np.arange(epochs * niter_per_ep - warmup_iters)
  schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

  schedule = np.concatenate((warmup_schedule, schedule))
  assert len(schedule) == epochs * niter_per_ep
  return schedule
