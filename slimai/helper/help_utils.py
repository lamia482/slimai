import os
import logging
from pathlib import Path
import mmengine
from mmengine.logging import print_log as _mm_print_log
from .utils.dist_env import dist_env
from .utils.network import PytorchNetworkUtils
from .utils.vis import put_gt_on_image, put_pred_on_image, hstack_imgs, vstack_imgs


def print_log(msg, logger="current", level="INFO", main_process_only=True):
  if not dist_env.is_main_process() and main_process_only:
    return
  assert (
    level in ["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
  ), "Invalid log level: {}".format(level)
  level = getattr(logging, level)
  _mm_print_log(msg, logger=logger, level=level)
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

  def update(self, msg: str, num_tasks: int = 1, sep="\t"):
    if not dist_env.is_main_process():
      return
    assert num_tasks > 0
    self.completed += num_tasks
    elapsed = self.timer.since_start()
    if elapsed > 0:
      fps = self.completed / elapsed
    else:
      fps = float('inf')

    msg = f"{self.desc}{msg}"
    print_log(msg, level="INFO")

    if self.task_num > 0:
      percentage = self.completed / float(self.task_num)
      eta = int(elapsed * (1 - percentage) / percentage + 0.5)
      msg += f'{sep}[{{}}] {self.completed}/{self.task_num}, ' \
            f'{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, ' \
            f'ETA: {eta:5}s\n'

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
