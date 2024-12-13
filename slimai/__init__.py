"""
* Date     :  2024/12/11 14:00
* File     :  __init__.py
* Brief    :  offer quick start to build & use the toolbox
* Author   :  lamia
* Email    :  wangqiang482@icloud.com
* License  :  (C)Copyright 2024-2027, KFBIO
"""
from . import (
  data, helper, losses, models, runner
)

__all__ = [
  "data", "helper", "losses", "models", "runner"
]

__version__ = "0.0.1"

def get_version():
  return __version__

def check_env():
  from .helper.help_utils import print_log

  import torch
  assert (
    torch.__version__ >= "2.0.0"
  ), "PyTorch version >= 2.0.0 is required, but got {}".format(torch.__version__)
  print_log("Pytorch Passed, version: {}".format(torch.__version__))
  print_log("Check environment success.")
  return

check_env()
