"""
* Date     :  2024/12/11 14:00
* File     :  __init__.py
* Brief    :  offer quick start to build & use the toolbox
* Author   :  lamia
* Email    :  wangqiang482@icloud.com
* License  :  (C)Copyright 2024-2027, KFBIO
"""
import os.path as osp
from . import (
  data, helper, models, runner
)

__all__ = [
  "data", "helper", "models", "runner"
]

def get_version():
  from .helper.common import VERSION
  return VERSION

def get_path():
  return osp.abspath(__file__)

def get_package_path():
  return osp.dirname(osp.dirname(get_path()))

def get_last_commit_id():
  from .helper.help_utils import get_last_commit_id_by_path
  return get_last_commit_id_by_path(get_package_path())

__version__ = get_version()
__path__ = get_path()

def check_env():
  import importlib
  from .helper.help_utils import print_log
  from .helper.common import REQUIREMENTS

  print_log(">>> Checking environment for slimai(VERSION={})...".format(__version__))

  for package_name, (min_version, max_version) in REQUIREMENTS.items():
    package = importlib.import_module(package_name)
    package_version = getattr(package, "__version__", None)
    assert (
      str(min_version) <= str(package_version) <= str(max_version)
    ), "{} <= version <= {} is required, but got {}".format(package_name, min_version, max_version, package_version)
    print_log("+++ {} version: {} passed the environment check".format(package_name, package_version))

  print_log(">>> All packages depended by slimai passed the environment check.")
  return
