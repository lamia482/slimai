"""
slimai package entry.

Keep package import lightweight to avoid pulling optional runtime dependencies
when users only need a small submodule (e.g. helper.features.compose).
"""

import importlib
import os
import os.path as osp
from typing import Any

_EXTRA_ENV_ = {
  "OPENCV_IO_MAX_IMAGE_PIXELS": "1099511627776",
}

for key, value in _EXTRA_ENV_.items():
  if key not in os.environ:
    os.environ[key] = value

__all__ = ["data", "helper", "models", "runner", "check_env"]

def __getattr__(name: str) -> Any:
  if name in ["data", "helper", "models", "runner"]:
    module = importlib.import_module(f".{name}", __name__)
    globals()[name] = module
    return module
  raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def get_path() -> str:
  return osp.abspath(__file__)

def get_package_path() -> str:
  return osp.dirname(osp.dirname(get_path()))

def get_last_commit_id() -> str:
  from .helper.help_utils import get_last_commit_id_by_path
  return get_last_commit_id_by_path(get_package_path())

def get_version():
  from .helper.common import VERSION
  commit = get_last_commit_id()[:8]
  version = f"{VERSION}-{commit}"
  return version

__version__ = get_version()

def check_env() -> None:
  from .helper.help_utils import print_log
  from .helper.common import REQUIREMENTS

  print_log(">>> Checking environment for slimai...")
  for package_name, (min_version, max_version) in REQUIREMENTS.items():
    package = importlib.import_module(package_name)
    package_version = getattr(package, "__version__", None)
    assert (
      (min_version is None or str(package_version) >= str(min_version)) and
      (max_version is None or str(package_version) <= str(max_version))
    ), "{} <= {}.ver <= {} is required, but got {}".format(
      min_version, package_name, max_version, package_version
    )
    print_log("+++ {} version: {} passed the environment check".format(package_name, package_version))
  print_log(">>> All packages depended by slimai passed the environment check.")
  return