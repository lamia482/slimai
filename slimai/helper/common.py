from pathlib import Path
from .utils.cache import CACHE_ROOT_DIR

##### ENVIRONMENT VARIABLES #####
TORCH_HUB_DIR = Path(CACHE_ROOT_DIR, "torch", "hub")

##### REQUIRMENTS #####
REQUIREMENTS = {
  # package: [min_version, max_version]
  "torch": ["2.7.0", None],
}

##### RESOURCE PATHS #####
import torch
torch.hub.set_dir(TORCH_HUB_DIR)

##### VERSION #####
VERSION = "0.1.7"
