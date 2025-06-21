import threading
import swanlab
import mmengine
import torch
from PIL import Image
from typing import Dict, Any, Optional, Union, List
from .utils.dist_env import dist_env
from .utils.visualize import Visualizer
from .utils.select import recursive_select


class Record(object):
  """
  A singleton class for recording experiment metrics and visualizations.
  Uses swanlab for experiment tracking and ensures only one process records data.
  """
  _instance = None
  _lock = threading.Lock()

  def __new__(cls, *args, **kwargs):
    """
    Implements the singleton pattern to ensure only one Record instance exists.
    """
    if cls._instance is None:
      with cls._lock:
        if cls._instance is None:
          cls._instance = super().__new__(cls)
          cls._instance._initialized = False
    return cls._instance

  @classmethod
  def create(cls, *args, **kwargs):
    """
    Factory method to create or return the singleton instance.
    """
    return cls(*args, **kwargs)
  
  @property
  def should_record(self):
    """
    Determines if the current process should record data.
    Only the main process across all nodes should record.
    """
    return dist_env.is_main_process(local=False)
  
  def __init__(self, *, cfg: Optional[mmengine.Config] = None):
    """
    Initialize the Record instance with configuration.
    
    Args:
      cfg: Configuration object containing experiment settings
    """
    if self._initialized:
      return

    # only one process across all nodes&ranks should record
    if not self.should_record:
      return
    
    assert (
      cfg is not None
    ), "cfg is not specified in Record, this may caused by the wrong initialization order"
    
    config = cfg.copy().to_dict()

    work_dir = config.get("_WORK_DIR_", None)
    assert (
      work_dir is not None
    ), "work_dir is not specified in config, please specify '_WORK_DIR_' in the config file or set 'work_dir' in CLI"

    project_name = config.get("_PROJECT_", None)
    assert (
      project_name is not None
    ), "project name is not specified in config, please specify '_PROJECT_' in the config file"

    experiment_name = config.get("_EXPERIMENT_", None)
    assert (
      experiment_name is not None
    ), "experiment name is not specified in config, please specify '_EXPERIMENT_' in the config file or set 'config' in CLI"

    swanlab.init(
      project=project_name, 
      experiment_name=experiment_name,
      config=config,
    )

    self._initialized = True
    return
  
  def log_step_data(self, data: Dict[str, Any], phase: Optional[str] = None):
    """
    Log metrics and other data to swanlab.
    
    Args:
      data: Dictionary of metrics to log
    """
    if not self.should_record:
      return
    if phase is not None:
      data = {
        f"{phase}/{key}": value
        for key, value in data.items()
      }
    return swanlab.log(data)
  
  def log_batch_sample(self, 
                 batch_image: Union[List, torch.Tensor], 
                 batch_output: Union[List, torch.Tensor], 
                 batch_targets: Dict[str, List[Any]],
                 class_names: List[str],
                 phase: Optional[str] = None,
                 topk: int = 1, 
                 progress_bar: bool = False):
    """
    Log visualization of batch samples.
    
    Args:
      batch_image: List of images to visualize
      batch_output: List of output to visualize
      batch_targets: Dict of List of targets to visualize
    """
    if not self.should_record:
      return
    
    if topk <= 0:
      topk = len(batch_image)

    topk_ids = torch.randperm(len(batch_image))[:topk].tolist()
    
    files = [v.file for v in batch_image] # type: ignore
    topk_files = recursive_select(files, topk_ids)
    topk_outputs = recursive_select(batch_output, topk_ids)
    topk_targets: dict = recursive_select(batch_targets, topk_ids) # type: ignore

    vis_list = Visualizer.render_batch_sample(
      topk_files, topk_outputs, topk_targets, 
      class_names, 
      progress_bar=progress_bar,
    )
    
    vis_list = [
      swanlab.Image(Image.fromarray(vis[..., ::-1]), caption=f"{topk_files[i]}")
      for i, vis in enumerate(vis_list)
    ] # not support in private deploy env yet.
    return self.log_step_data({"visualize": vis_list}, phase=phase)
  
  def finish(self):
    """
    Finish the recording session and clean up resources.
    """
    if not self.should_record:
      return
    return swanlab.finish()

  def __del__(self):
    """
    Destructor to ensure swanlab session is properly closed.
    """
    if not self.should_record:
      return
    return swanlab.finish()
