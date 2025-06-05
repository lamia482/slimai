import threading
import swanlab
import mmengine
from typing import Dict, Any, Optional
from .utils.dist_env import dist_env
from .structure import DataSample
from .utils.visualize import Visualizer


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

    description = config.get("_COMMENT_", None)
    assert (
      description is not None
    ), "description/comment is not specified in config, please specify '_COMMENT_' in the config file"

    swanlab.init(
      project=project_name, 
      experiment_name=experiment_name,
      description=description,
      config=config,
    )

    self._initialized = True
    return
  
  def log_data(self, data: Dict[str, Any]):
    """
    Log metrics and other data to swanlab.
    
    Args:
      data: Dictionary of metrics to log
    """
    if not self.should_record:
      return
    return swanlab.log(data)
  
  def log_sample(self, batch_info: DataSample):
    """
    Log visualization of batch samples.
    
    Args:
      batch_info: DataSample containing batch information to visualize
    """
    if not self.should_record:
      return
    vis_list = Visualizer.render_batch_sample(batch_info)
    vis_dict = {
      f"vis_batch_{i}": swanlab.Image(vis)
      for i, vis in enumerate(vis_list)
    }
    return swanlab.log(vis_dict)
  
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
