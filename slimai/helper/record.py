import swanlab
import mmengine
import torch
from pathlib import Path
from PIL import Image
from typing import Dict, Any, Optional, Union, List, Callable
from .utils.select import recursive_select, recursive_apply
from .help_utils import print_log, get_dist_env
from .help_build import build_visualizer
from .structure import DataSample
from .utils import singleton


@singleton.singleton_wrapper
class Record(object):
  """
  A singleton class for recording experiment metrics and visualizations.
  Uses swanlab for experiment tracking and ensures only one process records data.
  """

  def __init__(self, *, cfg: Optional[mmengine.Config] = None):
    """ no attribute will be changed after initialization, so it is safe to use singleton wrapper without classproperty
    Initialize the Record instance with configuration.
    
    Args:
      cfg: Configuration object containing experiment settings
    """
    self._swanlab_ok = True

    assert (
      cfg is not None
    ), "cfg is not specified in Record, this may caused by the wrong initialization order"
    
    config = cfg.copy().to_dict()

    self.work_dir = config.get("work_dir", None)
    assert (
      self.work_dir is not None
    ), "work_dir is not specified in config, please specify '_WORK_DIR_' in the config file or set 'work_dir' in CLI"

    self.project_name = config.get("_PROJECT_", None)
    assert (
      self.project_name is not None
    ), "project name is not specified in config, please specify '_PROJECT_' in the config file"

    self.workspace = config.get("_WORKSPACE_", None)

    self.experiment_name = config.get("_EXPERIMENT_", None)
    assert (
      self.experiment_name is not None
    ), "experiment name is not specified in config, please specify '_EXPERIMENT_' in the config file or set 'config' in CLI"

    visualizer_cfg = config["RUNNER"].get("visualizer", None)
    if visualizer_cfg is None:
      visualizer_cfg = dict()
    
    self.save_dir = visualizer_cfg.pop("save_dir", None) # save visuals to this dir under work_dir, setting None to disable
    self.every_n_steps_on_train = visualizer_cfg.pop("every_n_steps_on_train", 10) # save train visuals every n steps, set to None or ≤0 to disable
    self.topk_vis_on_train = visualizer_cfg.pop("topk_vis_on_train", None) # topk samples will be rendered on train, setting to None to render all samples, setting to ≤0 to disable
    self.topk_vis_on_eval = visualizer_cfg.pop("topk_vis_on_eval", 32) # topk samples will be rendered on eval, setting to None to render all samples, setting to ≤0 to disable
    
    if len(visualizer_cfg) == 0:
      visualizer_cfg = None
    self.visualizer = build_visualizer(visualizer_cfg)

    logger = config["RUNNER"]["logger"]
    self.log_precision = logger.get("log_precision", ".8f")
    self.log_loss_precision = logger.get("log_loss_precision", self.log_precision)
    self.log_latency_precision = logger.get("log_latency_precision", self.log_precision)

    # only one process across all nodes&ranks should record
    if not self.should_record:
      return
    
    try:
      swanlab.init(
        project=self.project_name, 
        workspace=self.workspace, # type: ignore
        experiment_name=self.experiment_name,
        config=config,
      )
    except Exception as ex:
      print_log(f"Error in initializing swanlab, skip recording\n{ex}", level="WARNING")
      self._swanlab_ok = False
      return

    print_log(self)
    return
  
  def __repr__(self):
    return (f"Record(\n"
            f"  swanlab_ok={self._swanlab_ok},\n"
            f"  should_record={self.should_record},\n"
            f"  project_name={self.project_name},\n"
            f"  workspace={self.workspace},\n"
            f"  experiment_name={self.experiment_name},\n"
            f"  work_dir={self.work_dir},\n"
            f"  save_dir={self.save_dir},\n"
            f"  every_n_steps_on_train={self.every_n_steps_on_train},\n"
            f"  topk_vis_on_train={self.topk_vis_on_train},\n"
            f"  topk_vis_on_eval={self.topk_vis_on_eval},\n"
            f")")
  __str__ = __repr__

  @property
  def should_record(self):
    """
    Determines if the current process should record data.
    Only the main process across all nodes should record.
    """
    return (
      get_dist_env().is_main_process(local=False) and 
      self._swanlab_ok
    )

  def format(self, log_data: Dict[str, Any]):
    def fix_type(value: Any):
      if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
          value = value.item()
        else:
          value = value.tolist()
      elif isinstance(value, Image.Image):
        value = swanlab.Image(value)
      elif isinstance(value, (str, Path)):
        value = swanlab.Text(str(value))
      return value
    
    log_data = recursive_apply(fix_type, log_data) # type: ignore

    msg = ""
    for key, value in log_data.items():
      if "loss" in key:
        msg += f", {key}: {value:{self.log_loss_precision}}"
      elif "latency" in key:
        msg += f", {key}: {value:{self.log_latency_precision}}"
      elif isinstance(value, float):
        msg += f", {key}: {value:{self.log_precision}}"
      elif not isinstance(value, list):
        msg += f", {key}: {value}"

    # cast list to adapt swanlab
    for key in list(log_data.keys()):
      value = log_data[key]
      if not isinstance(value, list):
        continue
      if all(map(lambda x: isinstance(x, (swanlab.Image, swanlab.Text)), value)):
        continue
      log_data.pop(key)
      for i, v in enumerate(value):
        if isinstance(v, (str, Path)):
          v = swanlab.Text(str(v))
        log_data[f"{key}_{i}"] = v

    return log_data, msg
  
  def log_step_data(self, data: Dict[str, Any], phase: Optional[str] = None, step: Optional[int] = None):
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

    data, _ = self.format(data) # type: ignore

    return swanlab.log(data, step=step) # type: ignore
  
  def check_visualize_batch(self, output, step):
    if not self.should_record:
      return False

    if self.visualizer is None:
      return False
    
    assert (
      step >= 0
    ), "step must be non-negative"

    status = (output is not None)

    if self.topk_vis_on_train is not None and self.topk_vis_on_train <= 0:
      status &= False

    if self.every_n_steps_on_train is None or self.every_n_steps_on_train <= 0 or step % self.every_n_steps_on_train != 0:
      status &= False

    return status

  def topk(self, phase: str):
    topk = self.topk_vis_on_train if (phase == "train") else self.topk_vis_on_eval
    return topk

  def log_batch_sample(self, 
                       batch_image: Union[List, torch.Tensor], 
                       batch_output: Union[List, torch.Tensor], 
                       batch_targets: Dict[str, List[Any]],
                       class_names: List[str],
                       phase: str = "runtime",
                       progress_bar: bool = False, 
                       step: Optional[int] = None):
    """
    Log visualization of batch samples.
    
    Args:
      batch_image: List of images to visualize
      batch_output: List of output to visualize
      batch_targets: Dict of List of targets to visualize
      class_names: List of class names
      phase: Phase of the batch
      progress_bar: Whether to show progress bar
    """
    if not self.should_record:
      return

    topk = self.topk(phase)
    if topk is None:
      topk = len(batch_image)

    if topk <= 0:
      topk = 0

    if 0 < topk < len(batch_image):
      topk_ids = torch.randperm(len(batch_image))[:topk].tolist()
    else:
      topk_ids = list(range(len(batch_image)))

    if  isinstance(batch_image, list) and (len(batch_image) > 0) and \
        all(list(map(lambda x: isinstance(x, DataSample), batch_image))):
      vis_images = [batch_image[i].meta["visual_file"] for i in topk_ids]
    else:
      vis_images = [batch_image[i] for i in topk_ids]

    vis_outputs = recursive_select(batch_output, topk_ids)
    vis_targets: dict = recursive_select(batch_targets, topk_ids) # type: ignore

    print_log(f"Visualizing top-{len(topk_ids)}(random on truncated) samples on {phase}...")

    try:
      vis_list = self.visualizer.render_batch_sample(
        vis_images, vis_outputs, vis_targets, 
        class_names,
        progress_bar=progress_bar,
      )
    except Exception as e:
      print_log(f"Error in rendering batch sample\n{e}", level="WARNING")
      return

    if self.save_dir is not None:
      print_log("Visualization Saving is not supported yet.", level="WARNING", warn_once=True)

    vis_list = [
      swanlab.Image(Image.fromarray(vis[..., ::-1]), 
                    caption=(name if isinstance(name, str) else str(i)))
      for i, (vis, name) in enumerate(zip(vis_list, vis_images))
    ] # not support in private deploy env yet.
    return self.log_step_data({"visualize": vis_list}, phase=phase, step=step)
  
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
