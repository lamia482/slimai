from abc import abstractmethod
from typing import Dict, Any, List
import concurrent.futures
from functools import partial
from ..utils.select import recursive_select
from ..help_utils import ProgressBar, get_dist_env


class Visualizer(object):
  @abstractmethod
  def _visualize(self, image, instance, output, class_names):
    raise NotImplementedError("Subclass of Visualizer must implement '_visualize' method")
  
  def render_batch_sample(self, images, outputs, 
                          targets: Dict[str, Any], 
                          class_names: List[str], 
                          progress_bar: bool = False,
                          num_workers: int = 1):
    if progress_bar:
      pbar = ProgressBar(len(images), desc="Visualizing")
    
    def process_single_sample(index, images, outputs, targets, class_names):
      image = recursive_select(images, index)
      output = recursive_select(outputs, index)
      target: dict = recursive_select(targets, index) # type: ignore
      if progress_bar:
        pbar.update(sep="\r\t")
      return self._visualize(image, target, output, class_names)
    
    indices = list(range(len(images)))
    
    # Create a partial function with fixed arguments
    process_func = partial(process_single_sample, 
                          images=images, 
                          outputs=outputs, 
                          targets=targets, 
                          class_names=class_names)
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(
      max_workers=num_workers) as executor:
      vis_list = list(executor.map(process_func, indices))
      
    return vis_list
