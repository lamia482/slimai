from abc import abstractmethod
from typing import Dict, Any, List
from ..utils.select import recursive_select
import mmengine


class Visualizer(object):
  @abstractmethod
  def _visualize(self, image, instance, output, class_names):
    raise NotImplementedError("Subclass of Visualizer must implement '_visualize' method")
  
  def render_batch_sample(self, images, outputs, 
                          targets: Dict[str, Any], 
                          class_names: List[str], 
                          progress_bar: bool = False):
    vis_list = []

    indices = list(range(len(images)))
    if progress_bar:
      indices = mmengine.track_iter_progress(indices)
  
    for index in indices:
      image = recursive_select(images, index)
      output = recursive_select(outputs, index)
      target: dict = recursive_select(targets, index) # type: ignore

      vis = self._visualize(image, target, output, class_names)

      vis_list.append(vis)
    return vis_list
