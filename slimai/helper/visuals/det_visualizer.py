from slimai.helper.utils.scale_image import to_image
from .visulizer import Visualizer
from ..utils.visualize import put_gt_on_image, put_pred_on_image
from ..help_build import VISUALIZERS


@VISUALIZERS.register_module("DetectionVisualizer")
class DetectionVisualizer(Visualizer):
  def _visualize(self, image, gt_instance, pred_instance, 
                 names, score_thr=0.01, color=(0, 0, 255), 
                 no_text=False):
    gt_instance = gt_instance["instance"]
    image = to_image(image)
    image = put_gt_on_image(image, gt_instance, names, color=(255, 0, 0), no_text=no_text)
    image = put_pred_on_image(image, pred_instance, names, score_thr=score_thr, color=color, no_text=no_text)
    return image
