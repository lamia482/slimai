import torch
from .visulizer import Visualizer
from ..utils.scale_image import to_batch_numpy_image
from ..utils.visualize import square_imgs
from ..help_build import VISUALIZERS


@VISUALIZERS.register_module("MILVisualizer")
class MILVisualizer(Visualizer):
  def _visualize(self, image, target, output, class_names):
    patch_imgs = image
    label = target["label"]
    pred_score = output["scores"]
    pred_label = output["labels"]
    atten_indices = output["atten_indices"]
    atten_scores = output["atten_scores"]
    atten_patches = to_batch_numpy_image(torch.stack([patch_imgs[index] for index in atten_indices]))
    vis = square_imgs(atten_patches)
    return vis
