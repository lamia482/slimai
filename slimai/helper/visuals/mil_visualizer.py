import torch
import cv2
import mmengine
from pathlib import Path
from .visulizer import Visualizer
from ..utils.scale_image import to_batch_numpy_image
from ..utils.visualize import square_imgs, pad_image, hstack_imgs
from ..help_build import VISUALIZERS


@VISUALIZERS.register_module()
class MILVisualizer(Visualizer):
  def _visualize(self, image, target, output, class_names):
    if isinstance(image, (str, Path)):
      image = mmengine.load(image) # cache from meta.visual_file
      image = image["meta"]["visual_image"]

    patch_imgs = image
    label = target["label"]
    pred_score = output["scores"]
    pred_label = output["labels"]
    
    title = f"GT: {class_names[label]}, Pred: {class_names[pred_label]}, Score: {pred_score:.4f}"

    def _visualize_atten(_atten_scores, _atten_patches, _title, _color):
      raw_size, font_scale, font_thickness = 60, 0.5, 2
      _atten_patches = list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2BGR), _atten_patches))
      for atten_score, atten_patch in zip(_atten_scores, _atten_patches):
        cv2.putText(atten_patch, f"Atten: {atten_score:.6f}", (0, raw_size//2), 
                  cv2.FONT_HERSHEY_SIMPLEX, font_scale*1.5, _color, font_thickness)
      vis = square_imgs(_atten_patches)
      vis = pad_image(vis, up=raw_size)
      cv2.putText(vis, _title, ((vis.shape[1] - len(_title))//3, raw_size//2), 
                  cv2.FONT_HERSHEY_SIMPLEX, font_scale*2, (0, 255, 0), font_thickness)
      return vis

    topk_atten_indices = output["topk_atten_indices"]
    topk_atten_scores = output["topk_atten_scores"]
    topk_atten_patches = to_batch_numpy_image(torch.stack([patch_imgs[index] for index in topk_atten_indices]))

    topk_vis = _visualize_atten(topk_atten_scores, topk_atten_patches, f"TOPK of {title}", (0, 0, 255))

    tailk_atten_indices = output["tailk_atten_indices"]
    tailk_atten_scores = output["tailk_atten_scores"]
    tailk_atten_patches = to_batch_numpy_image(torch.stack([patch_imgs[index] for index in tailk_atten_indices]))
    tailk_vis = _visualize_atten(tailk_atten_scores, tailk_atten_patches, f"TAILK of {title}", (255, 0, 0))

    vis = hstack_imgs([topk_vis, tailk_vis])
    return vis
