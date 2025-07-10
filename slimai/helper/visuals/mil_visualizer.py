import torch
import numpy as np
import cv2
from .visulizer import Visualizer
from ..utils.scale_image import to_batch_numpy_image
from ..utils.visualize import square_imgs, pad_image
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

    raw_size, font_scale, font_thickness = 60, 0.5, 2
    for atten_score, atten_patch in zip(atten_scores, atten_patches):
      cv2.putText(atten_patch, f"Atten: {atten_score:.6f}", (0, raw_size//2), 
                  cv2.FONT_HERSHEY_SIMPLEX, font_scale*1.5, (255, 0, 0), font_thickness)
    
    vis = square_imgs(atten_patches)
    vis = pad_image(vis, up=raw_size)
    title = f"GT: {class_names[label]}, Pred: {class_names[pred_label]}, Score: {pred_score:.4f}"
    cv2.putText(vis, title, ((vis.shape[1] - len(title))//3, raw_size//2), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale*2, (0, 0, 255), font_thickness)
    return vis
