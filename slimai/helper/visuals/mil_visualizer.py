import torch
import numpy as np
from PIL import Image, ImageDraw
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
    vis = square_imgs(atten_patches)

    raw_size = 30
    vis = pad_image(vis, up=raw_size)
    pil_image = Image.fromarray(vis)
    draw = ImageDraw.Draw(pil_image)
    title = f"GT: {class_names[label]}, Pred: {class_names[pred_label]}, Score: {pred_score:.4f}"
    length = draw.textlength(title)
    draw.text(((vis.shape[1] - length)//2, 0), title, fill=(255, 0, 0))
    vis = np.array(pil_image)[..., ::-1]
    return vis
