import numpy as np
import torch
from torch.utils.data import default_collate
from slimai.helper.help_build import LOADERS


@LOADERS.register_module()
class DataCollate():
  image_key = "image"
  label_key = "label"
  instance_key = "instance"
  mask_key = "mask"
  text_key = "text"
  latency_key = "latency"
  meta_key = "meta"

  def __init__(self):
    return
  
  def __call__(self, original_batch):
    assert (
      isinstance(original_batch, list) and len(original_batch) > 0 and 
      all(map(lambda x: isinstance(x, dict), original_batch)) and 
      all(map(lambda x: original_batch[0].keys() == x.keys(), original_batch))
    ), "original_batch must be a list of dicts with the same keys"

    keys = original_batch[0].keys()
    assert ( # make sure developer has properly handle with image_key
      self.image_key in keys
    ), "image_key must be in the keys of the batch"
    
    # pop attributes from original_batch for custom processing
    images = [v.pop(self.image_key) for v in original_batch if self.image_key in v]
    labels = [v.pop(self.label_key) for v in original_batch if self.label_key in v]
    instances = [v.pop(self.instance_key) for v in original_batch if self.instance_key in v]
    masks = [v.pop(self.mask_key) for v in original_batch if self.mask_key in v]
    texts = [v.pop(self.text_key) for v in original_batch if self.text_key in v]
    metas = [v.pop(self.meta_key) for v in original_batch if self.meta_key in v]

    # default collate for other keys
    data = default_collate(original_batch)

    # custom processing
    images = self.process_image(images)
    if (images is not None) and len(images) > 0:
      whwh = torch.stack([torch.as_tensor(image.shape[-2:] * 2) for image in images])
      width, height = whwh.unbind(dim=1)[:2]
      data["width"], data["height"] = width, height
    labels = self.process_label(labels)
    instances = self.process_instance(instances)
    masks = self.process_mask(masks)
    texts = self.process_text(texts)
    metas = self.process_meta(metas)

    # update data with custom processed keys
    data.update({
      k: v for k, v in zip(
        [self.image_key, self.label_key, self.instance_key, self.mask_key, self.text_key, self.meta_key],
        [images, labels, instances, masks, texts, metas]
      ) if v is not None
    })
    
    return data
  
  def process_image(self, image_list):
    if len(image_list) == 0:
      return None
    return torch.stack(image_list)
  
  def process_label(self, label_list): # [B]
    if len(label_list) == 0:
      return None
    labels = torch.stack(list(map(torch.as_tensor, label_list)))
    labels = labels.to(torch.int64)
    return labels
  
  def process_instance(self, instance_list):
    if len(instance_list) == 0:
      return None
    cls_targets = [
      tgt["labels"] for tgt in instance_list
    ]
    bbox_targets = [
      tgt["bboxes"] for tgt in instance_list
    ]
    # mask_targets = [
    #   tgt["mask"] for tgt in instance_list
    # ]

    return dict(
      labels=cls_targets, 
      bboxes=bbox_targets, 
      # masks=mask_targets, 
    )
  
  def process_mask(self, mask_list):
    if len(mask_list) == 0:
      return None

    raise NotImplementedError("Mask is not supported yet")
    cls_targets = [
      tgt["labels"] for tgt in mask_list
    ]
    mask_targets = [
      tgt["mask"] for tgt in mask_list
    ]

    return dict(
      labels=cls_targets, 
      masks=mask_targets, 
    )
  
  def process_text(self, text_list):
    if len(text_list) == 0:
      return None
    
    raise NotImplementedError("Text is not supported yet")
    return text_list

  def process_meta(self, meta_list):
    if len(meta_list) == 0:
      return None
    keys = list(meta_list[0].keys())
    metas = {
      k: [m[k] for m in meta_list]
      for k in keys
    }
    return metas
  