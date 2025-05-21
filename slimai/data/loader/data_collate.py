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

  def __init__(self):
    return
  
  def __call__(self, original_batch):
    assert (
      isinstance(original_batch, list) and len(original_batch) > 0 and 
      all(map(lambda x: isinstance(x, dict), original_batch)) and 
      all(map(lambda x: original_batch[0].keys() == x.keys(), original_batch))
    ), "original_batch must be a list of dicts with the same keys"

    keys = original_batch[0].keys()
    assert (
      self.image_key in keys
    ), "image_key must be in the keys of the batch"
    
    images = [v.pop(self.image_key) for v in original_batch if self.image_key in v]
    labels = [v.pop(self.label_key) for v in original_batch if self.label_key in v]
    instances = [v.pop(self.instance_key) for v in original_batch if self.instance_key in v]
    masks = [v.pop(self.mask_key) for v in original_batch if self.mask_key in v]
    texts = [v.pop(self.text_key) for v in original_batch if self.text_key in v]

    data = default_collate(original_batch)

    images = self.process_image(images)
    whwh = torch.stack([torch.as_tensor(image.shape[-2:] * 2) for image in images])
    width, height = whwh.unbind(dim=1)[:2]
    data["width"], data["height"] = width, height

    labels = self.process_label(labels)
    instances = self.process_instance(instances)
    masks = self.process_mask(masks)
    texts = self.process_text(texts)

    data.update({
      k: v for k, v in zip(
        [self.image_key, self.label_key, self.instance_key, self.mask_key, self.text_key],
        [images, labels, instances, masks, texts]
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
      torch.from_numpy(np.array([t["category_id"] for t in tgt], dtype="int64")).reshape(-1)
      for tgt in instance_list
    ]
    bbox_targets = [
      torch.from_numpy(np.array([t["bbox"] for t in tgt], dtype="float32").reshape(-1, 4))
      for tgt in instance_list
    ]
    mask_targets = [
      torch.from_numpy(np.array([t["segmentation"] for t in tgt], dtype="int64").reshape(-1, 2))
      for tgt in instance_list
    ]

    return dict(
      labels=cls_targets, 
      bboxes=bbox_targets, 
      masks=mask_targets, 
    )
  
  def process_mask(self, mask_list):
    if len(mask_list) == 0:
      return None
    cls_targets = [
      torch.from_numpy(np.array([t["category_id"] for t in tgt], dtype="int64")).reshape(-1)
      for tgt in mask_list
    ]
    mask_targets = [
      torch.from_numpy(np.array([t["segmentation"] for t in tgt], dtype="int64").reshape(-1, 2))
      for tgt in mask_list
    ]

    return dict(
      labels=cls_targets, 
      masks=mask_targets, 
    )
  
  def process_text(self, text_list):
    if len(text_list) == 0:
      return None
    return text_list
  