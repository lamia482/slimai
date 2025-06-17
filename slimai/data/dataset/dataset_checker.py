import hashlib
import numpy as np


__all__ = ["DatasetChecker"]


class DatasetChecker(object):
  @classmethod
  def check_annotations(cls, annotations, ann_keys, length):
    cls.check_signature(annotations)
    for key in ann_keys:
      getattr(cls, f"check_{key}")(annotations.get(key, None), length)
    return

  @classmethod
  def check_signature(cls, annotations):
    annotations = annotations.copy()
    signature = annotations.pop("signature", None)
    if signature is not None:
      dataset_signature = hashlib.md5(str(annotations).encode("utf8")).hexdigest()[:8]
      assert (
        signature == dataset_signature
      ), "Dataset signature mismatch"
    return

  @classmethod
  def check_label(cls, labels, length):
    labels = cls.check_length("labels", labels, length)
    labels = cls.check_element_type("labels", labels, int)
    return

  @classmethod
  def check_mask(cls, masks, length):
    masks = cls.check_length("masks", masks, length)
    masks = cls.check_element_type("masks", masks, np.ndarray)
    return

  @classmethod
  def check_instance(cls, instances, length):
    cls.check_length("instances", instances, length)
    instances = cls.check_element_type("instances", instances, list)
    assert (
      all(all(isinstance(item, dict) and 'bbox' in item and 'category_id' in item and 'segmentation' in item for item in i) for i in instances if len(i) > 0)
    ), "All elements in instances must be lists containing dictionaries with keys 'bbox', 'category_id', and 'segmentation'"
    return

  @classmethod
  def check_text(cls, texts, length):
    texts = cls.check_length("texts", texts, length)
    texts = cls.check_element_type("texts", texts, str)
    return
  
  @classmethod
  def check_length(cls, key, value, length):
    assert (
      value is not None
    ), f"Key: '{key}' must be provided"

    assert (
      (len(value) == length) and (len(value) > 0)
    ), f"The length of key: '{key}' must be equal to the length of files, but got: {len(value)} vs {length}"
    return value
  
  @classmethod
  def check_element_type(cls, key, value, element_type):
    assert (
      all(isinstance(v, element_type) for v in value)
    ), f"All elements in key: '{key}' must be of type: {element_type}, please check your dataset"
    return value