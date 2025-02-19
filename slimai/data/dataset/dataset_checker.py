import hashlib
import numpy as np


__all__ = ["DatasetChecker"]


class DatasetChecker(object):
  @classmethod
  def check_annotations(cls, annotations, ann_keys, length):
    cls.check_signature(annotations)
    for key in ann_keys:
      getattr(cls, f"check_{key}")(annotations.get(key, None), length)
    return True

  @classmethod  
  def check_signature(cls, annotations):
    annotations = annotations.copy()
    signature = annotations.pop("signature", None)
    if signature is not None:
      dataset_signature = hashlib.md5(str(annotations).encode("utf8")).hexdigest()[:8]
      assert (
        signature == dataset_signature
      ), "Dataset signature mismatch"
    return True

  @classmethod
  def check_label(cls, labels, length):
    assert (
      labels is not None
    ), "Labels must be provided"

    assert (
      (len(labels) == length) and (len(labels) > 0)
    ), "The length of labels must be equal to the length of files"

    assert (
      all(isinstance(l, int) for l in labels)
    ), "All elements in labels must be integers"
    return True

  @classmethod
  def check_mask(cls, masks, length):
    assert (
      masks is not None
    ), "Masks must be provided"

    assert (
      (len(masks) == length) and (len(masks) > 0)
    ), "The length of masks must be equal to the length of files"

    assert (
      all(isinstance(m, np.ndarray) for m in masks)
    ), "All elements in masks must be numpy arrays"
    return True

  @classmethod
  def check_instance(cls, instances, length):
    assert (
      instances is not None
    ), "Instances must be provided"

    assert (
      (len(instances) == length) and (len(instances) > 0)
    ), "The length of instances must be equal to the length of files"

    assert (
      all((isinstance(i, dict) and set(i.keys()) == {"bbox", "category_id", "segmentation"}) for i in instances)
    ), "All elements in instances must be dictionaries with keys 'bbox', 'category_id', and 'segmentation'"
    return True

  @classmethod
  def check_text(cls, texts, length):
    assert (
      texts is not None
    ), "Texts must be provided"

    assert (
      (len(texts) == length) and (len(texts) > 0)
    ), "The length of texts must be equal to the length of files"

    assert (
      all(isinstance(t, str) for t in texts)
    ), "All elements in texts must be strings"
    return True