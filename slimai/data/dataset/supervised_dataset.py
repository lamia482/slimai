import numpy as np
import mmcv
import mmengine
import torch
from slimai.helper.help_utils import print_log
from slimai.helper.help_build import DATASETS, TRANSFORMS, build_transform, compose_components
from .dataset_checker import DatasetChecker
from .basic_dataset import BasicDataset


__all__ = ["SupervisedDataset"]


""" Expected dataset format:
{
  "version": "1.0",
  "signature": "12345678",
  "class_names": ["class1", "class2", ...],
  "files": [
    "path/to/image1.jpg",
    "path/to/image2.jpg",
    ...
  ],
  "annotations": {
    "label": [0, 1, 2, ...],
    "mask": [[0, 1, 2, ...], ....],
    "instance": [{"bbox": [xmin,ymin,xmax,ymax], "category_id": 0, "segmentation": [[0, 1, 2, ...], ....]}, ....],
    "text": ["text1", "text2", ...]
  }
}
"""
@DATASETS.register_module()
class SupervisedDataset(BasicDataset):
  def __init__(self, *args, 
               ann_keys=["label", "mask", "instance", "text"], 
               **kwargs):
    super().__init__(*args, **kwargs)
    dataset = self.dataset
    files = self.files

    assert (
      isinstance(dataset, dict) and {"class_names", "files", "annotations"}.issubset(set(dataset.keys()))
    ), "Dataset must be a dictionary at least with keys: `class_names`, `files`, `annotations`, but got: {}".format(dataset.keys())

    class_names = dataset.pop("class_names")
    annotations = dataset.pop("annotations")

    self.class_names = class_names
    if isinstance(annotations, str):  
      annotations = mmengine.load(annotations)
    
    DatasetChecker.check_annotations(annotations, ann_keys, len(files))

    self.collect_keys = sorted(self.collect_keys + ann_keys)
    
    self.annotations = annotations.copy()
    self.ann_keys = ann_keys

    print_log(f"Dataset {self}", level="INFO")
    return

  def load_extra_keys(self, data, index):
    for key in self.ann_keys:
      data[key] = self.annotations[key][index]
    return data

  def __str__(self):
    repr_str = super().__str__()
    repr_str += f"\tCLASS NAMES: {self.class_names}\n"
    repr_str += f"\tHas Ann keys: {self.ann_keys}\n"
    return repr_str
  __repr__=__str__
  