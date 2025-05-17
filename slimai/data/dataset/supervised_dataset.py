import numpy as np
import mmcv
import mmengine
import torch
from slimai.helper.help_utils import print_log
from slimai.helper.help_build import DATASETS, TRANSFORMS, build_transform, compose_components
from .dataset_checker import DatasetChecker
from .basic_dataset import BasicDataset
from .sample_strategy import SampleStrategy


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
    "label": [c1, c2, ...], 
    "mask": [
      [
        {
          "category_id": 0,
          "segmentation": [x1, y1, x2, y2, ..., xn, yn] # polygon in shape of [2N], can reshape to [N, 2], where 2 is (x, y)
        }, 
        ... # other masks of other images
      ], 
    ],
    "instance": [
      [
        {
          "bbox": [xmin, ymin, xmax, ymax], # bbox in shape of [4]
          "category_id": 0, # category id
          "segmentation": [x1, y1, x2, y2, ..., xn, yn] # segmentation in shape of [2N], can reshape to [N, 2], where 2 is (x, y)
        }, 
        ... # other annotations in the same image
      ], 
      ... # list of annotations of other images
    ],
    "text": [
      "text1", # text of image
      ... # list of texts of other images
    ]
  }
}
"""
@DATASETS.register_module()
class SupervisedDataset(BasicDataset):
  def __init__(self, *args, 
               class_names=None, 
               ann_keys=["label", "mask", "instance", "text"], 
               sample_strategy=None, 
               filter_empty=False, 
               **kwargs):
    super().__init__(*args, **kwargs)
    dataset = self.dataset
    files = self.files

    if class_names is not None:
      dataset["class_names"] = class_names
    assert (
      isinstance(dataset, dict) and {"class_names", "annotations"}.issubset(set(dataset.keys()))
    ), "Dataset must be a dictionary at least with keys: `class_names`, `annotations`, but got: {}".format(dataset.keys())

    class_names = dataset.pop("class_names")
    self.class_names = class_names

    annotations = dataset.pop("annotations")
    if isinstance(annotations, str):  
      annotations = mmengine.load(annotations)
    DatasetChecker.check_annotations(annotations, ann_keys, len(files))

    # filter empty annotations
    non_empty_indices = []
    for index in range(len(files)):
      for key in ann_keys:
        anns = annotations[key][index]
        if not isinstance(anns, (list, tuple)):
          continue
        if len(anns) == 0:
          break
      else:
        non_empty_indices.append(index)
    if filter_empty:
      files = [files[i] for i in non_empty_indices]
      annotations = {key: [annotations[key][i] for i in non_empty_indices] for key in ann_keys}

    self.collect_keys = sorted(self.collect_keys + ann_keys)
    
    self.files = files
    self.indices = list(range(len(files)))
    self.annotations = annotations.copy()
    self.ann_keys = ann_keys

    self.sample_strategy = sample_strategy
    if sample_strategy is not None:
      # update indices by sample strategy, change indices from list to dict[sampled id -> file id]
      self.indices = SampleStrategy.update_indices(annotations, ann_keys, 
                                                   sample_strategy, self.indices)
    return
    
  def load_extra_keys(self, data, index):
    for key in self.ann_keys:
      data[key] = self.annotations[key][index]
    return data

  def __str__(self):
    repr_str = super().__str__()
    repr_str += f"\tCLASS NAMES: {self.class_names}\n"
    repr_str += f"\tHas Ann keys: {self.ann_keys} with {len(self.files)} filtered samples\n"
    repr_str += f"\tSample strategy: {self.sample_strategy}\n"
    return repr_str
  __repr__=__str__
  