import hashlib
import numpy as np
import mmcv
import mmengine
import torch
from loguru import logger
from slimai.helper.help_build import DATASETS, build_transform


__all__ = ["SupervisedDataset", "DatasetChecker"]


@DATASETS.register_module()
class SupervisedDataset(torch.utils.data.Dataset):
  version = "version"
  signature = "signature"
  collect_keys = ["indice", "image"]

  def __init__(self, dataset, 
               *, 
               std_func=None, 
               ann_keys=["labels", "masks", "instances", "texts"],
               transform=None, 
               to_rgb=True, 
               desc=None, 
               max_sample_num=None,
               **kwargs):
    self.dataset_file = "Not file"
    if isinstance(dataset, str):
      self.dataset_file = dataset
      dataset = mmengine.load(dataset)
    if std_func is not None:
      assert (
        callable(std_func)
      ), "Standard function must be callable"
      dataset = std_func(dataset)

    assert (
      isinstance(dataset, dict) and {"class_names", "files", "annotations"}.issubset(set(dataset.keys()))
    ), "Dataset must be a dictionary at least with keys: `class_names`, `files`, `annotations`, but got: {}".format(dataset.keys())

    class_names = dataset.pop("class_names")
    files = dataset.pop("files")
    annotations = dataset.pop("annotations")
    self.version = dataset.pop(self.version, None)
    self.signature = dataset.pop(self.signature, None)

    self.class_names = class_names
    self.files = files
    if isinstance(annotations, str):  
      annotations = mmengine.load(annotations)
    
    DatasetChecker.check_annotations(annotations, ann_keys, len(files))

    self.collect_keys = sorted(self.collect_keys + ann_keys)
    
    self.annotations = annotations.copy()
    self.ann_keys = ann_keys

    self.transform = build_transform(transform)
    
    self.to_rgb = to_rgb

    self.desc = desc or "No specific description."

    if max_sample_num is None:
      self.max_sample_num = None
      self.length = len(files)
    else:
      self.max_sample_num = max_sample_num
      self.length = min(max_sample_num, len(self))

    logger.info(f"Dataset {self}")
    return

  def __getitem__(self, item):
    if self.version is None:
      data = self.select_sample(item)
    else:
      raise NotImplementedError("Versioned dataset is not implemented")

    # check data keys
    assert (
      set(self.collect_keys).issubset(set(list(data.keys())))
    ), f"Collect key({self.collect_keys}) must all contained in data, but got: {list(data.keys())}"

    return data

  def select_sample(self, item):
    file = self.files[item]
    image = mmcv.imread(file)
    if image is None:
      logger.warning(f"Image '{file}' is None, select another sample randomly")
      item = np.random.randint(0, len(self))
      return self.select_sample(item)

    if self.to_rgb:
      image = mmcv.bgr2rgb(image)

    data = dict(indice=item, image=image)  
    for key in self.ann_keys:
      data[key] = self.annotations[key][item]

    # TODO: transform ann data
    data = self.transform(data)
    return data

  def __str__(self):
    repr_str = f"Total {len(self)} samples(selected from {self.files} samples with max_sample_num: '{self.max_sample_num}')\n"
    repr_str += f"\tWith Signature: {self.signature}\n"
    repr_str += f"\tDataset file: {self.dataset_file}\n"
    repr_str += f"\tCLASS NAMES: {self.class_names}\n"
    repr_str += f"\tDescription: {self.desc}\n"
    repr_str += f"\tVersion: {self.version}\n"
    repr_str += f"\tHas Ann keys: {self.ann_keys}\n"
    repr_str += f"\tTransform: {self.transform}\n"
    return repr_str
  __repr__=__str__
  
  def __len__(self):
    return self.length


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
  def check_labels(cls, labels, length):
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
  def check_masks(cls, masks, length):
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
  def check_instances(cls, instances, length):
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
  def check_texts(cls, texts, length):
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


def split(dataset, split_rule, desc=None, seed=10482):
  assert (
    isinstance(split_rule, dict) and len(split_rule) > 0
  ), "Split rule must be a non-empty dictionary, phase:rule[file,ratio,indice]"

  phases, rules = zip(*split_rule.items())
  assert (
    all(map(type, rules))
  ), "type of rule must be the same."

  final_indices_list = []

  if isinstance(rules[0], (int, float)):
    ratios = list(map(lambda r: r/np.sum(rules), rules))
    random_indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
    for ratio in ratios:
      split_indices = random_indices[:int(len(random_indices)*ratio)]
      random_indices = random_indices[int(len(random_indices)*ratio):]
      final_indices_list.append(split_indices)
  elif isinstance(rules[0], (tuple, list)) and isinstance(rules[0][0], int):
    for rule in rules:
      final_indices_list.append(rule)
  elif isinstance(rules[0], (tuple, list)) and isinstance(rules[0][0], str):
    for rule in rules:
      final_indices_list.append(dataset.files.index(rule))
  else:
    raise ValueError(f"Unsupported rule type: {type(rules[0])}")

  def wrap_dataset(indices, ds_desc):
    return type(dataset)(dataset.class_names, [dataset.files[i] for i in indices], 
                      annotations=dict(version=dataset.version, signature=dataset.signature, 
                                      labels=dataset.labels[indices]), 
                      transform=dataset.transform, to_rgb=dataset.to_rgb, desc=ds_desc)
  split_datasets = {
    phase: wrap_dataset(indices, dataset.desc+f"<Random split as {phase} with seed: {seed}>")
    for phase, indices in zip(phases, final_indices_list)
  }
  return split_datasets
