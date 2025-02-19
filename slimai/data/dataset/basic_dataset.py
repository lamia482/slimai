import numpy as np
import mmcv
import mmengine
import torch
from slimai.helper.help_utils import print_log
from slimai.helper.help_build import DATASETS, build_transform, build_loader, build_source


__all__ = ["BasicDataset"]


@DATASETS.register_module()
class BasicDataset(torch.utils.data.Dataset):
  version = "version"
  signature = "signature"
  collect_keys = ["indice", "image"]

  def __init__(self, dataset, 
               *, 
               std_func=None, 
               transform=None, 
               loader=None, 
               to_rgb=True, 
               desc=None, 
               max_sample_num=None,
               **kwargs):
    self.dataset_file = "Not file"
    if isinstance(dataset, str):
      self.dataset_file = dataset
      dataset = mmengine.load(dataset)
    elif isinstance(dataset, dict):
      dataset = build_source(dataset)
      dataset = dataset()
      
    if std_func is not None:
      assert (
        callable(std_func)
      ), "Standard function must be callable"
      dataset = std_func(dataset)

    assert (
      isinstance(dataset, dict) and {"files", }.issubset(set(dataset.keys()))
    ), "Dataset must be a dictionary at least with keys: `files`, but got: {}".format(dataset.keys())

    self.dataset = dataset

    files = dataset.pop("files")
    self.version = dataset.pop(self.version, None)
    self.signature = dataset.pop(self.signature, None)

    self.files = files

    self.transform = build_transform(transform)
    
    self.to_rgb = to_rgb

    self.loader = build_loader(loader)

    self.desc = desc or "No specific description."

    if max_sample_num is None:
      self.max_sample_num = None
      self.length = len(files)
    else:
      self.max_sample_num = max_sample_num
      self.length = min(max_sample_num, len(files))

    print_log(f"Dataset {self}", level="INFO")
    return

  def __getitem__(self, item):
    data = self.select_sample(item)

    # check data keys
    assert (
      set(self.collect_keys).issubset(set(list(data.keys())))
    ), f"Collect key({self.collect_keys}) must all contained in data, but got: {list(data.keys())}"

    return data

  def select_sample(self, item):
    file = self.files[item]
    image = self.loader(file)
    if image is None:
      print_log(f"Image '{file}' is None, select another sample randomly", level="WARNING")
      item = np.random.randint(0, len(self))
      return self.select_sample(item)

    if self.to_rgb:
      image = mmcv.bgr2rgb(image)

    data = dict(indice=item, image=image)  
    data = self.transform(data)
    
    return data

  def __str__(self):
    repr_str = f"Total {len(self)} samples(selected from {len(self.files)} samples with max_sample_num: '{self.max_sample_num}')\n"
    repr_str += f"\tWith Signature: {self.signature}\n"
    repr_str += f"\tDataset file: {self.dataset_file}\n"
    repr_str += f"\tDescription: {self.desc}\n"
    repr_str += f"\tVersion: {self.version}\n"
    repr_str += f"\tTransform: {self.transform}\n"
    return repr_str
  __repr__=__str__
  
  def __len__(self):
    return self.length
  