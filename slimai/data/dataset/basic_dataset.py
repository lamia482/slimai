import hashlib
import numpy as np
import mmengine
import torch
import time
import cv2
from PIL import Image
from torchvision import tv_tensors
from pathlib import Path
from functools import partial
from slimai.helper.help_utils import print_log
from slimai.helper.help_build import DATASETS, build_transform, build_loader, build_source
from slimai.helper.common import CACHE_ROOT_DIR


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
               repeat=1,
               cache=False, 
               **kwargs):
    self.dataset_file = "Not file"

    cache_file = Path(CACHE_ROOT_DIR, "dataset", self.__class__.__name__, "{}.pkl".format(
      hashlib.md5("+".join(map(str, [dataset, desc])).encode(encoding="UTF-8")
    ).hexdigest()))

    if cache_file.exists() and cache:
      dataset = mmengine.load(cache_file)
      print_log(f"Dataset loaded from cache<{cache_file}>")
      self.dataset_file, dataset = dataset
    else:
      print_log(f"Building dataset from scratch")
      if isinstance(dataset, str):
        self.dataset_file = dataset
        dataset = mmengine.load(dataset)
      elif isinstance(dataset, dict):
        dataset_fn = build_source(dataset)
        dataset = dataset_fn()
    
      print_log(f"Build dataset done, save cache to: {cache_file}")
      mmengine.dump((self.dataset_file, dataset), cache_file)
      
    if std_func is not None:
      if isinstance(std_func, str):
        std_func = eval(std_func)
      elif isinstance(std_func, dict):
        raise NotImplementedError("Standard function must be callable")
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
    self.indices = list(range(len(files)))

    self.transform = build_transform(transform)
    
    self.to_rgb = to_rgb

    self.loader = build_loader(loader)

    self.desc = desc or "No specific description."

    self.max_sample_num = max_sample_num
    self.repeat = repeat
    return

  def __getitem__(self, item):
    item = item % self.length
    item = self.indices[item]
    data = self.select_sample(item)
    return data

  def select_sample(self, item):
    data_select_start_time = time.time()

    file = self.files[item]

    data_loader_start_time = time.time()
    image = self.loader(file)
    data_loader_latency = time.time() - data_loader_start_time
    
    if image is None:
      print_log(f"Image '{file}' is None, select another sample randomly", level="WARNING")
      item = np.random.randint(0, len(self))
      return self.select_sample(item)

    # convert image to PIL Image
    data_to_pil_start_time = time.time()
    to_pil_image = partial(self.to_pil_image, to_rgb=self.to_rgb)
    image = self.apply_kernel(to_pil_image, image)
    data_to_pil_latency = time.time() - data_to_pil_start_time

    data = dict(indice=item, image=image)
    data = self.load_extra_keys(data, index=item)

    # wrap data tv_tensors
    data_wrap_start_time = time.time()
    data = self.wrap_data(data)
    data_wrap_latency = time.time() - data_wrap_start_time

    data_transform_start_time = time.time()
    data = self.transform(data)
    data_transform_latency = time.time() - data_transform_start_time

    data_select_latency = time.time() - data_select_start_time

    # wrap data latency
    data["latency"] = dict(
      data_select_latency=data_select_latency,
      data_loader_latency=data_loader_latency,
      data_to_pil_latency=data_to_pil_latency,
      data_wrap_latency=data_wrap_latency,
      data_transform_latency=data_transform_latency,
    )
    return data
  
  def to_pil_image(self, image, to_rgb=True):
    """ convert image to PIL Image
    """
    if isinstance(image, np.ndarray):
      if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      pil_image = Image.fromarray(np.ascontiguousarray(image))
    else:
      pil_image = image

    assert (
      isinstance(pil_image, Image.Image)
    ), f"Image must be a PIL Image, but got: {type(pil_image)}"

    if to_rgb:
      pil_image = pil_image.convert("RGB")
        
    return pil_image
  
  def apply_kernel(self, kernel, data):
    if isinstance(data, (list, tuple)):
      return list(map(kernel, data))
    else:
      return kernel(data)
  
  def load_extra_keys(self, data, index):
    data["file"] = self.files[index]
    return data
  
  def wrap_data(self, data):
    # check data keys
    assert (
      set(self.collect_keys).issubset(set(list(data.keys())))
    ), f"Collect key({self.collect_keys}) must all contained in data, but got: {list(data.keys())}"
    
    data["indice"] = torch.tensor(data["indice"])
    data["image"] = self.apply_kernel(tv_tensors.Image, data["image"])
    return data

  @property
  def length(self):
    size = len(self.indices)
    if self.max_sample_num is not None:
      size = min(self.max_sample_num, size)
    return size

  def __str__(self):
    repr_str = f"Total {len(self)} samples(selected from {len(self.files)} samples with max_sample_num: '{self.max_sample_num}' with repeat: '{self.repeat}')\n"
    repr_str += f"\tWith Signature: {self.signature}\n"
    repr_str += f"\tDataset file: {self.dataset_file}\n"
    repr_str += f"\tDescription: {self.desc}\n"
    repr_str += f"\tVersion: {self.version}\n"
    repr_str += f"\tTransform: {self.transform}\n"
    return repr_str
  __repr__=__str__
  
  def __len__(self):
    return int(self.length * self.repeat)
  