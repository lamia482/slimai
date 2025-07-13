import torch
import numpy as np
from mmengine.structures import BaseDataElement
from .utils.select import check_list_dict_keys


class DataSample(BaseDataElement):  

  def to(self, *args, **kwargs):
    def _to_(value):
      if isinstance(value, list):
        return list(map(_to_, value))
      elif isinstance(value, tuple):
        return tuple(map(_to_, value))
      elif isinstance(value, dict):
        return {k: _to_(v) for k, v in value.items()}
      
      if hasattr(value, "to"):
        return value.to(*args, **kwargs)
      return value
    
    new_data = self.new()
    for k, v in self.items():
      v = _to_(v)
      data = {k: v}
      new_data.set_data(data)
    return new_data

  def cpu(self):
    return self.to("cpu")

  def cuda(self):
    return self.to("cuda")
  
  def musa(self):
    return self.to("musa")
  
  def npu(self):
    return self.to("npu")
  
  def mlu(self):
    return self.to("mlu")
  
  def split_as_list(self):
    """
    Split the DataSample into a list of DataSample.
    The DataSample is a dictionary of values, and each value can be 
    1. a number
    2. an element in type of list, tuple, tensor or array, and have the same batch size
    3. a dict of elements in type of list, tuple, tensor or array, and have the same batch size
    4. element is expected to have shape dim in [B] or [B, ...]
    This function splits the DataSample into a list of DataSample, where each DataSample
    contains non-batched data, and each single data split from batched data.
    """

    # Separate values into iterables and non-iterables
    batch_size = []
    non_iterable_data = {}
    array_keys, array_values = [], []
    dict_keys, dict_values = [], []
    
    for key, value in self.items():
      if isinstance(value, (list, tuple, np.ndarray, torch.Tensor)):
        array_keys.append(key)
        array_values.append(value)
        batch_size.append(len(value))
      elif isinstance(value, dict):
        assert (
          all(map(lambda x: (not isinstance(x, dict)), value.values()))
        ), "dict_values cannot be a dict"
        dict_keys.append(key)
        dict_values.append(value)
        batch_size.extend([len(v) for v in value.values()])
      else:
        non_iterable_data[key] = value

    assert (
      len(set(batch_size)) == 1
    ), f"all values must have the same batch size, but got: {batch_size} for {self.keys()}"
    batch_size = batch_size[0]

    result_list = []
    for i in range(batch_size):
      result = dict()
      for key, value in self.items():
        if key in array_keys:
          result[key] = value[i]
        elif key in dict_keys:
          result[key] = {k: v[i] for k, v in value.items()}
        else:
          result[key] = value

      result = DataSample(metainfo=self.metainfo, **result)
      result_list.append(result)
    return result_list

  @classmethod
  def merge_from_list(cls, list_data):
    """
    Merge a list of DataSample into a single DataSample.
    The DataSample is a dictionary of values, and each value can be 
    1. a number in type of torch.Tensor
    2. an element in type of torch.Tensor, and may have different sizes
    3. a dict of elements in type of torch.Tensor, and may have different sizes
    This function merges a list of DataSample into a single DataSample key by key.
    """
    result = dict()
    if not list_data:
      return cls()
    
    # Verify all elements have the same keys
    first_data = list_data[0]
    first_keys = check_list_dict_keys(list_data)
    
    def concat(inputs):
      if isinstance(inputs, list):
        assert (
          len(inputs) > 0
        ), "inputs must be a non-empty list"
        if isinstance(inputs[0], dict): # merge every key in dict
          keys = check_list_dict_keys(inputs)
          return {k: concat([d[k] for d in inputs]) for k in keys}
        
        if not all(map(lambda x: isinstance(x, torch.Tensor), inputs)):
          return inputs
        
        shape = list(set(list(map(lambda v: v.shape, inputs))))
        if len(shape) == 1: # same shape
          output = torch.stack(inputs, dim=0)
        else: # different shape
          output = inputs
        return output

      elif isinstance(inputs, dict):
        return {k: concat(v) for k, v in inputs.items()}
      
      raise ValueError(f"Expected a list or dict, but got {type(inputs)}")
      
    # expect to have all values as torch.Tensor
    for key in first_keys:
      elem_list = [getattr(data, key) for data in list_data]
      result[key] = concat(elem_list)

    return DataSample(metainfo=first_data.metainfo, **result)
