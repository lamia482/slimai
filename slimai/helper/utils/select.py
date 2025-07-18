from typing import Union, List, Dict, Any, Callable
import torch


__all__ = [
  "recursive_select",
  "recursive_apply",
]

def recursive_select(
    inputs: Union[List, Dict, Any], 
    ids: Union[List[int], int]
  ) -> Union[List, Dict, Any]:
  """
  Recursively select elements from a list, tuple, or dictionary.

  Args:
    inputs: The input list, tuple, or dictionary.
    ids: The indices to select, or a single index.

  Returns:
    The selected elements.
  """
  def _get_item_by_ids(_inputs, _ids):
    if isinstance(_ids, (list, tuple)):
      return [_get_item_by_ids(_inputs, _id) for _id in _ids]
    return _inputs[_ids]
  
  if isinstance(inputs, list):
    return _get_item_by_ids(inputs, ids)
  elif isinstance(inputs, torch.Tensor):
    return inputs[ids]
  elif isinstance(inputs, tuple):
    return tuple(map(lambda x: recursive_select(x, ids), inputs))
  elif isinstance(inputs, dict):
    return {k: recursive_select(v, ids) for k, v in inputs.items()}
  return inputs

def recursive_apply(
  func: Callable,
  inputs: Union[List, Dict, Any],
) -> Union[List, Dict, Any]:
  if isinstance(inputs, list):
    return [recursive_apply(func, item) for item in inputs]
  elif isinstance(inputs, tuple):
    return tuple(map(lambda x: recursive_apply(func, x), inputs))
  elif isinstance(inputs, dict):
    return {k: recursive_apply(func, v) for k, v in inputs.items()}
  return func(inputs)

def chunks(lst, n):
  """Yield successive n-sized chunks from lst."""
  for i in range(0, len(lst), n):
    yield lst[i:i + n]

def check_list_dict_keys(kv_item_list: List[Dict]):
  first_keys = set(kv_item_list[0].keys())
  for kv_item in kv_item_list[1:]:
    current_keys = set(kv_item.keys())
    assert (
      first_keys == current_keys
    ), f"All DataSample elements must have the same keys, but got {first_keys} and {current_keys}"
  return first_keys
