from typing import Union, List, Dict, Any

__all__ = [
  "recursive_select",
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
  
  if isinstance(inputs, (list, tuple)):
    return _get_item_by_ids(inputs, ids)
  elif isinstance(inputs, dict):
    return {k: recursive_select(v, ids) for k, v in inputs.items()}
  return inputs
