from mmengine.structures import BaseDataElement


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
  
  def split_as_list(self):
    keys = self.keys()
    values = self.values()

    value_lengths = [len(value) for value in values]
    assert (
      1 >= len(set(value_lengths))
    ), "All values must have the same length"

    batch_size = 0 if len(value_lengths) == 0 else value_lengths[0]

    return [
      DataSample(metainfo=self.metainfo, **{
        key: batch[i]
        for key, batch in zip(keys, values)
      })
      for i in range(batch_size)
    ]

class DataListElement(BaseDataElement):
  def __init__(self, values=None):
    if values is None:
      values = []
    wrapped_dict = {str(index): value for (index, value) in enumerate(values)}
    super().__init__(**wrapped_dict)
    return
