from mmengine.structures import BaseDataElement


class DataSample(BaseDataElement):
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
