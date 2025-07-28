import torchvision.datasets as datasets
from typing import Dict, List
from slimai.helper.help_build import SOURCES


@SOURCES.register_module()
class TorchSource(object):
  def __init__(self, dataset, *args, **kwargs):
    self.dataset = dataset
    self.args = args
    self.kwargs = kwargs
    return

  def __call__(self) -> Dict[str, List[str]]:
    dataset = self.load_dataset()
    return dict(
      files=dataset.data.numpy(), 
      class_names=dataset.classes,
      annotations=dict(label=dataset.targets.tolist()),
    )

  def load_dataset(self):
    dataset = getattr(datasets, self.dataset)
    dataset = dataset(*self.args, **self.kwargs)
    return dataset
