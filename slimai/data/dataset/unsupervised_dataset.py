from .basic_dataset import BasicDataset
from slimai.helper.help_build import DATASETS


@DATASETS.register_module()
class UnSupervisedDataset(BasicDataset):
  pass
