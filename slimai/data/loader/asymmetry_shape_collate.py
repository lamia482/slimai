from torch.utils.data import default_collate
from slimai.helper.help_build import LOADERS
from .data_collate import DataCollate


@LOADERS.register_module()
class AsymmetryShapeCollate(DataCollate):
  def process_image(self, image_list):
    return image_list
