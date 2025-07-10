from slimai.helper.help_build import LOADERS
from .asymmetry_shape_collate import AsymmetryShapeCollate


@LOADERS.register_module()
class MILCollate(AsymmetryShapeCollate):
  def process_image(self, image_list):
    if None in image_list:
      return []
    return super().process_image(image_list)
