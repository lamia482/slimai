from slimai.helper.help_build import LOADERS
from .asymmetry_shape_collate import AsymmetryShapeCollate


@LOADERS.register_module()
class MILCollate(AsymmetryShapeCollate):
  pass
