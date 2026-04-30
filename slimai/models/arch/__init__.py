from .base_arch import BaseArch
from .cls_arch import ClassificationArch
from .dino import DINO
from .mil import HierarchicalMIL, MIL
from .det_arch import DetectionArch


__all__ = [
  "BaseArch",
  "ClassificationArch",
  "DINO",
  "MIL",
  "HierarchicalMIL",
  "DetectionArch",
]
