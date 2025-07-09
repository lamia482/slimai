from .basic_classification_loss import BasicClassificationLoss
from .dino_loss import DINOLoss
from .mil_loss import MILLoss
from .detr_loss import DETRLoss
from .focal_loss import FocalLoss


__all__ = [
  "BasicClassificationLoss",
  "DINOLoss",
  "MILLoss",
  "DETRLoss",
  "FocalLoss",
]
