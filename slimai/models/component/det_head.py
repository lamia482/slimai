import torch
from slimai.helper.help_build import MODELS
from slimai.helper.help_utils import print_log


@MODELS.register_module()
class DetectionHead(torch.nn.Module):

  MLP: torch.nn.Module = MODELS.get("MLP") # type: ignore

  def __init__(self, 
               *, 
               input_dim, 
               num_classes, 
               num_layers=3, 
               dropout=0.1,
               ):
    super().__init__()
    self.input_dim = input_dim
    self.num_classes = num_classes
    self.num_classes_with_bg = num_classes + 1 # +1 for background
    print_log("Expand num_classes from {} to {}, where final channel represents as 'background'".format(
      self.num_classes, self.num_classes_with_bg), level="WARNING")
    
    self.cls_head = self.MLP(input_dim=input_dim, output_dim=self.num_classes_with_bg,
                             hidden_dim=input_dim, bottleneck_dim=input_dim, 
                             n_layer=num_layers, act="relu", norm="layer_norm", dropout=dropout)
    self.bbox_head = self.MLP(input_dim=input_dim, output_dim=4, hidden_dim=input_dim, 
                              bottleneck_dim=input_dim, n_layer=num_layers, act="relu", norm="layer_norm", dropout=dropout)
    return
  
  def forward(self, x):
    cls_logits = self.cls_head(x) # [B, Q, C+1] # C+1 for background, not used in focal loss
    bbox_logits = self.bbox_head(x) # [B, Q, 4] where 4 indicates [cx, cy, w, h]
    return cls_logits, bbox_logits
